"""Train one block of the v2 population: persons of a single (task, algorithm, disability) cell, vmapped in one
compiled program, each with its own key and preferences (population/manifest_v2.json).

It uses each algorithm's own trainer (make_train(..., dynamic_preferences=True), as the zoo-gen scripts do) with the
config from population/config.py. The zoo-gen scripts give every person of a block the same key and draw preferences
from the block's seed; here a person trains the same whatever block it is in.

Output, <out root>/<task>/<block>/ (refused if it exists; nothing is ever overwritten):
- zoo/: a v2 zoo shard with each person's robot and human (one team per person). Each config records POPULATION
  (person, block, key, git SHA) and the person's preferences.
- metrics.npz: the trainer's metrics per person (IPPO/MAPPO per update, MASAC per checkpoint).
- block.json: final task return per person and provenance, written last: a block is done when it exists.

Usage, from the checkout root:
  python -m population.run_block population/manifest_v2.json <task> <first>-<last> <out root> [--smoke] [-- overrides]
Hydra overrides (tiny budgets for tests) are only accepted with --smoke.
"""
import argparse
import copy
import datetime
import importlib
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

import assistax
from assistax.wrappers.aht import ZooManager
from population.config import compose
from population.manifest import PREF_KEYS, load_manifest

ROOT = Path(__file__).resolve().parents[1]
ZOO_VERSION = "v2"
TRAINERS = {  # module, its train() hyperparameters after the key, where the actor's parameters are
    "ippo": ("assistax.baselines.IPPO.ippo_ff_nps", ("LR", "ENT_COEF", "CLIP_EPS"), "train_state.params"),
    "mappo": ("assistax.baselines.MAPPO.mappo_ff_nps", ("LR", "ENT_COEF", "CLIP_EPS"), "train_state.actor.params"),
    "masac": ("assistax.baselines.MASAC.masac_ff_nps", ("POLICY_LR", "Q_LR", "ALPHA_LR", "TAU"),
              "train_states.actor.params"),
}


def block_name(task: str, alg: str, first: int, last: int) -> str:
    """A block's name, e.g. scratchitch-ippo-p000-014."""
    return f"{task}-{alg}-p{first:03d}-{last:03d}"


def check_cell(persons: Sequence[dict]) -> tuple[str, float, float]:
    """The block's (algorithm, restriction, strength); a block may not mix them, as the disability is fixed when the
    program compiles."""
    cells = {(p["algorithm"], p["restriction"], p["strength"]) for p in persons}
    if len(cells) != 1:
        raise ValueError(f"a block must share one algorithm and disability, got {sorted(cells)}")
    return cells.pop()


def pref_arrays(persons: Sequence[dict], config: dict) -> dict:
    """The persons' preference configs as generate_preference_configs returns them: their 7 sampled values plus the
    task's reward budget, overall weight and touch threshold."""
    pref, n = config["ENV_KWARGS"]["preference_rewards"], len(persons)
    out = {k: jnp.array([p["prefs"][k] for p in persons], jnp.float32) for k in PREF_KEYS}
    for k in ("reward_budget", "overall_weight", "touch_threshold"):
        out[k] = jnp.full((n,), pref[k], jnp.float32)
    return out


def _memory(compiled, stats: dict) -> dict:
    """XLA's memory analysis of a compiled program, plus the device's peak bytes in use when known."""
    mem = compiled.memory_analysis()
    return {"temp_bytes": getattr(mem, "temp_size_in_bytes", None),
            "argument_bytes": getattr(mem, "argument_size_in_bytes", None),
            "output_bytes": getattr(mem, "output_size_in_bytes", None), "peak_bytes": stats.get("peak_bytes_in_use")}


def train_block(task: str, persons: Sequence[dict], extra: Sequence[str] = (), compile_only: bool = False) -> dict:
    """Train the persons of one cell in one vmapped program. Returns the config (as make_train completed it), the
    env's agent ids, actor params and metrics with a leading person axis, each person's final task returns per env
    (the env's own reward: the preference reward is added outside the logging wrapper), timings and memory.
    compile_only stops after compiling and returns the config, compile time and memory analysis (to size blocks)."""
    alg, restriction, strength = check_cell(persons)
    config = compose(task, alg, restriction, strength, extra)
    module, hp_keys, params_at = TRAINERS[alg]
    train = importlib.import_module(module).make_train(config, save_train_state=False, dynamic_preferences=True)
    env_kwargs = {k: v for k, v in config["ENV_KWARGS"].items() if k != "preference_rewards"}
    agents = list(assistax.make(config["ENV_NAME"], **env_kwargs).agents)

    def run(keys, hparams, prefs):
        out = jax.vmap(train, in_axes=(0, *[None] * len(hparams), 0))(keys, *hparams, prefs)
        params = out["runner_state"]
        for attr in params_at.split("."):
            params = getattr(params, attr)
        return {"params": params, "metrics": out["metrics"],
                "final_returns": out["runner_state"].env_state.returned_episode_returns}

    args = (jnp.array([p["key"] for p in persons], jnp.uint32), tuple(config[k] for k in hp_keys),
            pref_arrays(persons, config))
    t0 = time.time()
    compiled = jax.jit(run).lower(*args).compile()
    t1 = time.time()
    if compile_only:
        return {"algorithm": alg, "config": config, "agents": agents, "compile_s": t1 - t0,
                "memory": _memory(compiled, {})}
    res = jax.block_until_ready(compiled(*args))
    t2 = time.time()
    return {"algorithm": alg, "config": config, "agents": agents, "params": res["params"],
            "metrics": jax.device_get(res["metrics"]), "final_returns": np.asarray(res["final_returns"]),
            "compile_s": t1 - t0, "train_s": t2 - t1,
            "memory": _memory(compiled, jax.local_devices()[0].memory_stats() or {})}


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def _flat(tree) -> dict:
    """A pytree's leaves as numpy arrays named by their path (a/b)."""
    return {"/".join(str(getattr(k, "key", getattr(k, "name", k))) for k in path): np.asarray(x)
            for path, x in jax.tree_util.tree_flatten_with_path(tree)[0]}


def save_block(out_dir: Path, task: str, persons: Sequence[dict], result: dict, smoke: bool,
               overrides: Sequence[str] = (), started_utc: str = "") -> dict:
    """Write a trained block to out_dir (which must not exist): its v2 zoo shard, metrics.npz and, last, block.json.
    Returns the block.json summary."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    zoo = ZooManager(str(out_dir / "zoo"), zoo_version=ZOO_VERSION)
    sha, dirty = _git("rev-parse", "HEAD"), bool(_git("status", "--porcelain", "--untracked-files=no"))
    for i, p in enumerate(persons):
        cfg = copy.deepcopy(result["config"])
        cfg["ZOO_PATH"] = str(out_dir / "zoo")
        pr, pref = p["prefs"], cfg["ENV_KWARGS"]["preference_rewards"]
        pref["preference_weights"] = {"speed_preference": pr["w_speed"], "force_preference": pr["w_force"],
                                      "touch_penalty": pr["w_touch"]}
        pref["preference_ranges"] = {"speed_range": [pr["speed_range_min"], pr["speed_range_max"]],
                                     "force_range": [pr["force_range_min"], pr["force_range_max"]]}
        cfg["POPULATION"] = {"person": p["id"], "block": out_dir.name, "key": p["key"], "git_sha": sha,
                             "git_dirty": dirty, "smoke": smoke}
        team = str(uuid.uuid4())
        for a, agent in enumerate(result["agents"]):
            zoo.save_agent(config=cfg, param_dict=jax.tree.map(lambda x: x[i, a], result["params"]),
                           scenario_agent_id=agent, team_uuid=team,
                           preference_weights={k: round(pr[k], 4) for k in ("w_speed", "w_force", "w_touch")})
    np.savez(out_dir / "metrics.npz", **_flat(result["metrics"]))
    alg, r, s = check_cell(persons)
    n, steps = len(persons), result["config"]["TOTAL_TIMESTEPS"]
    summary = {
        "block": out_dir.name, "task": task, "algorithm": alg, "disability": persons[0]["disability"],
        "restriction": r, "strength": s, "persons": [p["id"] for p in persons],
        "final_return": result["final_returns"].mean(axis=(1, 2)).tolist(),
        "final_return_sd_over_envs": result["final_returns"].mean(axis=2).std(axis=1).tolist(),
        "env_steps_per_person": steps, "compile_s": result["compile_s"], "train_s": result["train_s"],
        "env_steps_per_s": n * steps / result["train_s"], "memory": result["memory"], "zoo_version": ZOO_VERSION,
        "git_sha": sha, "git_dirty": dirty, "host": socket.gethostname(), "device": jax.devices()[0].device_kind,
        "xla_flags": os.environ.get("XLA_FLAGS", ""), "mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "pod": os.environ.get("POD_NAME", ""),
        "started_utc": started_utc,
        "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "smoke": smoke, "overrides": list(overrides)}
    tmp = out_dir / "block.json.tmp"
    tmp.write_text(json.dumps(summary, indent=1) + "\n")
    tmp.rename(out_dir / "block.json")
    return summary


def main(argv: Sequence[str]) -> None:
    """CLI: train one block and save it."""
    own, extra = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("task")
    ap.add_argument("persons", help="first-last person ids, inclusive")
    ap.add_argument("out_root", type=Path)
    ap.add_argument("--smoke", action="store_true", help="mark the block as a smoke test (allows overrides)")
    ap.add_argument("--compile-only", action="store_true", help="print the memory analysis and exit; writes nothing")
    args = ap.parse_args(own)
    if extra and not (args.smoke or args.compile_only):
        ap.error("hydra overrides change the population's settings; they are only allowed with --smoke")
    first, last = (int(x) for x in args.persons.split("-"))
    persons = load_manifest(args.manifest)["persons"][first:last + 1]
    if args.compile_only:
        res = train_block(args.task, persons, extra, compile_only=True)
        print(json.dumps({"task": args.task, "algorithm": res["algorithm"], "persons": len(persons),
                          "device": jax.devices()[0].device_kind, "compile_s": round(res["compile_s"], 1),
                          **res["memory"]}), flush=True)
        return
    out_dir = args.out_root / args.task / block_name(args.task, check_cell(persons)[0], first, last)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; blocks are never overwritten")
    started = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    print(f"{started} block {out_dir.name}: {len(persons)} persons, devices {jax.devices()}", flush=True)
    result = train_block(args.task, persons, extra)
    print(f"compile {result['compile_s']:.0f} s, train {result['train_s']:.0f} s, memory {result['memory']}", flush=True)
    summary = save_block(out_dir, args.task, persons, result, args.smoke, extra, started)
    print(f"saved {out_dir}: final return mean {np.mean(summary['final_return']):.1f}, "
          f"{summary['env_steps_per_s']:.0f} env steps/s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
