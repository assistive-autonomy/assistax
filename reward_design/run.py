"""Reward-design runs: train IPPO on an instrumented env version, or evaluate saved policies; then score and render.

Same training call, config (ippo.yaml defaults) and evaluation episodes as results/.../0810/train_render.py; the env
is swapped for reward_design.tasks' instrumented class. Writes to <out_dir>:
  config.json, train_metrics.npz, robot/human.safetensors      (train only)
  eval.npz            per-step metrics (all x_* diagnostics), rewards and dones: (seeds, T, episodes)
  rollout_seed0.npz   qpos/qvel/done of seed 0 (+ itch / wipe state where the task has one) for rendering
  episodes.json       per-episode fitness/success table per seed (fitness.py)
  summary.json        returns, fitness, success, timings
  videos/seed0_{worst,median,best}.mp4   picked by fitness, with a per-step overlay

Usage (from the worktree root):
  python -m reward_design.run train <env> <version> <out_dir> [--steps 2e7] [--seeds 1]
  python -m reward_design.run eval <env> <version> <out_dir> --params <dir> [--human-zero]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "  # as ippo_run.py

import distrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import safetensors.flax  # noqa: E402
from flax.traverse_util import flatten_dict, unflatten_dict  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import assistax  # noqa: E402
from assistax.baselines.utils import _unstack_tree  # noqa: E402
from reward_design import fitness, render, tasks  # noqa: E402

IPPO_DIR = Path(assistax.__file__).parent / "baselines" / "IPPO"
sys.path.insert(0, str(IPPO_DIR))
from ippo_ff_nps import MultiActorCritic, batchify, make_evaluation, make_train, unbatchify  # noqa: E402

RECORD_INFO = {"scratchitch": ("scratch",), "bedbathing": ("contact_vector", "gentle_cv")}


def load_config(env_name: str, overrides: list[str]) -> dict:
    """Compose ippo.yaml as `python ippo_run.py ENV_NAME=<env> ...` would, and resolve it."""
    with initialize_config_dir(config_dir=str(IPPO_DIR / "config"), version_base=None):
        cfg = compose("ippo", overrides=[f"ENV_NAME={env_name}", *overrides])
    return OmegaConf.to_container(cfg, resolve=True)


def load_params(run_dir: Path, agents: list[str], seed: int) -> dict:
    """Network params of one training seed: each agent's saved params (seed axis first), stacked on the agent axis."""
    per_agent = []
    for agent in agents:
        flat = safetensors.flax.load_file(str(run_dir / f"{agent}.safetensors"))
        per_agent.append(unflatten_dict({tuple(k.split("/")): jnp.asarray(v[seed]) for k, v in flat.items()}))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *per_agent)


def make_rollout(eval_env, apply_fn, n_episodes: int, env_name: str, human_zero: bool, deterministic: bool = False):
    """Sampled-action evaluation like ippo_ff_nps.run_evaluation (or the policy means if `deterministic`),
    recording metrics, rewards, dones and qpos/qvel."""
    human = eval_env.agents.index("human")

    def rollout(rng: jax.Array, params) -> dict:
        rng_reset, rng = jax.random.split(rng)
        obs, env_state = jax.vmap(eval_env.reset)(jax.random.split(rng_reset, n_episodes))
        last_done = jnp.zeros((eval_env.num_agents, n_episodes), dtype=bool)

        def step(carry, _):
            obs, env_state, last_done, rng = carry
            obs_batch = batchify(obs, eval_env.agents)
            avail = batchify(jax.vmap(eval_env.get_avail_actions)(env_state.env_state), eval_env.agents)
            (mean, std), _ = apply_fn(params, (obs_batch, last_done, avail))
            rng, k_act, k_step = jax.random.split(rng, 3)
            action = distrax.MultivariateNormalDiag(mean, jnp.expand_dims(std, axis=1)).sample(seed=k_act)
            if deterministic:
                action = mean
            if human_zero:
                action = action.at[human].set(0.0)
            obs, env_state, reward, done, _ = jax.vmap(eval_env.step)(
                jax.random.split(k_step, n_episodes), env_state, unbatchify(action, eval_env.agents))
            inner = env_state.env_state
            ps = inner.pipeline_state
            rec = {"qpos": ps.qpos, "qvel": ps.qvel, "done": done["__all__"], "metrics": inner.metrics,
                   "reward": reward["__all__"], "info": {k: inner.info[k] for k in RECORD_INFO.get(env_name, ())}}
            return (obs, env_state, batchify(done, eval_env.agents), rng), rec

        _, recs = jax.lax.scan(step, (obs, env_state, last_done, rng), None, eval_env.episode_length)
        return recs

    return rollout


def load_rec(out_dir: Path) -> dict:
    """Seed 0's saved rollout and metrics (rollout_seed0.npz, eval.npz) in the form render.render_picks takes."""
    r, e = np.load(out_dir / "rollout_seed0.npz"), np.load(out_dir / "eval.npz")
    rec = {"qpos": r["qpos"], "qvel": r["qvel"], "done": r["done"], "info": {},
           "metrics": {k.split("/", 1)[1]: e[k][0] for k in e.files if k.startswith("metrics/")}}
    for k in r.files:
        if k.startswith("info/"):
            *path, leaf = k.split("/")[1:]
            node = rec["info"]
            for p in path:
                node = node.setdefault(p, {})
            node[leaf] = r[k]
    return rec


def evaluate(config: dict, env_name: str, version: str, out_dir: Path, params_per_seed: list, human_zero: bool,
             timings: dict, deterministic: bool = False, render_videos: bool = True) -> None:
    """Evaluate each seed's params on the same 32 episodes, score them, save arrays and render seed 0."""
    rng = jax.random.key(config["SEED"])
    _, eval_rng = jax.random.split(rng)
    t0 = time.time()
    eval_env, _ = make_evaluation(config)
    apply_fn = MultiActorCritic(config=config).apply
    rollout = jax.jit(make_rollout(eval_env, apply_fn, config["NUM_EVAL_EPISODES"], env_name, human_zero,
                                   deterministic))
    recs = [jax.device_get(rollout(eval_rng, p)) for p in params_per_seed]
    timings["eval_s"] = time.time() - t0

    arrays = {"done": np.stack([r["done"] for r in recs]), "reward": np.stack([r["reward"] for r in recs])}
    for k in recs[0]["metrics"]:
        arrays[f"metrics/{k}"] = np.stack([np.asarray(r["metrics"][k]) for r in recs])
    np.savez(out_dir / "eval.npz", **arrays)
    info0 = jax.tree.map(np.asarray, recs[0]["info"])
    np.savez(out_dir / "rollout_seed0.npz", qpos=recs[0]["qpos"], qvel=recs[0]["qvel"], done=recs[0]["done"],
             **{f"info/{'/'.join(map(str, path))}": v for path, v in _flat(info0)})

    tables, returns = [], []
    for r in recs:
        metrics = {k: np.asarray(v) for k, v in r["metrics"].items()}
        mask = fitness.first_episode_mask(np.asarray(r["done"]))
        returns.append((np.asarray(r["reward"]) * mask).sum(0))
        tables.append(fitness.episode_table(env_name, metrics, np.asarray(r["done"])))
    (out_dir / "episodes.json").write_text(json.dumps(
        [{k: np.asarray(v).tolist() for k, v in t.items()} for t in tables]))

    t0 = time.time()
    order = np.argsort(tables[0]["fitness"] + 1e-6 * returns[0])
    picks = {"worst": int(order[0]), "median": int(order[len(order) // 2]), "best": int(order[-1])}
    if not human_zero and render_videos:
        render.render_picks(env_name, version, out_dir, recs[0], picks, float(eval_env.env.dt))
    timings["render_s"] = time.time() - t0

    per_seed = [fitness.summarise(t) for t in tables]
    summary = {
        "env": env_name, "version": version, "human_zero": human_zero, "deterministic": deterministic,
        "dt": float(eval_env.env.dt), "timings": timings,
        "total_timesteps": config["TOTAL_TIMESTEPS"], "num_seeds": len(params_per_seed),
        "return_per_seed": [float(x.mean()) for x in returns],
        "fitness_per_seed": [s["fitness"] for s in per_seed], "success_per_seed": [s["success"] for s in per_seed],
        "per_seed": per_seed, "rendered_seed0_episodes": picks,
        "rendered": {k: {"fitness": float(tables[0]["fitness"][v]), "return": float(returns[0][v])} for k, v in picks.items()},
        "spec": fitness.SPEC[env_name],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"[{env_name} {version}] return {np.mean(returns):.1f}, fitness {np.mean([s['fitness'] for s in per_seed]):.3f}, "
          f"success {np.mean([s['success'] for s in per_seed]):.2f}, safe {np.mean([s['safe'] for s in per_seed]):.2f}",
          flush=True)


def _flat(tree, prefix=()):
    """(path, leaf) pairs of a nested dict."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _flat(v, prefix + (k,))
    else:
        yield prefix, tree


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "render"])
    p.add_argument("env")
    p.add_argument("version")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--steps", type=float, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--params", type=Path, default=None)
    p.add_argument("--human-zero", action="store_true")
    p.add_argument("--deterministic", action="store_true", help="evaluate the policy means instead of sampling")
    p.add_argument("--no-render", action="store_true", help="skip the videos (render later with the render mode)")
    a = p.parse_args()
    if a.mode == "render":  # videos of an evaluated run, from its saved files
        s = json.loads((a.out_dir / "summary.json").read_text())
        dt = s.get("dt") or float(assistax.make(a.env).env.dt)
        render.render_picks(a.env, a.version, a.out_dir, load_rec(a.out_dir), s["rendered_seed0_episodes"], dt)
        return
    tasks.install(a.env, a.version)
    overrides = [f"TOTAL_TIMESTEPS={a.steps}"] * (a.steps is not None) + [f"NUM_SEEDS={a.seeds}"] * (a.seeds is not None)
    config = load_config(a.env, overrides)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    timings = {}
    if a.mode == "train":
        (a.out_dir / "config.json").write_text(json.dumps({**config, "REWARD_VERSION": a.version}, indent=1))
        train_rng, _ = jax.random.split(jax.random.key(config["SEED"]))
        t0 = time.time()
        train_jit = jax.jit(make_train(config, save_train_state=False), device=jax.devices()[config["DEVICE"]])
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            jax.random.split(train_rng, config["NUM_SEEDS"]), config["LR"], config["ENT_COEF"], config["CLIP_EPS"])
        jax.block_until_ready(out)
        timings["train_s"] = time.time() - t0
        print(f"[{a.env} {a.version}] trained in {timings['train_s']:.0f} s", flush=True)
        np.savez(a.out_dir / "train_metrics.npz", **{k: np.asarray(v) for k, v in out["metrics"].items()})
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        params = out["runner_state"].train_state.params
        for agent, ap in zip(env.agents, _unstack_tree(jax.tree.map(lambda x: x.swapaxes(0, 1), params))):
            safetensors.flax.save_file(flatten_dict(ap, sep="/"), str(a.out_dir / f"{agent}.safetensors"))
        params_dir = a.out_dir
    else:
        params_dir = a.params
    eval_env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    n_seeds = next(iter(safetensors.flax.load_file(str(params_dir / "robot.safetensors")).values())).shape[0]
    params = [load_params(params_dir, list(eval_env.agents), s) for s in range(n_seeds)]
    config["NUM_SEEDS"] = n_seeds
    evaluate(config, a.env, a.version, a.out_dir, params, a.human_zero, timings, a.deterministic, not a.no_render)


if __name__ == "__main__":
    main()
