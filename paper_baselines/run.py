"""Run one of the paper's MARL baselines (IPPO, MAPPO or MASAC, feed-forward, no preferences) through its unmodified
runner, with the paper's launch overrides from settings.json. W&B stays off: the runner's W&B outputs are written to
the output directory instead.

- config.json: the runner's full resolved config (what it passes to wandb.init).
- wandb_log.jsonl: every scalar the runner logs (per-checkpoint training and evaluation metrics, final stats).
- eval/episode_returns_<agent>.npz (seeds, checkpoints, episodes) and eval/mean_episode_returns_<agent>.npz
  (seeds, checkpoints), in the runner's compact npz format: the files the paper's Figure 3 is built from.
- params/<agent>.safetensors: final parameters per agent, all seeds stacked. The per-checkpoint parameter history
  (all_params) is not kept, except that the MASAC runner writes its own (actor/q1/q2 *_all_params and finals) and
  metrics.npy into hydra/, its working directory (masac.yaml sets hydra.job.chdir). MAPPO writes metrics.npy here.
- artifacts/<name>/: anything else the runner uploads (best, median and worst episode trajectories).

Seeds: the runner trains NUM_SEEDS seeds from key SEED. The paper ran 16 seeds from SEED 0 in one process on an
A100-80GB; on smaller GPUs, split them into processes with --seeds 16/k and --seed 0..k-1 (as the paper's recurrent
runs did), which changes the seeds' keys but nothing else.

Usage, from the checkout root:
  python -m paper_baselines.run <ippo|mappo|masac> <env> <out dir> [--seeds 4] [--seed 0] [--capacity 8192]
                                [-- <extra hydra overrides>]
"""
import argparse
import json
import os
import runpy
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {"ippo": "IPPO/ippo_run.py", "mappo": "MAPPO/mappo_run.py", "masac": "MASAC/masac_run.py"}


def scalar_items(data: dict) -> dict:
    """The entries of a wandb.log dict that are single numbers, as floats."""
    out = {}
    for k, v in data.items():
        try:
            a = np.asarray(v)
        except Exception:
            continue
        if a.size == 1 and a.dtype.kind in "biuf":
            out[k] = float(a.reshape(()))
    return out


def capture_outputs(out: Path) -> None:
    """Redirect the runner's W&B calls (init, log, artifacts, parameter and eval-data uploads) to files in `out`."""
    import wandb
    from flax.traverse_util import flatten_dict
    import jax
    import safetensors.flax
    from assistax.baselines import utils

    init = wandb.init

    def init_to_disk(*args, **kw):
        (out / "config.json").write_text(json.dumps(kw.get("config", {}), indent=1, default=str))
        kw["mode"] = "disabled"
        run = init(*args, **kw)
        wandb.log = log_to_disk  # wandb.init rebinds wandb.log to the run's own
        return run

    def log_to_disk(data: dict, step=None, **kw) -> None:
        with open(out / "wandb_log.jsonl", "a") as f:
            f.write(json.dumps({"step": step, **scalar_items(data)}) + "\n")

    def add_file(self, local_path: str, name: str | None = None, **kw) -> None:
        dest = out / "artifacts" / self.name / (name or Path(local_path).name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def add_dir(self, local_path: str, name: str | None = None, **kw) -> None:
        shutil.copytree(local_path, out / "artifacts" / self.name / (name or ""), dirs_exist_ok=True)

    def save_params(all_train_states, final_train_state, config: dict, env, run) -> None:
        """Final parameters per agent (seeds stacked), as upload_model_parameters_to_wandb names them."""
        (out / "params").mkdir(exist_ok=True)
        split = utils._unstack_tree(jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state.params))
        for agent, params in zip(env.agents, split):
            safetensors.flax.save_file(flatten_dict(params, sep="/"), str(out / "params" / f"{agent}.safetensors"))
        print(f"Saved final parameters to {out / 'params'}")

    def save_eval_returns(eval_info, config: dict, run, suffix: str = "") -> None:
        """Per-episode and mean episode returns, as upload_eval_data_to_wandb saves them (same names and format)."""
        (out / "eval").mkdir(exist_ok=True)
        returns = utils._compute_episode_returns(eval_info, time_axis=2)
        for agent, r in returns.items():
            utils._save_compact_npz(str(out / "eval" / f"episode_returns_{agent}{suffix}.npz"), r)
            utils._save_compact_npz(str(out / "eval" / f"mean_episode_returns_{agent}{suffix}.npz"), r.mean(axis=-1))
        print(f"Saved evaluation returns to {out / 'eval'}")

    wandb.init, wandb.log = init_to_disk, log_to_disk
    wandb.Artifact.add_file, wandb.Artifact.add_dir = add_file, add_dir
    utils.upload_model_parameters_to_wandb, utils.upload_eval_data_to_wandb = save_params, save_eval_returns


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("alg", choices=sorted(RUNNERS))
    p.add_argument("env")
    p.add_argument("out", type=Path)
    p.add_argument("--seeds", type=int, default=16, help="NUM_SEEDS (the paper: 16)")
    p.add_argument("--seed", type=int, default=0, help="SEED (the paper: 0)")
    p.add_argument("--capacity", type=int, default=None, help="GPU_ENV_CAPACITY: evaluation batch, no effect on results")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a, extra = p.parse_args(argv[:cut]), argv[cut + 1:]  # extra hydra overrides, after --
    settings = json.loads((Path(__file__).parent / "settings.json").read_text())
    replaced = {"NUM_SEEDS", *(o.lstrip("+").split("=")[0] for o in extra)}  # extra overrides win
    overrides = [o for o in settings["overrides"][a.alg][a.env] if o.lstrip("+").split("=")[0] not in replaced]
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runner = ROOT / "assistax/baselines" / RUNNERS[a.alg]
    args = ["-cn", a.alg, "network=ff_nps", f"++ENV_NAME={a.env}", *overrides, f"++NUM_SEEDS={a.seeds}",
            f"++SEED={a.seed}", "++WANDB_MODE=disabled", "++RENDER_VIDEOS=False", f"hydra.run.dir={out / 'hydra'}"]
    if a.capacity:
        args.append(f"++GPU_ENV_CAPACITY={a.capacity}")
    args += extra
    os.environ.setdefault("EXP_ID", "paper010")
    (out / "launch.json").write_text(json.dumps({"runner": str(runner.relative_to(ROOT)), "args": args,
                                                 "settings_source": settings["source"]}, indent=1))
    sys.path[:0] = [str(runner.parent), str(ROOT)]  # the runners import their algorithm module by bare name
    capture_outputs(out)
    os.chdir(out)  # the MAPPO and MASAC runners write metrics and parameter files to the working directory
    sys.argv = [str(runner), *args]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
