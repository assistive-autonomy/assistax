"""Compare paper_baselines runs with the paper's own runs (W&B lh-from-kb/assistax-dev, tags MARL_FINAL + <ALG> + FF_NPS,
without PREF: the runs behind the paper's Figure 3). Read-only W&B access via ~/.netrc.

The curve is mean_episode_returns___all__: the team return averaged over evaluation episodes, per seed and checkpoint.
As in the paper's plotting script (assistax/baselines/plotting/paper_plots.py on dev-main), runs of the same algorithm
and task are pooled along the seed axis, and a seed's final return is its mean over the last 10 checkpoints.

Usage:
  python -m paper_baselines.compare fetch <cache dir>              # the paper's curves -> <cache>/<alg>_<env>/<run id>.npz
  python -m paper_baselines.compare table <cache dir> <runs root>  # final returns, paper vs ours (runs from slurm.sbatch)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assistax.baselines.utils import load_compact_npz  # noqa: E402

PROJECT = "lh-from-kb/assistax-dev"
METRIC = "mean_episode_returns___all__.npz"
ALGS = ("ippo", "mappo", "masac")
FINAL_N = 10


def fetch(cache: Path) -> None:
    """Download the metric file (only) of each paper run's evaluation_data artifact, with an index of the runs."""
    import wandb
    api = wandb.Api(timeout=120)
    index = []
    for alg in ALGS:
        for r in api.runs(PROJECT, filters={"tags": {"$all": ["MARL_FINAL", alg.upper(), "FF_NPS"]}}):
            if "PREF" in r.tags:
                continue
            env = r.config["ENV_NAME"]
            dest = cache / f"{alg}_{env}"
            entry = api.artifact(f"{PROJECT}/evaluation_data_{r.name}:latest").get_entry(METRIC)
            entry.download(root=str(dest))
            (dest / METRIC).rename(dest / f"{r.id}.npz")
            index.append({"alg": alg, "env": env, "id": r.id, "name": r.name, "state": r.state, "created": r.created_at,
                          "NUM_EVAL_EPISODES": r.config.get("NUM_EVAL_EPISODES"),
                          "shape": list(load_compact_npz(str(dest / f"{r.id}.npz")).shape)})
            print(json.dumps(index[-1]), flush=True)
    (cache / "runs.json").write_text(json.dumps(index, indent=1))


def curves(files: list[Path]) -> np.ndarray | None:
    """(seeds, checkpoints) curves from several files, pooled along seeds and cut to the shortest."""
    arrays = [load_compact_npz(str(f)) for f in files]
    if not arrays:
        return None
    n = min(a.shape[1] for a in arrays)
    return np.concatenate([a[:, :n] for a in arrays])


def final(c: np.ndarray, reps: int = 10000) -> dict:
    """Mean over seeds of each seed's last-10-checkpoint mean, with a 95% percentile bootstrap CI over seeds."""
    f = c[:, -FINAL_N:].mean(axis=1)
    boot = np.random.default_rng(0).choice(f, (reps, f.size)).mean(axis=1)
    return {"mean": float(f.mean()), "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
            "seeds": int(f.size), "checkpoints": int(c.shape[1])}


def table(cache: Path, runs: Path) -> None:
    """Print final returns, paper vs ours, and write them to <runs root>/compare.json."""
    rows = []
    for d in sorted(cache.glob("*_*")):
        alg, env = d.name.split("_", 1)
        paper = curves(sorted(d.glob("*.npz")))
        ours = curves(sorted((runs / d.name).glob(f"seed*/eval/{METRIC}")))
        row = {"alg": alg, "env": env, "paper": final(paper), "ours": final(ours) if ours is not None else None}
        rows.append(row)
        fmt = lambda s: f"{s['mean']:.0f} [{s['lo']:.0f}, {s['hi']:.0f}] (n={s['seeds']})" if s else "-"
        print(f"{alg:6s} {env:16s} paper {fmt(row['paper'])}   ours {fmt(row['ours'])}")
    (runs / "compare.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["fetch", "table"])
    p.add_argument("cache", type=Path)
    p.add_argument("runs", type=Path, nargs="?")
    a = p.parse_args()
    if a.cmd == "fetch":
        a.cache.mkdir(parents=True, exist_ok=True)
        fetch(a.cache)
    else:
        table(a.cache, a.runs)


if __name__ == "__main__":
    main()
