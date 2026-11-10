"""Does the human act? Compare each task's evaluation with the same policies but the human's actions set to zero.

python -m reward_design.checks <normal run root> <human-zero run root>   (each holding <env>/summary.json)
Prints one row per task: return, task distance, fitness and the human's mean |action| with and without the human.
"""
import json
import sys
from pathlib import Path

ENVS = ("teethbrushing", "feeding", "scratchitch", "bedbathing", "armmanipulation")


def stats(path: Path) -> dict:
    """Mean over seeds of the summary numbers used here."""
    s = json.loads((path / "summary.json").read_text())
    per = s["per_seed"]
    mean = lambda k: sum(p[k] for p in per) / len(per)
    return {"return": sum(s["return_per_seed"]) / len(per), "task_d": mean("task_d_p50"), "fitness": mean("fitness"),
            "human_act": mean("human_act")}


def table(normal: Path, zero: Path) -> list[dict]:
    """Rows for every task that has both runs."""
    rows = []
    for env in ENVS:
        if (normal / env / "summary.json").exists() and (zero / env / "summary.json").exists():
            a, b = stats(normal / env), stats(zero / env)
            rows.append({"env": env, **{f"{k}_human": v for k, v in a.items()}, **{f"{k}_zero": v for k, v in b.items()}})
    return rows


if __name__ == "__main__":
    for r in table(Path(sys.argv[1]), Path(sys.argv[2])):
        print(f"{r['env']:16s} return {r['return_human']:7.0f} -> {r['return_zero']:7.0f} "
              f"({100 * (r['return_zero'] / r['return_human'] - 1):+.0f}%) | task distance p50 "
              f"{100 * r['task_d_human']:5.1f} -> {100 * r['task_d_zero']:5.1f} cm | fitness {r['fitness_human']:.3f} -> "
              f"{r['fitness_zero']:.3f} | human |a| {r['human_act_human']:.2f}")
