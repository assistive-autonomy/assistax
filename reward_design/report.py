"""Print a compact table of reward-design run summaries: python -m reward_design.report <run dir> [<run dir> ...]."""
import json
import sys
from pathlib import Path

KEYS = ("fitness", "success", "safe", "f_rh_p99", "contact_frac", "f_tool_p50_in_contact", "f_other_mean", "task_d_p50",
        "human_act", "robot_act", "speed_p50", "root_disp_max", "root_tilt_max_deg", "neck_w_p99", "wiped", "gentle_wiped")


def row(path: Path) -> str:
    """One line: env, version, return, and the per-episode means of the fitness table (mean over seeds)."""
    s = json.loads((path / "summary.json").read_text())
    per = s["per_seed"]
    vals = {k: sum(p[k] for p in per) / len(per) for k in KEYS if k in per[0]}
    tag = f"{s['env']}/{s['version']}{' human=0' if s.get('human_zero') else ''}{' mean-act' if s.get('deterministic') else ''}"
    ret = sum(s["return_per_seed"]) / len(s["return_per_seed"])
    return f"{tag:32s} ret {ret:7.1f} " + " ".join(f"{k}={v:.3g}" for k, v in vals.items())


if __name__ == "__main__":
    for p in sys.argv[1:]:
        if (Path(p) / "summary.json").exists():
            print(row(Path(p)))
