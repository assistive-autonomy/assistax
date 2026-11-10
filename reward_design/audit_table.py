"""Compare human-involvement audits across code versions: python -m reward_design.audit_table <root> [<root> ...]

Each root holds <env>/audit.json (one per env, from human_audit.py), e.g. the 0.1.0 and pre-0.1.0 policies.
Prints one markdown row per env and root.
"""
import json
import sys
from pathlib import Path

ENVS = ("teethbrushing", "feeding", "scratchitch", "bedbathing", "armmanipulation")


def row(env: str, root: Path) -> str:
    """One markdown table row for an env's audit under `root`."""
    a = json.loads((root / env / "audit.json").read_text())
    joints = ", ".join(f"{k.replace('right_', '')} {v['range_deg']:.0f}°" for k, v in a["joints"].items())
    ret = 100 * (a["human_zero_return"] / a["sampled_return"] - 1)
    itch = ""
    if "elbow_deg_lower_arm_itch" in a:
        itch = f" (elbow {a['elbow_deg_upper_arm_itch']:.0f}° upper-arm itch vs {a['elbow_deg_lower_arm_itch']:.0f}° lower)"
    return (f"| {env} | {root.name} | {a['shared_reward_max_diff']:.0g} | {100 * a['mean_effort']:.0f}% / "
            f"{100 * a['mean_pinned']:.0f}% / {100 * a['mean_change']:.0f}% | {joints}{itch} | "
            f"{100 * a['sampled_gap_share']:.0f}% → {100 * a['human_zero_gap_share']:.0f}% | "
            f"{a['sampled_tool_target_cm']:.1f} → {a['human_zero_tool_target_cm']:.1f} cm | {ret:+.0f}% |")


if __name__ == "__main__":
    roots = [Path(p) for p in sys.argv[1:]]
    print("| task | code | reward diff | human effort / pinned / change (mean actions) | human joint range per episode "
          "(mean actions) | approach made by the human's side, human acting → zeroed | tool–target, acting → zeroed "
          "| return, human zeroed |")
    print("|---|---|---|---|---|---|---|---|")
    for env in ENVS:
        for root in roots:
            if (root / env / "audit.json").exists():
                print(row(env, root))
