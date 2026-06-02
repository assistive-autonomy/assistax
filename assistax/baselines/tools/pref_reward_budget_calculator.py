"""Compute reward_budget values that achieve a target preference-to-total ratio.

Usage:
    python assistax/baselines/pref_reward_budget_calculator.py
    python assistax/baselines/pref_reward_budget_calculator.py --fraction 0.4
    python assistax/baselines/pref_reward_budget_calculator.py --env scratchitch
    python assistax/baselines/pref_reward_budget_calculator.py --config assistax/baselines/IPPO/config/ippo.yaml
"""

import argparse
from pathlib import Path
from typing import Optional

# Theoretical max task reward per step, derived from env constructor weights
# and known bounds of reward functions (exp/tanh/Boltzmann peak at 1.0).
# See assistax/envs/README.md Section 3 for derivation details.
#
# BedBathing uses the sustained per-step max (distance only = 1.0), not the
# discrete wiping event bonus (+3.0), since preference rewards are also
# sustained per-step signals.
ENV_TASK_MAX_PER_STEP: dict[str, float] = {
    "scratchitch": 1.0,  # dist_weight=1.0 * max(exp)=1.0
    "bedbathing": 1.0,  # dist only (wiping is a discrete event, not sustained)
    "armmanipulation": 11.28,  # waist=10*1.0 + hook=1*1.0 + rot=0.1*~2.8
    "feeding": 4.67,  # dist=2*1.0 + orient~1.3 + vel=1*1.0 + force=1*0.368
    "teethbrushing": 3.14,  # dist=2*1.0 + align~1.0 + brush=1*0.135
}


def compute_budget(task_max: float, fraction: float) -> float:
    """Compute reward_budget such that pref/(task+pref) = fraction.

    Derivation:
        fraction = budget / (task_max + budget)
        => budget = fraction * task_max / (1 - fraction)
    """
    return fraction * task_max / (1.0 - fraction)


def compute_ratio(task_max: float, budget: float) -> float:
    """Compute the actual preference fraction given task_max and budget."""
    return budget / (task_max + budget)


def read_budget_from_yaml(config_path: Path) -> Optional[float]:
    """Read reward_budget from a YAML config file (simple text parsing)."""
    try:
        for line in config_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("reward_budget:"):
                value = stripped.split(":", 1)[1].strip()
                return float(value)
    except (FileNotFoundError, ValueError):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute reward_budget values for a target preference fraction."
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.3,
        help="Target preference fraction: pref/(task+pref). Default: 0.3 (30%%)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        choices=list(ENV_TASK_MAX_PER_STEP.keys()),
        help="Show results for a single environment only.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file to read current reward_budget from.",
    )
    args = parser.parse_args()

    if not 0.0 < args.fraction < 1.0:
        parser.error("--fraction must be between 0 and 1 (exclusive)")

    # Determine which envs to show
    envs = {args.env: ENV_TASK_MAX_PER_STEP[args.env]} if args.env else ENV_TASK_MAX_PER_STEP

    # Try to read current budget from config
    current_budget: Optional[float] = None
    config_label = ""
    if args.config:
        config_path = Path(args.config)
        current_budget = read_budget_from_yaml(config_path)
        config_label = config_path.name
    else:
        # Try default ippo.yaml relative to this script
        default_config = Path(__file__).parent / "IPPO" / "config" / "ippo.yaml"
        current_budget = read_budget_from_yaml(default_config)
        if current_budget is not None:
            config_label = "ippo.yaml"

    # Print header
    frac_pct = args.fraction * 100
    print(f"\nPreference Reward Budget Calculator")
    print("=" * 80)
    print(f"Target preference fraction: {args.fraction:.2f} (prefs = {frac_pct:.0f}% of total reward signal)")
    print(f"Formula: reward_budget = fraction * task_max / (1 - fraction)\n")

    # Table header
    current_hdr = f"Current ({config_label})" if current_budget is not None else ""
    cur_col_w = max(len(current_hdr), 13) if current_budget is not None else 0
    header = f"{'Environment':<20} {'Task Max/Step':>13} {'Recommended Budget':>19}"
    separator_len = 55
    if current_budget is not None:
        header += f"  {current_hdr:>{cur_col_w}} {'Current Ratio':>14}"
        separator_len = 57 + cur_col_w + 14

    print(header)
    print("-" * separator_len)

    # Print rows
    for env_name, task_max in envs.items():
        budget = compute_budget(task_max, args.fraction)
        row = f"{env_name:<20} {task_max:>13.2f} {budget:>19.2f}"
        if current_budget is not None:
            ratio = compute_ratio(task_max, current_budget)
            row += f"  {current_budget:>{cur_col_w}.2f} {ratio:>14.2f}"
        print(row)

    # Notes
    print(f"\nNotes:")
    print(f"  - BedBathing task_max uses sustained (distance-only) reward, not discrete wiping events")
    if current_budget is not None:
        print(f'  - "Current" column reads reward_budget from {config_label}')
    print(f"  - These are theoretical upper bounds; actual achieved rewards will be lower")
    print(f"  - After setting reward_budget, do 1-2 verification runs and check wandb for")
    print(f"    total_pref_reward / (total_pref_reward + env_reward) to confirm actual ratio")
    print()


if __name__ == "__main__":
    main()
