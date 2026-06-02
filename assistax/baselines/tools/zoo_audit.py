"""Audit zoo agents for incorrect reward_budget in their training config.

Scans index.csv for agents matching a scenario, reads each agent's config YAML,
and checks whether ``ENV_KWARGS.preference_rewards.reward_budget`` matches the
expected value.  Reports totals and lists bad UUIDs grouped by team_uuid.

With ``--fix``, backs up index.csv and rewrites it excluding the bad agents
(does NOT delete config/params files).

Usage:
    # Audit bedbathing agents
    python zoo_audit.py --zoo-path /pvc/assistax/zoo --scenario bedbathing --expected-reward-budget 1

    # Remove bad agents from the index
    python zoo_audit.py --zoo-path /pvc/assistax/zoo --scenario bedbathing --expected-reward-budget 1 --fix
"""

import argparse
import os.path as osp
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import OmegaConf


def audit_reward_budget(
    zoo_path: str, scenario: str, expected: float
) -> Tuple[pd.DataFrame, List[Dict], List[Dict]]:
    """Check reward_budget for all agents in *scenario*.

    Returns:
        (full_index, good_agents, bad_agents) where each agent dict has keys
        ``agent_uuid``, ``team_uuid``, ``actual_reward_budget``.
    """
    index_path = osp.join(zoo_path, "index.csv")
    index = pd.read_csv(index_path)
    scenario_agents = index.query(f'scenario == "{scenario}"')

    good: List[Dict] = []
    bad: List[Dict] = []

    for _, row in scenario_agents.iterrows():
        agent_uuid = str(row["agent_uuid"])
        team_uuid = str(row.get("team_uuid", ""))
        config_path = osp.join(zoo_path, "config", f"{agent_uuid}.yaml")

        if not osp.exists(config_path):
            bad.append({
                "agent_uuid": agent_uuid,
                "team_uuid": team_uuid,
                "actual_reward_budget": None,
                "note": "MISSING_CONFIG",
            })
            continue

        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        pref = cfg.get("ENV_KWARGS", {}).get("preference_rewards")
        if pref is None:
            actual: Optional[float] = None
        else:
            actual = pref.get("reward_budget")

        entry = {
            "agent_uuid": agent_uuid,
            "team_uuid": team_uuid,
            "actual_reward_budget": actual,
        }

        if actual == expected:
            good.append(entry)
        else:
            bad.append(entry)

    return index, good, bad


def print_report(
    scenario: str, expected: float, good: List[Dict], bad: List[Dict]
) -> None:
    """Print a summary report of the audit."""
    total = len(good) + len(bad)
    print(f"Scenario             : {scenario}")
    print(f"Expected reward_budget: {expected}")
    print(f"Total agents         : {total}")

    if total == 0:
        print("No agents found.")
        return

    pct_good = 100 * len(good) / total
    pct_bad = 100 * len(bad) / total
    print(f"Correct              : {len(good):>4}  ({pct_good:.1f}%)")
    print(f"Incorrect            : {len(bad):>4}  ({pct_bad:.1f}%)")

    if not bad:
        print("\nAll agents have the correct reward_budget.")
        return

    # Group bad agents by actual reward_budget value
    by_value: Dict[Optional[float], int] = defaultdict(int)
    for a in bad:
        by_value[a["actual_reward_budget"]] += 1
    print("\nIncorrect agents by actual reward_budget:")
    for val, count in sorted(by_value.items(), key=lambda x: str(x[0])):
        print(f"  reward_budget={val}  -> {count} agent(s)")

    # Group bad UUIDs by team_uuid
    by_team: Dict[str, List[str]] = defaultdict(list)
    for a in bad:
        by_team[a["team_uuid"]].append(a["agent_uuid"])
    print(f"\nBad agents by team_uuid ({len(by_team)} team(s)):")
    for team, uuids in sorted(by_team.items()):
        print(f"  team {team}:")
        for uid in uuids:
            print(f"    {uid}")


def fix_index(
    zoo_path: str, index: pd.DataFrame, bad: List[Dict]
) -> None:
    """Back up index.csv and rewrite it without the bad agents."""
    if not bad:
        print("Nothing to fix.")
        return

    index_path = osp.join(zoo_path, "index.csv")
    backup_path = index_path + ".bak"
    shutil.copy2(index_path, backup_path)
    print(f"\nBacked up index to {backup_path}")

    bad_uuids = {a["agent_uuid"] for a in bad}
    cleaned = index[~index["agent_uuid"].isin(bad_uuids)]
    cleaned.to_csv(index_path, index=False)
    print(f"Updated {index_path}  ({len(index)} -> {len(cleaned)} rows, removed {len(bad_uuids)})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit zoo agents for incorrect reward_budget."
    )
    parser.add_argument("--zoo-path", required=True, help="Path to the zoo directory")
    parser.add_argument(
        "--scenario", default="bedbathing",
        help="Scenario/env name to filter (default: bedbathing)",
    )
    parser.add_argument(
        "--expected-reward-budget", type=float, default=1.0,
        help="Expected reward_budget value (default: 1.0)",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Back up index.csv and remove bad agents from it",
    )
    args = parser.parse_args()

    print(f"Zoo path              : {args.zoo_path}")
    print(f"Fix mode              : {args.fix}\n")

    index, good, bad = audit_reward_budget(
        args.zoo_path, args.scenario, args.expected_reward_budget
    )
    print_report(args.scenario, args.expected_reward_budget, good, bad)

    if args.fix:
        fix_index(args.zoo_path, index, bad)


if __name__ == "__main__":
    main()
