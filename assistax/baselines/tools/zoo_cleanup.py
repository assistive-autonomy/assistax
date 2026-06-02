"""Remove zoo agents that were trained without preference_rewards.

Usage:
    # Dry-run (default) — show what would be removed
    python zoo_cleanup.py --zoo-path /pvc/assistax/zoo --scenario feeding

    # Actually remove
    python zoo_cleanup.py --zoo-path /pvc/assistax/zoo --scenario feeding --no-dry-run
"""

import argparse
import os
import os.path as osp
from typing import List, Tuple

import pandas as pd
from omegaconf import OmegaConf


def find_agents_without_pref(
    zoo_path: str, scenario: str
) -> Tuple[pd.DataFrame, List[str]]:
    """Return (full_index, list_of_uuids_to_remove) for agents missing preference_rewards."""
    index_path = osp.join(zoo_path, "index.csv")
    index = pd.read_csv(index_path)
    scenario_agents = index.query(f'scenario == "{scenario}"')

    remove_uuids: List[str] = []
    for _, row in scenario_agents.iterrows():
        agent_uuid = row["agent_uuid"]
        config_path = osp.join(zoo_path, "config", f"{agent_uuid}.yaml")
        if not osp.exists(config_path):
            print(f"  [warn] config missing for {agent_uuid}, marking for removal")
            remove_uuids.append(agent_uuid)
            continue

        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        pref = cfg.get("ENV_KWARGS", {}).get("preference_rewards", None)
        if pref is None:
            remove_uuids.append(agent_uuid)

    return index, remove_uuids


def remove_agents(
    zoo_path: str,
    index: pd.DataFrame,
    remove_uuids: List[str],
    dry_run: bool = True,
) -> None:
    """Remove agents from index and delete their param/config files."""
    if not remove_uuids:
        print("Nothing to remove.")
        return

    action = "Would remove" if dry_run else "Removing"
    for uid in remove_uuids:
        params_file = osp.join(zoo_path, "params", f"{uid}.safetensors")
        config_file = osp.join(zoo_path, "config", f"{uid}.yaml")
        print(f"  {action}: {uid}")
        print(f"    params: {params_file} (exists={osp.exists(params_file)})")
        print(f"    config: {config_file} (exists={osp.exists(config_file)})")

        if not dry_run:
            if osp.exists(params_file):
                os.remove(params_file)
            if osp.exists(config_file):
                os.remove(config_file)

    if not dry_run:
        cleaned = index[~index["agent_uuid"].isin(remove_uuids)]
        index_path = osp.join(zoo_path, "index.csv")
        cleaned.to_csv(index_path, index=False)
        print(f"\nUpdated {index_path}  ({len(index)} -> {len(cleaned)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove zoo agents without preference_rewards from a given scenario."
    )
    parser.add_argument("--zoo-path", required=True, help="Path to the zoo directory")
    parser.add_argument(
        "--scenario", default="feeding", help="Scenario/env name to filter (default: feeding)"
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually delete files (default is dry-run)",
    )
    args = parser.parse_args()

    dry_run = not args.no_dry_run
    print(f"Zoo path : {args.zoo_path}")
    print(f"Scenario : {args.scenario}")
    print(f"Dry run  : {dry_run}\n")

    index, remove_uuids = find_agents_without_pref(args.zoo_path, args.scenario)
    print(f"Found {len(remove_uuids)} agent(s) without preference_rewards\n")
    remove_agents(args.zoo_path, index, remove_uuids, dry_run=dry_run)


if __name__ == "__main__":
    main()
