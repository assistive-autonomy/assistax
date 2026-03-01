"""
Merge Zoo Shards into a Single Unified Zoo

Scans all zoo shards under a root directory, deduplicates agents that were
trained with identical preference weights AND disability settings (keeping
only the first encountered), and copies the unique agents into a single
target zoo with one unified index.csv.

Usage:
    python merge_zoo_shards.py --shards_root /pvc/zoo_shards \
                               --target_zoo /pvc/assistax/zoo \
                               [--dry_run]

The script NEVER deletes source files — it only copies.
"""

import argparse
import shutil
import hashlib
import json
import pandas as pd
import yaml
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# Fingerprinting: what makes two agents "duplicates"?
# ─────────────────────────────────────────────────────────────

def _extract_fingerprint(config: dict) -> str:
    """
    Build a canonical fingerprint string from the training config.

    Two agents are duplicates if they share the SAME:
      - scenario (ENV_NAME)
      - scenario_agent_id (which agent slot, e.g. 'robot' vs 'human')
      - algorithm
      - disability settings
      - preference weights + ranges

    We hash a sorted JSON of these fields so floating-point key order
    doesn't cause false negatives.
    """
    env_kwargs = config.get("ENV_KWARGS", {})

    fp_dict = {
        "scenario": config.get("ENV_NAME", ""),
        "algorithm": config.get("ALGORITHM", ""),
        # Disability block (may not exist for all envs)
        "disability": env_kwargs.get("disability", {}),
        # Full preference rewards block
        "preference_rewards": env_kwargs.get("preference_rewards", {}),
    }

    # Canonical JSON with sorted keys for deterministic hashing
    canonical = json.dumps(fp_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _extract_fingerprint_with_agent_id(config: dict, scenario_agent_id: str) -> str:
    """
    Same as _extract_fingerprint but also includes the scenario_agent_id
    so that 'robot' and 'human' agents from the same training run are NOT
    considered duplicates of each other.
    """
    env_kwargs = config.get("ENV_KWARGS", {})

    fp_dict = {
        "scenario": config.get("ENV_NAME", ""),
        "scenario_agent_id": scenario_agent_id,
        "algorithm": config.get("ALGORITHM", ""),
        "disability": env_kwargs.get("disability", {}),
        "preference_rewards": env_kwargs.get("preference_rewards", {}),
    }

    canonical = json.dumps(fp_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────
# Discovery: find all zoo shards
# ─────────────────────────────────────────────────────────────

def discover_shard_zoos(shards_root: Path) -> list[Path]:
    """
    Recursively find all directories that contain an index.csv
    (i.e. valid zoo shards).
    """
    return sorted(shards_root.rglob("index.csv"))


# ─────────────────────────────────────────────────────────────
# Merge logic
# ─────────────────────────────────────────────────────────────

def merge_zoos(shards_root: str, target_zoo: str, dry_run: bool = False):
    shards_root = Path(shards_root)
    target = Path(target_zoo)

    index_files = discover_shard_zoos(shards_root)
    print(f"Found {len(index_files)} zoo shards under {shards_root}")
    if not index_files:
        print("Nothing to merge.")
        return

    # ── Pass 1: Load all shard indices and configs, compute fingerprints ──
    all_rows = []  # (shard_zoo_dir, row_dict, config_dict, fingerprint)

    for idx_path in index_files:
        shard_zoo_dir = idx_path.parent
        df = pd.read_csv(idx_path)
        print(f"  Shard: {shard_zoo_dir}  ({len(df)} agents)")

        for _, row in df.iterrows():
            agent_uuid = row["agent_uuid"]
            config_path = shard_zoo_dir / "config" / f"{agent_uuid}.yaml"

            if not config_path.exists():
                print(f"    WARNING: config missing for {agent_uuid}, skipping")
                continue

            params_path = shard_zoo_dir / "params" / f"{agent_uuid}.safetensors"
            if not params_path.exists():
                print(f"    WARNING: params missing for {agent_uuid}, skipping")
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f)

            scenario_agent_id = row.get("scenario_agent_id", "")
            fp = _extract_fingerprint_with_agent_id(config, scenario_agent_id)

            all_rows.append({
                "shard_zoo_dir": shard_zoo_dir,
                "row": row.to_dict(),
                "config": config,
                "fingerprint": fp,
                "agent_uuid": agent_uuid,
            })

    print(f"\nTotal agents across all shards: {len(all_rows)}")

    # ── Pass 2: Deduplicate (seeded with existing target zoo if present) ──
    seen_fingerprints: dict[str, str] = {}  # fingerprint -> first agent_uuid
    existing_count = 0

    index_path = target / "index.csv"
    if index_path.exists():
        existing_df = pd.read_csv(index_path)
        print(f"Target zoo already has {len(existing_df)} agents — seeding fingerprints...")
        for _, row in existing_df.iterrows():
            agent_uuid = row["agent_uuid"]
            config_path = target / "config" / f"{agent_uuid}.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                scenario_agent_id = row.get("scenario_agent_id", "")
                fp = _extract_fingerprint_with_agent_id(config, scenario_agent_id)
                seen_fingerprints[fp] = agent_uuid
                existing_count += 1
        print(f"  Seeded {existing_count} fingerprints from existing zoo")

    unique_agents = []
    duplicate_count = 0
    dup_details = defaultdict(list)

    for entry in all_rows:
        fp = entry["fingerprint"]
        if fp in seen_fingerprints:
            duplicate_count += 1
            dup_details[fp].append(entry["agent_uuid"])
        else:
            seen_fingerprints[fp] = entry["agent_uuid"]
            unique_agents.append(entry)

    print(f"Unique agents to copy:  {len(unique_agents)}")
    print(f"Duplicates skipped:     {duplicate_count}")

    if dup_details:
        print(f"\nDuplicate groups (showing fingerprints with >0 extra copies):")
        for fp, dup_uuids in sorted(dup_details.items(), key=lambda x: -len(x[1])):
            first = seen_fingerprints[fp]
            print(f"  Kept {first}, skipped {len(dup_uuids)}: {dup_uuids[:3]}{'...' if len(dup_uuids) > 3 else ''}")
        print()

    if dry_run:
        print("DRY RUN — no files copied.")
        _print_summary(unique_agents)
        return

    # ── Pass 3: Copy unique agents to target zoo ──
    target.mkdir(parents=True, exist_ok=True)
    (target / "config").mkdir(exist_ok=True)
    (target / "params").mkdir(exist_ok=True)

    merged_rows = []

    for entry in unique_agents:
        shard_zoo_dir = entry["shard_zoo_dir"]
        agent_uuid = entry["agent_uuid"]

        # Copy config YAML
        src_config = shard_zoo_dir / "config" / f"{agent_uuid}.yaml"
        dst_config = target / "config" / f"{agent_uuid}.yaml"
        if not dst_config.exists():
            shutil.copy2(src_config, dst_config)

        # Copy params safetensors
        src_params = shard_zoo_dir / "params" / f"{agent_uuid}.safetensors"
        dst_params = target / "params" / f"{agent_uuid}.safetensors"
        if not dst_params.exists():
            shutil.copy2(src_params, dst_params)

        merged_rows.append(entry["row"])

    # ── Write unified index.csv ──
    merged_df = pd.DataFrame(merged_rows)

    index_path = target / "index.csv"
    if index_path.exists():
        existing_df = pd.read_csv(index_path)
        # Avoid duplicating UUIDs already in target
        existing_uuids = set(existing_df["agent_uuid"].values)
        new_rows = merged_df[~merged_df["agent_uuid"].isin(existing_uuids)]
        merged_df = pd.concat([existing_df, new_rows], ignore_index=True)
        print(f"Appended {len(new_rows)} new agents to existing index ({len(existing_df)} prior)")
    
    merged_df.to_csv(index_path, index=False)

    print(f"\n✅  Merged zoo written to: {target}")
    print(f"   Total agents in index: {len(merged_df)}")
    _print_summary_from_df(merged_df)


def _print_summary(unique_agents: list[dict]):
    """Print a summary breakdown of unique agents."""
    by_scenario = defaultdict(lambda: defaultdict(int))
    for entry in unique_agents:
        scenario = entry["row"].get("scenario", "unknown")
        algorithm = entry["row"].get("algorithm", "unknown")
        by_scenario[scenario][algorithm] += 1

    print("Summary by scenario / algorithm:")
    for scenario in sorted(by_scenario):
        for algo in sorted(by_scenario[scenario]):
            print(f"  {scenario:25s}  {algo:10s}  {by_scenario[scenario][algo]:4d} agents")


def _print_summary_from_df(df: pd.DataFrame):
    """Print summary from a DataFrame."""
    print("\nBreakdown by scenario / algorithm:")
    for (scenario, algo), group in df.groupby(["scenario", "algorithm"]):
        agent_ids = group["scenario_agent_id"].unique()
        print(f"  {scenario:25s}  {algo:10s}  {len(group):4d} agents  (slots: {', '.join(map(str, agent_ids))})")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge zoo shards into a single unified zoo with deduplication."
    )
    parser.add_argument(
        "--shards_root",
        type=str,
        required=True,
        help="Root directory containing zoo shards (e.g. /pvc/zoo_shards)",
    )
    parser.add_argument(
        "--target_zoo",
        type=str,
        required=True,
        help="Target directory for the merged zoo (e.g. /pvc/assistax/zoo)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only report what would happen, don't copy anything",
    )
    args = parser.parse_args()
    merge_zoos(args.shards_root, args.target_zoo, dry_run=args.dry_run)