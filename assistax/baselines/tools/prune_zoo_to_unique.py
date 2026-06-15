"""Build a minimal copy of the zoo: exactly one team per unique human per task.

The full zoo keeps every team (a human may be trained by several algorithms). This
tool produces a separate, pruned copy where each task has exactly TARGET distinct
humans, each represented by a single robot+human team, with the teams split evenly
across algorithms (TARGET/3 each) via the same max-matching used by verify_zoo_630.

The source zoo is NEVER modified — only read and copied from.

Usage:
    python prune_zoo_to_unique.py --source_zoo /pvc/assistax/zoo \
                                  --target_zoo /pvc/assistax/zoo_unique630 [--dry_run]
"""

import argparse
import shutil
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Set

import pandas as pd

ALGOS = ["IPPO", "MAPPO", "MASAC"]
TARGET = 630
CAP = TARGET // len(ALGOS)  # 210
Combo = tuple


def _key(w_speed: float, w_force: float, w_touch: float) -> Combo:
    return (f"{w_speed:.4f}", f"{w_force:.4f}", f"{w_touch:.4f}")


def assign_combos(combo_algos: Dict[Combo, Set[str]]) -> Dict[Combo, str]:
    """Max-matching: assign each combo to one algorithm it has, each algo capped at CAP."""
    combos = list(combo_algos)
    H = len(combos)
    src, sink = 0, H + len(ALGOS) + 1
    cap: Dict = defaultdict(int)
    adj: Dict = defaultdict(set)

    def add(u: int, v: int, c: int) -> None:
        cap[(u, v)] += c
        adj[u].add(v)
        adj[v].add(u)

    ai = {a: 1 + H + j for j, a in enumerate(ALGOS)}
    for i, c in enumerate(combos):
        add(src, 1 + i, 1)
        for a in combo_algos[c]:
            if a in ai:
                add(1 + i, ai[a], 1)
    for a in ALGOS:
        add(ai[a], sink, CAP)

    while True:
        parent = {src: src}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in parent and cap[(u, v)] > 0:
                    parent[v] = u
                    q.append(v)
        if sink not in parent:
            break
        v = sink
        while v != src:
            u = parent[v]
            cap[(u, v)] -= 1
            cap[(v, u)] += 1
            v = u

    assignment: Dict[Combo, str] = {}
    for i, c in enumerate(combos):
        for a in combo_algos[c]:
            if a in ai and cap[(1 + i, ai[a])] == 0:  # saturated forward edge -> assigned
                assignment[c] = a
                break
    return assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_zoo", required=True)
    parser.add_argument("--target_zoo", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    source = Path(args.source_zoo)
    df = pd.read_csv(source / "index.csv")

    keep_uuids: list = []
    all_ok = True
    print(f"{'task':18s} combos  assigned  IPPO/MAPPO/MASAC  teams_kept")
    for scen in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scen]
        # combo -> algorithm -> set(team_uuid)
        ca: Dict[Combo, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
        for r in sub.itertuples():
            ca[_key(float(r.w_speed), float(r.w_force), float(r.w_touch))][r.algorithm].add(r.team_uuid)
        combo_algos = {c: set(d) for c, d in ca.items()}
        assignment = assign_combos(combo_algos)

        per_algo: Dict[str, int] = defaultdict(int)
        chosen_teams: Set[str] = set()
        for c, a in assignment.items():
            chosen_teams.add(sorted(ca[c][a])[0])  # deterministic: first team_uuid
            per_algo[a] += 1

        kept = sub[sub["team_uuid"].isin(chosen_teams)]
        keep_uuids.extend(kept["agent_uuid"].tolist())
        ok = len(assignment) == TARGET and all(per_algo[a] == CAP for a in ALGOS)
        all_ok &= ok
        print(f"{scen:18s} {len(combo_algos):6d}  {len(assignment):8d}  "
              f"{per_algo['IPPO']:4d}/{per_algo['MAPPO']:4d}/{per_algo['MASAC']:4d}  "
              f"{len(chosen_teams):4d} ({len(kept)} agents){'' if ok else '  <-- not 630/210-210-210'}")

    print(f"\nTotal agents to keep: {len(keep_uuids)} (expect {TARGET * 2 * df['scenario'].nunique()})")
    if not all_ok:
        print("WARNING: at least one task is not a clean 630 / 210-210-210 — check the source zoo.")

    if args.dry_run:
        print("DRY RUN — nothing copied.")
        return

    target = Path(args.target_zoo)
    (target / "config").mkdir(parents=True, exist_ok=True)
    (target / "params").mkdir(parents=True, exist_ok=True)
    keep_set = set(keep_uuids)
    for u in keep_uuids:
        shutil.copy2(source / "config" / f"{u}.yaml", target / "config" / f"{u}.yaml")
        shutil.copy2(source / "params" / f"{u}.safetensors", target / "params" / f"{u}.safetensors")
    df[df["agent_uuid"].isin(keep_set)].to_csv(target / "index.csv", index=False)
    print(f"\nPruned zoo written to: {target}  ({len(keep_set)} agents)")


if __name__ == "__main__":
    main()
