"""Verify the zoo is balanced to 630 humans per task, evenly split across algorithms.

For each scenario, reports the number of distinct humans (``(w_speed, w_force,
w_touch)`` rounded to 4 dp) and the best disjoint assignment of those humans to the
three algorithms (each capped at TARGET/3). The goal state is 630 distinct with a
210/210/210 partition. Also checks human/robot team pairing. Read-only.

Run before the fill jobs to see the current gap, and after merging the fill shards
to confirm the target is met.

Usage:
    python verify_zoo_630.py --index-path /pvc/assistax/zoo/index.csv
"""

import argparse
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple

import pandas as pd

ALGOS = ["IPPO", "MAPPO", "MASAC"]
TARGET = 630
CAP = TARGET // len(ALGOS)  # 210
Combo = Tuple[str, str, str]


def _key(w_speed: float, w_force: float, w_touch: float) -> Combo:
    return (f"{w_speed:.4f}", f"{w_force:.4f}", f"{w_touch:.4f}")


def best_partition(combo_algos: Dict[Combo, Set[str]]) -> Dict[str, int]:
    """Max disjoint assignment of humans to algorithms, each algo capped at CAP."""
    humans = [frozenset(s & set(ALGOS)) for s in combo_algos.values()]
    H = len(humans)
    src, sink = 0, 1 + H + len(ALGOS) + 1 - 1
    cap: Dict[Tuple[int, int], int] = defaultdict(int)
    adj: Dict[int, Set[int]] = defaultdict(set)

    def add_edge(u: int, v: int, c: int) -> None:
        cap[(u, v)] += c
        adj[u].add(v)
        adj[v].add(u)

    ai = {a: 1 + H + j for j, a in enumerate(ALGOS)}
    for i in range(H):
        add_edge(src, 1 + i, 1)
        for a in humans[i]:
            add_edge(1 + i, ai[a], 1)
    for a in ALGOS:
        add_edge(ai[a], sink, CAP)

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
    return {a: CAP - cap[(ai[a], sink)] for a in ALGOS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.index_path)
    tasks = sorted(df["scenario"].unique())

    # team pairing check
    bad_teams: List[str] = []
    for tid, grp in df.groupby("team_uuid"):
        roles = set(grp["scenario_agent_id"])
        if not {"robot", "human"}.issubset(roles):
            bad_teams.append(str(tid))

    print(f"{'task':18s} {'distinct':>8s}  partition(IPPO/MAPPO/MASAC)  status")
    all_ok = True
    for t in tasks:
        sub = df[df["scenario"] == t]
        ca: Dict[Combo, Set[str]] = defaultdict(set)
        for r in sub.itertuples():
            ca[_key(float(r.w_speed), float(r.w_force), float(r.w_touch))].add(r.algorithm)
        distinct = len(ca)
        part = best_partition(ca)
        ok = distinct >= TARGET and all(part[a] == CAP for a in ALGOS)
        all_ok &= ok
        print(f"{t:18s} {distinct:>8d}  {part['IPPO']:>4d}/{part['MAPPO']:>4d}/{part['MASAC']:>4d}"
              f"            {'OK' if ok else 'short'}")

    print(f"\nteams missing a robot/human partner: {len(bad_teams)}")
    print("ALL TASKS AT 630 WITH 210/210/210" if all_ok and not bad_teams
          else "NOT yet at target (see rows above)")


if __name__ == "__main__":
    main()
