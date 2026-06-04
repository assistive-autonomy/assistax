"""Preflight check for the zoo "fill to 630" jobs (no training).

Regenerates the preference-weight combos that the fill kube scripts will produce
(IPPO/MAPPO/MASAC each on a disjoint fresh seed range) and verifies, against the
live ``index.csv``, that they are the right size, mutually disjoint across
algorithms, and collision-free with existing humans. Run this BEFORE launching the
fill jobs (ideally in the cluster environment so the generated values match what
the cluster will produce).

Combos depend only on ``(SEED, num_configs, PREFERENCE_SWEEP ranges)`` — the three
configs share identical ranges, so disjoint seed ranges keep the algorithm groups
distinct. A "human" is identified by ``(w_speed, w_force, w_touch)`` rounded to 4 dp
(matching ``ippo_zoo_gen.py``).

Usage:
    python preview_fill_combos.py --index-path /pvc/assistax/zoo/index.csv
"""

import argparse
import os.path as osp
from typing import Dict, List, Set, Tuple

import jax
import pandas as pd
from omegaconf import OmegaConf

from assistax.baselines.utils import generate_preference_configs

# (algorithm, config yaml, seeds, num_configs-per-seed) — matches the fill scripts.
_HERE = osp.dirname(osp.abspath(__file__))
_CFG = osp.normpath(osp.join(_HERE, ".."))
FILL_PLAN: List[Tuple[str, str, List[int], int]] = [
    ("IPPO", osp.join(_CFG, "IPPO/config/ippo_zoo_gen.yaml"), [20, 21, 22], 70),
    ("MAPPO", osp.join(_CFG, "MAPPO/config/mappo_zoo_gen.yaml"), [23, 24, 25], 70),
    ("MASAC", osp.join(_CFG, "MASAC/config/masac_zoo_gen.yaml"), [26, 27, 28], 20),
]

Combo = Tuple[str, str, str]


def _key(w_speed: float, w_force: float, w_touch: float) -> Combo:
    """Canonical 4-decimal key for a human (matches the index's rounding)."""
    return (f"{w_speed:.4f}", f"{w_force:.4f}", f"{w_touch:.4f}")


def combos_for(config_path: str, seeds: List[int], num_configs: int) -> List[Combo]:
    """Generate the rounded (w_speed, w_force, w_touch) combos for given seeds."""
    cfg = OmegaConf.load(config_path)
    # Only the (interpolation-free) preference blocks are needed; avoid resolving
    # the whole config (hydra ${now:...} interpolations would fail here).
    sweep = OmegaConf.to_container(cfg.PREFERENCE_SWEEP, resolve=True)
    config = {"ENV_KWARGS": {"preference_rewards":
              OmegaConf.to_container(cfg.ENV_KWARGS.preference_rewards, resolve=True)}}
    sweep["num_configs"] = num_configs
    out: List[Combo] = []
    for seed in seeds:
        pref_rng = jax.random.split(jax.random.PRNGKey(seed), 3)[2]
        pc = generate_preference_configs(pref_rng, sweep, config)
        for i in range(num_configs):
            out.append(_key(float(pc["w_speed"][i]), float(pc["w_force"][i]), float(pc["w_touch"][i])))
    return out


def existing_combos(index_path: str) -> Set[Combo]:
    """Distinct (w_speed, w_force, w_touch) keys already in the index."""
    df = pd.read_csv(index_path)
    return {_key(float(r.w_speed), float(r.w_force), float(r.w_touch)) for r in df.itertuples()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", default=osp.join(_CFG, "..", "zoo", "index.csv"),
                        help="Path to the live zoo index.csv to check collisions against.")
    args = parser.parse_args()

    existing = existing_combos(args.index_path)
    print(f"Existing distinct humans in index: {len(existing)}\n")

    per_algo: Dict[str, List[Combo]] = {}
    ok = True
    for algo, cfg, seeds, n in FILL_PLAN:
        combos = combos_for(cfg, seeds, n)
        uniq = set(combos)
        target = len(seeds) * n
        dup_internal = len(combos) - len(uniq)
        clash = uniq & existing
        per_algo[algo] = combos
        print(f"{algo}: seeds={seeds} num_configs={n} -> {len(combos)} combos "
              f"(target {target}), unique={len(uniq)}, internal-dups={dup_internal}, "
              f"collisions-with-existing={len(clash)}")
        if len(uniq) != target or dup_internal or clash:
            ok = False

    # Cross-algorithm disjointness (different seeds must give different humans).
    algos = list(per_algo)
    for a in range(len(algos)):
        for b in range(a + 1, len(algos)):
            inter = set(per_algo[algos[a]]) & set(per_algo[algos[b]])
            print(f"{algos[a]} ∩ {algos[b]} = {len(inter)}")
            if inter:
                ok = False

    total_new = len(set().union(*[set(v) for v in per_algo.values()]))
    print(f"\nTotal distinct NEW humans: {total_new} (expected 480)")
    print("PREFLIGHT PASSED" if ok and total_new == 480 else "PREFLIGHT FAILED — adjust seeds")


if __name__ == "__main__":
    main()
