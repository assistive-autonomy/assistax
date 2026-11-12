"""The v2 partner population's persons: 630 per task, the same 630 in every task (paper Sec. 3.4, App. A.3).

Each person has an algorithm (IPPO, MAPPO or MASAC), one of the paper's 7 arm-mobility disabilities (the
(joint_restriction_factor, joint_strength) pairs of the old population scripts), its own preferences sampled from the
code's PREFERENCE_SWEEP ranges (Leo, 2026-11-11: the code's ranges, not App. A.3 Table 7's), and its own training key.
With a key per person, a person's training no longer depends on which block it is trained in.

The persons are written once to manifest_v2.json and committed, so a later JAX version that samples differently
cannot change who was trained.

Usage, from the checkout root:
  python -m population.manifest population/manifest_v2.json --seed 20261112
"""
import argparse
import json
from pathlib import Path

import jax
from omegaconf import OmegaConf

from assistax.baselines.utils import generate_preference_configs

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("manifest_v2.json")
ALGORITHMS = ("ippo", "mappo", "masac")
DISABILITIES = ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.5, 0.5), (0.5, 1.0), (1.0, 0.5), (1.0, 1.0))
PER_CELL = 30
N_PERSONS = len(ALGORITHMS) * len(DISABILITIES) * PER_CELL
PREF_KEYS = ("w_speed", "w_force", "w_touch", "speed_range_min", "speed_range_max", "force_range_min",
             "force_range_max")


def build_persons(seed: int) -> list[dict]:
    """The 630 persons of one global seed. Person id = algorithm index * 210 + disability index * 30 + j; keys and
    preferences are drawn for all 630 at once, in id order."""
    zoo_gen = OmegaConf.to_container(OmegaConf.load(ROOT / "assistax/baselines/IPPO/config/ippo_zoo_gen.yaml"))
    train_root, pref_key = jax.random.split(jax.random.PRNGKey(seed))
    keys = jax.random.split(train_root, N_PERSONS)
    prefs = generate_preference_configs(pref_key, {**zoo_gen["PREFERENCE_SWEEP"], "num_configs": N_PERSONS}, zoo_gen)
    persons = []
    for i in range(N_PERSONS):
        a, d = i // (len(DISABILITIES) * PER_CELL), i // PER_CELL % len(DISABILITIES)
        persons.append({"id": i, "algorithm": ALGORITHMS[a], "disability": d, "restriction": DISABILITIES[d][0],
                        "strength": DISABILITIES[d][1], "key": [int(x) for x in keys[i]],
                        "prefs": {k: float(prefs[k][i]) for k in PREF_KEYS}})
    return persons


def write_manifest(path: Path, seed: int) -> None:
    """Write the persons of `seed` as JSON, one person per line."""
    lines = ",\n".join(json.dumps(p) for p in build_persons(seed))
    path.write_text(f'{{"seed": {seed}, "persons": [\n{lines}\n]}}\n')


def load_manifest(path: Path = MANIFEST) -> dict:
    """The manifest's seed and persons."""
    return json.loads(Path(path).read_text())


def plan_blocks(sizes: dict[str, int]) -> list[tuple[str, int, int]]:
    """Split each (algorithm, disability) cell into blocks of at most sizes[algorithm] consecutive persons, as
    (algorithm, first id, last id): a block never mixes algorithms or disabilities."""
    blocks = []
    for a, alg in enumerate(ALGORITHMS):
        for d in range(len(DISABILITIES)):
            start = (a * len(DISABILITIES) + d) * PER_CELL
            for first in range(start, start + PER_CELL, sizes[alg]):
                blocks.append((alg, first, min(first + sizes[alg], start + PER_CELL) - 1))
    return blocks


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    write_manifest(args.out, args.seed)
    print(f"wrote {N_PERSONS} persons (seed {args.seed}) to {args.out}")
