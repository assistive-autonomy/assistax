"""The v2 partner population's persons (population/manifest.py): 630, the same in every task, each with an algorithm,
one of the paper's 7 disabilities, its own preferences from the code's sampling ranges and its own training key."""
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from population import manifest as m

ROOT = Path(__file__).resolve().parents[1]
RANGES = OmegaConf.to_container(OmegaConf.load(ROOT / "assistax/baselines/IPPO/config/ippo_zoo_gen.yaml"))[
    "PREFERENCE_SWEEP"]
SEED = 20261112


class TestPersons(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.persons = m.build_persons(SEED)

    def test_counts(self) -> None:
        p = self.persons
        self.assertEqual([q["id"] for q in p], list(range(630)))
        self.assertEqual(Counter(q["algorithm"] for q in p), {"ippo": 210, "mappo": 210, "masac": 210})
        self.assertEqual(set(Counter((q["algorithm"], q["disability"]) for q in p).values()), {30})
        self.assertEqual(Counter(q["disability"] for q in p), {d: 90 for d in range(7)})
        self.assertEqual(m.DISABILITIES, ((0.0, 0.0), (0.0, 0.5), (0.0, 1.0), (0.5, 0.5), (0.5, 1.0), (1.0, 0.5),
                                          (1.0, 1.0)))
        for q in p:
            self.assertEqual((q["restriction"], q["strength"]), m.DISABILITIES[q["disability"]])

    def test_unique_prefs_and_keys(self) -> None:
        self.assertEqual(len({tuple(q["prefs"][k] for k in m.PREF_KEYS) for q in self.persons}), 630)
        self.assertEqual(len({tuple(q["key"]) for q in self.persons}), 630)

    def test_prefs_within_code_ranges(self) -> None:
        f32 = np.float32
        for q in self.persons:
            pr = q["prefs"]
            self.assertEqual(set(pr), set(m.PREF_KEYS))
            for k in ("w_speed", "w_force", "w_touch", "speed_range_min", "force_range_min"):
                self.assertTrue(f32(RANGES[k]["min"]) <= f32(pr[k]) < f32(RANGES[k]["max"]), (k, pr[k]))
            self.assertTrue(pr["speed_range_min"] < pr["speed_range_max"] < f32(RANGES["speed_range_max"]["max"]))
            self.assertTrue(pr["force_range_min"] < pr["force_range_max"] < f32(RANGES["force_range_max"]["max"]))

    def test_deterministic(self) -> None:
        self.assertEqual(m.build_persons(SEED), self.persons)
        self.assertNotEqual(m.build_persons(1), self.persons)

    def test_committed_manifest_matches_its_seed(self) -> None:
        man = m.load_manifest(m.MANIFEST)
        self.assertEqual(man["seed"], SEED)
        self.assertEqual(man["persons"], self.persons)

    def test_plan_blocks(self) -> None:
        blocks = m.plan_blocks({"ippo": 15, "mappo": 15, "masac": 10})
        self.assertEqual(Counter(a for a, _, _ in blocks), {"ippo": 14, "mappo": 14, "masac": 21})
        self.assertEqual(sorted(i for _, f, l in blocks for i in range(f, l + 1)), list(range(630)))
        for alg, first, last in blocks:
            cells = {(self.persons[i]["algorithm"], self.persons[i]["disability"]) for i in range(first, last + 1)}
            self.assertEqual(cells, {(alg, self.persons[first]["disability"])})


if __name__ == "__main__":
    unittest.main()
