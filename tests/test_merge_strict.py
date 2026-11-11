"""Strict zoo merge for the v2 population (open issue F): shards of one zoo version go into a new, empty zoo, and
anything that would be dropped or mixed in silently is an error instead."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from assistax.baselines.tools.merge_zoo_shards import merge_zoos
from assistax.wrappers.aht import ZooManager


def shard(path: Path, speeds: list[float], version: str | None = "v2") -> Path:
    """A zoo shard with one human per speed preference weight (distinct configs, so distinct fingerprints)."""
    zoo = ZooManager(str(path), zoo_version=version)
    for w in speeds:
        cfg = {"ENV_NAME": "scratchitch", "ALGORITHM": "IPPO", "network": {"recurrent": False}, "OBS_DIM": 3,
               "ACT_DIM": 2, "ENV_KWARGS": {"preference_rewards": {"preference_weights": {"speed_preference": w}}}}
        zoo.save_agent(cfg, {"w": np.zeros(2)}, "human")
    return path


class TestStrictMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.shards, self.target = self.root / "shards", self.root / "zoo_v2"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_merges_into_a_new_zoo_marked_with_the_version(self) -> None:
        shard(self.shards / "b0", [0.1, 0.2])
        shard(self.shards / "b1", [0.3, 0.4])
        merge_zoos(str(self.shards), str(self.target), strict=True)
        self.assertEqual(len(pd.read_csv(self.target / "index.csv")), 4)
        self.assertEqual(len(list((self.target / "params").iterdir())), 4)
        self.assertEqual(json.loads((self.target / "zoo_meta.json").read_text())["zoo_version"], "v2")

    def test_duplicate_humans_are_an_error(self) -> None:
        shard(self.shards / "b0", [0.1, 0.2])
        shard(self.shards / "b1", [0.2])
        with self.assertRaises(ValueError):
            merge_zoos(str(self.shards), str(self.target), strict=True)
        self.assertFalse(self.target.exists())

    def test_target_must_be_new_or_empty(self) -> None:
        shard(self.shards / "b0", [0.1])
        shard(self.target, [0.9])
        with self.assertRaises(ValueError):
            merge_zoos(str(self.shards), str(self.target), strict=True)

    def test_shards_need_one_shared_version(self) -> None:
        shard(self.shards / "b0", [0.1])
        shard(self.shards / "b1", [0.2], version=None)
        with self.assertRaises(ValueError):
            merge_zoos(str(self.shards), str(self.target), strict=True)
        shard(self.shards / "b1x", [0.3], version="v3")
        with self.assertRaises(ValueError):
            merge_zoos(str(self.shards), str(self.target), strict=True)


if __name__ == "__main__":
    unittest.main()
