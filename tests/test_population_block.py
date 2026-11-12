"""The v2 population's block runner (population/run_block.py): a block trains persons of one (task, algorithm,
disability) cell with each person's own key and preferences, and writes a v2 zoo shard that is never overwritten.
Tiny budgets on the CPU (two IPPO compiles, a few minutes)."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jax
import numpy as np

from assistax.wrappers.aht import ZooManager
from population import run_block as rb
from population.manifest import load_manifest

PERSONS = load_manifest()["persons"]
TINY = ["++NUM_ENVS=4", "++NUM_STEPS=8", "++TOTAL_TIMESTEPS=64", "++NUM_MINIBATCHES=4"]  # 2 updates


class TestBlock(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.block = rb.train_block("scratchitch", PERSONS[0:2], TINY)
        cls.shifted = rb.train_block("scratchitch", PERSONS[1:3], TINY)

    def test_person_trains_the_same_in_any_block_of_the_same_size(self) -> None:
        # Person 1 in slot 1 of [0, 1] and slot 0 of [1, 2]: its own key and preferences, nothing from its partner.
        # (Blocks of different sizes compile to different kernels; stiff servos and Adam then amplify the rounding,
        # so a person trained alone is not float-equal to the same person in a block.)
        in_block = [np.asarray(x[1]) for x in jax.tree.leaves(self.block["params"])]
        shifted = [np.asarray(x[0]) for x in jax.tree.leaves(self.shifted["params"])]
        for a, b in zip(in_block, shifted):
            np.testing.assert_array_equal(a, b)
        other = [np.asarray(x[0]) for x in jax.tree.leaves(self.block["params"])]  # person 0 trains differently
        self.assertFalse(all(np.allclose(a, b) for a, b in zip(other, in_block)))

    def test_compile_only_reports_memory_without_training(self) -> None:
        res = rb.train_block("scratchitch", PERSONS[0:2], TINY, compile_only=True)
        self.assertNotIn("params", res)
        self.assertGreater(res["memory"]["temp_bytes"] + res["memory"]["argument_bytes"], 0)
        self.assertEqual(res["config"]["OBS_DIM"], self.block["config"]["OBS_DIM"])

    def test_save_block_flags_non_finite_persons(self) -> None:
        bad = dict(self.block, params=jax.tree.map(lambda x: x.at[1].set(np.nan), self.block["params"]),
                   final_returns=self.block["final_returns"].copy())
        bad["final_returns"][1] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            summary = rb.save_block(Path(tmp) / "b", "scratchitch", PERSONS[0:2], bad, smoke=True)
            text = (Path(tmp) / "b" / "block.json").read_text()
        self.assertEqual(summary["params_finite"], [True, False])
        self.assertIsNone(summary["final_return"][1])
        json.loads(text, parse_constant=lambda c: self.fail(f"non-strict JSON constant {c}"))

    def test_main_reads_git_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rb, "_git", side_effect=RuntimeError("no git here")), \
                mock.patch.object(rb, "train_block") as train:
            with self.assertRaises(RuntimeError):
                rb.main(["population/manifest_v2.json", "scratchitch", "0-1", tmp])
            train.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_rejects_mixed_cell(self) -> None:
        for persons in ([PERSONS[0], PERSONS[30]], [PERSONS[0], PERSONS[210]]):
            with self.assertRaises(ValueError):
                rb.train_block("scratchitch", persons, TINY)

    def test_save_block_writes_v2_zoo_and_refuses_existing_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / rb.block_name("scratchitch", "ippo", 0, 1)
            summary = rb.save_block(out, "scratchitch", PERSONS[0:2], self.block, smoke=True)
            self.assertEqual(out.name, "scratchitch-ippo-p000-001")
            self.assertEqual(json.loads((out / "zoo" / "zoo_meta.json").read_text())["zoo_version"], "v2")
            zoo = ZooManager(str(out / "zoo"), zoo_version="v2")
            self.assertEqual(sorted(zoo.index["scenario_agent_id"]), ["human", "human", "robot", "robot"])
            self.assertEqual(zoo.index.groupby("team_uuid").size().tolist(), [2, 2])
            for uuid in zoo.index["agent_uuid"]:
                cfg = zoo._load_config(uuid)
                p = PERSONS[cfg["POPULATION"]["person"]]
                self.assertEqual(cfg["POPULATION"]["key"], p["key"])
                self.assertEqual(cfg["ENV_KWARGS"]["preference_rewards"]["preference_weights"]["speed_preference"],
                                 p["prefs"]["w_speed"])
                self.assertEqual(cfg["ZOO_VERSION"], "v2")
                zoo.load_agent(uuid)
            saved = json.loads((out / "block.json").read_text())
            self.assertEqual(saved, summary)
            self.assertEqual((saved["persons"], saved["smoke"], len(saved["final_return"])), ([0, 1], True, 2))
            metrics = np.load(out / "metrics.npz")
            self.assertEqual(metrics["returned_episode_returns"].shape[0], 2)
            with self.assertRaises(FileExistsError):
                rb.save_block(out, "scratchitch", PERSONS[0:2], self.block, smoke=True)


if __name__ == "__main__":
    unittest.main()
