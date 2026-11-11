"""Zoo separation (open issue F, 2026-11-11): v1 and v2 humans never mix, and nothing in a zoo is overwritten."""
import json
import os
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np

from assistax.wrappers.aht import ZooManager

V1_INDEX = "agent_uuid,scenario,scenario_agent_id,algorithm,is_rnn,rnn_dim,team_uuid\n"
CONFIG = {"ENV_NAME": "scratchitch", "ALGORITHM": "IPPO", "network": {"recurrent": False}, "OBS_DIM": 3, "ACT_DIM": 2}


def make_v1_zoo(root: Path) -> Path:
    """A zoo as the v1 population left it: an older index format, one agent, no version marker."""
    for sub in ("config", "params"):
        (root / sub).mkdir(parents=True)
    (root / "index.csv").write_text(V1_INDEX + "abc,scratchitch,human,IPPO,False,0,\n")
    (root / "params" / "abc.safetensors").write_bytes(b"x")
    return root


class TestZooVersions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_new_zoo_is_marked_with_its_version(self) -> None:
        ZooManager(str(self.root / "zoo_v2"), zoo_version="v2")
        meta = json.loads((self.root / "zoo_v2" / "zoo_meta.json").read_text())
        self.assertEqual(meta["zoo_version"], "v2")

    def test_refuses_a_zoo_of_another_version(self) -> None:
        ZooManager(str(self.root / "zoo"), zoo_version="v2")
        with self.assertRaises(ValueError):
            ZooManager(str(self.root / "zoo"), zoo_version="v3")

    def test_refuses_an_unmarked_zoo_that_has_agents(self) -> None:
        v1 = make_v1_zoo(self.root / "zoo")
        with self.assertRaises(ValueError):
            ZooManager(str(v1), zoo_version="v2")
        self.assertFalse((v1 / "zoo_meta.json").exists())

    def test_opening_a_zoo_never_rewrites_it(self) -> None:
        v1 = make_v1_zoo(self.root / "zoo")
        before = {p: p.read_bytes() for p in v1.rglob("*") if p.is_file()}
        zoo = ZooManager(str(v1))
        self.assertIn("w_speed", zoo.index.columns)  # still migrated, in memory
        self.assertEqual({p: p.read_bytes() for p in v1.rglob("*") if p.is_file()}, before)

    def test_save_refuses_an_old_format_zoo(self) -> None:
        v1 = make_v1_zoo(self.root / "zoo")
        with self.assertRaises(ValueError):
            ZooManager(str(v1)).save_agent(dict(CONFIG), {"w": np.zeros(2)}, "human")
        self.assertEqual(len((v1 / "index.csv").read_text().splitlines()), 2)

    def test_save_never_overwrites_an_agent(self) -> None:
        zoo = ZooManager(str(self.root / "zoo_v2"), zoo_version="v2")
        fixed = uuid.UUID("00000000-0000-4000-8000-000000000001")
        with mock.patch("assistax.wrappers.aht.uuid.uuid4", return_value=fixed):
            zoo.save_agent(dict(CONFIG), {"w": np.zeros(2)}, "human")
            with self.assertRaises(FileExistsError):
                zoo.save_agent(dict(CONFIG), {"w": np.ones(2)}, "human")
        self.assertEqual(len(ZooManager(str(self.root / "zoo_v2"), zoo_version="v2").index), 1)

    def test_saved_config_records_the_zoo_version(self) -> None:
        zoo = ZooManager(str(self.root / "zoo_v2"), zoo_version="v2")
        zoo.save_agent(dict(CONFIG), {"w": np.zeros(2)}, "human")
        agent = ZooManager(str(self.root / "zoo_v2"), zoo_version="v2").index.agent_uuid[0]
        self.assertEqual(zoo._load_config(agent)["ZOO_VERSION"], "v2")


if __name__ == "__main__":
    unittest.main()
