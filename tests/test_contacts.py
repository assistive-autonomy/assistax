"""Every hardcoded MJX contact-slot id must hold the geom pair it is meant to read.

MJX gives each model a fixed contact layout: slot ``i`` always holds the same geom pair, whether or not the
geoms touch. Env code reads contact forces by slot id, so an id silently changes meaning when collision pairs
are added or removed in a scene XML, or when a MuJoCo/MJX upgrade changes the layout. Run this test after any
such change; inspect slots with ``python -m assistax.envs.contact_utils <env> [--pair GEOM_A GEOM_B]``.
"""
import unittest

from assistax import envs
from assistax.envs.contact_utils import contact_geom_pairs, env_contact_ids


def _pairs(*pairs: tuple[str, str]) -> set[frozenset[str]]:
    """Allowed geom-name pairs for one attribute (order-insensitive)."""
    return {frozenset(p) for p in pairs}


# Intended geom pair(s) for every contact-slot attribute, keyed by env and attribute name.
EXPECTED = {
    "scratchitch": {
        "UARM_TOOL_CONTACT_ID": _pairs(("right_uarm1", "scratcher_stick")),
        "LARM_TOOL_CONTACT_ID": _pairs(("right_larm", "scratcher_stick")),
    },
    "bedbathing": {
        "UARM_TOOL_CONTACT_ID1": _pairs(("right_uarm", "wiper_pad")),
        "UARM_TOOL_CONTACT_ID2": _pairs(("right_uarm", "wiper_pad")),
        "LARM_TOOL_CONTACT_ID1": _pairs(("right_larm", "wiper_pad")),
        "LARM_TOOL_CONTACT_ID2": _pairs(("right_larm", "wiper_pad")),
    },
    "armmanipulation": {
        "UARM_HPLATFORM_CONTACT_ID": _pairs(("right_uarm", "hook_platform")),
        "LARM_HPLATFORM_CONTACT_ID": _pairs(("right_larm", "hook_platform")),
        "UARM_HEND_CONTACT_ID": _pairs(("right_uarm", "hook_end")),
        "LARM_HEND_CONTACT_ID": _pairs(("right_larm", "hook_end")),
    },
    "teethbrushing": {
        "TOOTHBRUSH_HEAD_CONTACT_ID": _pairs(("head", "toothbrush_head")),
        "TOOTHBRUSH_RSIDE_CONTACT_ID": _pairs(("head", "toothbrush_right_side")),
    },
    "feeding": {
        "SPOON_HEAD_CONTACT_ID": _pairs(("head", "spoon_bowl")),
        "SPOON_RSIDE_CONTACT_ID": _pairs(("head", "spoon_right_side")),
    },
    "handover": {
        **{
            f"panda{k}_{side}_finger_pad_contact{i}": _pairs((f"panda{k}_{side}_finger_pad", "box_object"))
            for k in (1, 2)
            for side in ("left", "right")
            for i in (1, 2, 3, 4)
        },
        **{f"object_floor_contact_id{i}": _pairs(("box_object", "floor")) for i in (1, 2, 3, 4)},
    },
    "pushcoop": {
        "contact_id_tmain": _pairs(("t_main", "floor")),
        "contact_id_tcross": _pairs(("t_cross", "floor")),
        "panda1_contact_id_t": _pairs(("panda1_pusher_stick", "t_main"), ("panda1_pusher_stick", "t_cross")),
        "panda2_contact_id_t": _pairs(("panda2_pusher_stick", "t_main"), ("panda2_pusher_stick", "t_cross")),
        **{f"contact_t_obs{n}": _pairs(("t_main", f"obs{n}"), ("t_cross", f"obs{n}")) for n in (1, 2, 4, 5)},
    },
}


class TestContactIds(unittest.TestCase):
    def test_every_contact_id_hits_its_intended_geom_pair(self) -> None:
        """Every contact-slot attribute of every env holds its intended geom pair, and none is unaccounted for."""
        for name, expected in EXPECTED.items():
            with self.subTest(env=name):
                env = envs.create(name, episode_length=None, auto_reset=False)
                m = env.sys.mj_model
                pairs = contact_geom_pairs(m)
                found = env_contact_ids(env)
                self.assertEqual(set(found), set(expected), "contact attributes changed: update EXPECTED")
                for attr, slots in found.items():
                    for slot in slots:
                        held = frozenset((m.geom(int(pairs[slot, 0])).name, m.geom(int(pairs[slot, 1])).name))
                        self.assertIn(held, expected[attr], f"{name}.{attr} slot {slot} holds {sorted(held)}")


if __name__ == "__main__":
    unittest.main()
