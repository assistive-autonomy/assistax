"""Arm assist: the scene gives the human's three arm actuators gear 0 (no torque); `human_gear` overrides it.
v2 (Leo, 2026-11-12) defaults to gear 2: very weak, so the human cannot do the task alone."""
import unittest

from assistax import envs

ARM = ("right_shoulder1", "right_shoulder2", "right_elbow")


class TestHumanGear(unittest.TestCase):
    def test_v2_default_is_gear_2(self) -> None:
        m = envs._envs["armmanipulation"]().sys.mj_model
        self.assertEqual([float(m.actuator_gear[m.actuator(n).id, 0]) for n in ARM], [2.0, 2.0, 2.0])

    def test_none_keeps_the_scene_gear(self) -> None:
        m = envs._envs["armmanipulation"](human_gear=None).sys.mj_model
        self.assertEqual([float(m.actuator_gear[m.actuator(n).id, 0]) for n in ARM], [0.0, 0.0, 0.0])

    def test_human_gear_sets_the_three_arm_actuators_only(self) -> None:
        m0 = envs._envs["armmanipulation"](human_gear=None).sys.mj_model
        m = envs._envs["armmanipulation"](human_gear=5.0).sys.mj_model
        self.assertEqual([float(m.actuator_gear[m.actuator(n).id, 0]) for n in ARM], [5.0, 5.0, 5.0])
        others = [i for i in range(m.nu) if m.actuator(i).name not in ARM]
        self.assertTrue((m.actuator_gear[others] == m0.actuator_gear[others]).all())


if __name__ == "__main__":
    unittest.main()
