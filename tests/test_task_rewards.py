"""Task-reward terms changed for the population retrain (Leo, 2026-11-11)."""
import unittest

import jax.numpy as jp
import numpy as np

from assistax import envs
from tests._rollout import rollout


class TestArmAssistRotation(unittest.TestCase):
    def test_rotation_term_is_a_penalty(self) -> None:
        """The hook-to-target orientation difference ||R_target - R_tool||_F (0 when aligned, up to 2.83 when upside
        down) costs 0.1 per unit. It used to pay +0.1, rewarding misalignment. The step reward is the sum of the
        weighted terms."""
        _, states = rollout("armmanipulation")
        for s in states[1:]:
            m = {k: float(v) for k, v in s.metrics.items()}
            self.assertGreater(m["reward_rot"], 0.0)
            self.assertAlmostEqual(m["weighted_reward_rot"], -0.1 * m["reward_rot"], places=5)
            total = sum(m[f"weighted_reward_{k}"] for k in ("waist_dist", "hook_dist", "ctrl", "rot"))
            self.assertAlmostEqual(float(s.reward), total, places=4)


class TestScratchReward(unittest.TestCase):
    """Scratch v6: the 0.1.0 reward plus a tight distance term exp(-d^2/0.005), with the scratching bonus paid only
    within 2.5 cm of the itch (it was 10 cm)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.env = envs._envs["scratchitch"]()

    def terms(self, distance: float, speed: float = 0.1, force: float = 3.0) -> dict:
        """The reward terms at a scratcher-tip-to-itch distance (m), tip speed (m/s) and contact force (N)."""
        out = self.env._reward_terms(jp.array(distance), jp.array(speed), jp.array(force))
        return {k: float(v) for k, v in out.items()}

    def test_distance_terms(self) -> None:
        for d in (0.0, 0.02, 0.05, 0.2):
            t = self.terms(d)
            self.assertAlmostEqual(t["reward_dist"], np.exp(-d**2 / 0.1), places=6)
            self.assertAlmostEqual(t["reward_tight"], np.exp(-d**2 / 0.005), places=6)

    def test_scratching_bonus_is_gated_at_2_5_cm(self) -> None:
        peak = np.exp(-1.0) * np.exp(-1.0)  # speed at its 0.1 m/s target and force at its 3 N target
        self.assertAlmostEqual(self.terms(0.02)["reward_scratching"], peak, places=6)
        for d in (0.03, 0.05, 0.09):
            self.assertEqual(self.terms(d)["reward_scratching"], 0.0, d)

    def test_step_reward_is_the_weighted_sum(self) -> None:
        env, states = rollout("scratchitch")
        for s in states[1:]:
            m = {k: float(v) for k, v in s.metrics.items()}
            expected = (m["reward_dist"] + m["reward_tight"] + 4.0 * m["reward_scratching"]
                        + env._ctrl_cost_weight * m["reward_ctrl"])
            self.assertAlmostEqual(float(s.reward), expected, places=5)


if __name__ == "__main__":
    unittest.main()
