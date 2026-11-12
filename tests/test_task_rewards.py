"""Task-reward terms changed for the population retrain (Leo, 2026-11-11)."""
import unittest

import jax.numpy as jp
import numpy as np

from assistax import envs
from tests._rollout import rollout


class TestArmAssistRotation(unittest.TestCase):
    def test_rotation_term_is_a_penalty(self) -> None:
        """The hook-to-target orientation difference ||R_target - R_tool||_F (0 when aligned, up to 2.83 when upside
        down) costs 1.0 per unit in v2 (Leo, 2026-11-12; -0.1 left the hook-less strategy in place). It used to pay
        +0.1, rewarding misalignment. The step reward is the sum of the weighted terms."""
        _, states = rollout("armmanipulation")
        for s in states[1:]:
            m = {k: float(v) for k, v in s.metrics.items()}
            self.assertGreater(m["reward_rot"], 0.0)
            self.assertAlmostEqual(m["weighted_reward_rot"], -1.0 * m["reward_rot"], places=5)
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


class TestFeedingTightTerm(unittest.TestCase):
    """Feeding v2 (Leo, 2026-11-12): the 0.1.0 reward plus a tight mouth term 2 exp(-d^2/0.005), d the spoon-centre-to-
    mouth distance, so that touching the mouth pays more than hovering near it. Weight 0 gives the 0.1.0 reward."""

    def test_weight_zero_is_the_0_1_0_reward(self) -> None:
        """With tight_dist_reward_weight=0 the step reward is the 0.1.0 sum of terms."""
        env, states = rollout("feeding", env_kwargs=(("tight_dist_reward_weight", 0.0),))
        for s in states[1:]:
            m = {k: float(v) for k, v in s.metrics.items()}
            expected = (2.0 * m["reward_dist"] + m["reward_orientation"] + m["reward_velocity"] + m["reward_force"]
                        + env._ctrl_cost_weight * m["reward_ctrl"])
            self.assertAlmostEqual(float(s.reward), expected, places=5)

    def test_v2_default_weight_is_2_and_scale_is_0_005(self) -> None:
        env = envs._envs["feeding"]()
        self.assertEqual((env._tight_dist_reward_weight, env._tight_dist_scale), (2.0, 0.005))

    def test_default_weight_adds_2_exp_minus_d2_over_scale(self) -> None:
        """At the default weight the same rollout's reward is 2 exp(-d^2/scale) above the 0.1.0 reward, and reward_tight
        reports the unweighted term. The rollout starts ~57 cm from the mouth, where exp(-d^2/0.005) is ~1e-29, so the
        test widens the scale to 1 m^2 to make the term visible."""
        _, base = rollout("feeding", env_kwargs=(("tight_dist_scale", 1.0), ("tight_dist_reward_weight", 0.0)))
        env, tight = rollout("feeding", env_kwargs=(("tight_dist_scale", 1.0),))
        for s0, s in zip(base[1:], tight[1:]):
            d = float(jp.linalg.norm(s.pipeline_state.site_xpos[env.panda_spoon_centre]
                                     - s.pipeline_state.site_xpos[env.human_mouth]))
            self.assertGreater(np.exp(-d**2), 0.5)
            self.assertAlmostEqual(float(s.metrics["reward_tight"]), np.exp(-d**2), places=5)
            self.assertAlmostEqual(float(s.reward) - float(s0.reward), 2.0 * np.exp(-d**2), places=4)


if __name__ == "__main__":
    unittest.main()
