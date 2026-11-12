"""The trainers' preference reward (compute_preference_reward), which population training calls with all of a
person's envs at once: a new touch is detected per env, as PreferenceRewardWrapper detects it inside one env."""
import unittest

import jax.numpy as jp
import numpy as np

from assistax.wrappers.training import compute_preference_reward

# norm factor B / (w_speed + w_force) = 2, so a new touch costs 2 * -0.2 = -0.4
PREF = dict(w_speed=0.5, w_force=0.5, w_touch=-0.2, speed_range_min=0.05, speed_range_max=0.15, force_range_min=1.0,
            force_range_max=3.0, reward_budget=2.0, overall_weight=1.0, touch_threshold=0.3)


class TestTouchPerEnv(unittest.TestCase):
    def test_only_the_touching_env_is_penalised(self) -> None:
        _, cf, comps = compute_preference_reward(speed=jp.full(4, 0.1), force=jp.array([0.0, 5.0, 0.0, 0.0]),
                                                 prev_contact_force=jp.zeros(4), **PREF)
        np.testing.assert_allclose(comps["touch_penalty_reward"], [0.0, -0.4, 0.0, 0.0], rtol=1e-6)
        np.testing.assert_allclose(cf, [0.0, 5.0, 0.0, 0.0])

    def test_a_held_touch_is_not_new(self) -> None:
        _, _, comps = compute_preference_reward(speed=jp.full(2, 0.1), force=jp.array([5.0, 5.0]),
                                                prev_contact_force=jp.array([5.0, 0.0]), **PREF)
        np.testing.assert_allclose(comps["touch_penalty_reward"], [0.0, -0.4], rtol=1e-6)

    def test_single_env_call_as_the_aht_wrappers_make_it(self) -> None:
        _, cf, comps = compute_preference_reward(speed=jp.array(0.1), force=jp.array(1.0),
                                                 prev_contact_force=jp.array(0.0), **PREF)
        self.assertAlmostEqual(float(comps["touch_penalty_reward"]), -0.4, places=6)
        self.assertAlmostEqual(float(cf), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
