"""Wrapper behaviour that depends on MuJoCo indexing, and the preference-reward weights."""
import types
import unittest

import jax
import jax.numpy as jp
import numpy as np

import assistax
from assistax import envs
from assistax.wrappers.training import PreferenceRewardWrapper

# The README's feeding example: it sets two of the three preference weights and leaves out touch_penalty.
README_PREFS = {
    "preference_weights": {"speed_preference": 0.25, "force_preference": 0.35},
    "preference_ranges": {"speed_range": [0.06, 0.14], "force_range": [1.5, 3.5]},
    "reward_budget": 1.0,
}


class TestDisabilityWrapper(unittest.TestCase):
    def test_range_restriction_hits_joint_of_masked_actuator(self) -> None:
        """`joint_idx` indexes the action vector (actuators), so the range restriction must hit the joint that
        actuator drives. In the 17-hinge scenes actuator 13 drives right_elbow (joint id 14), while joint id 13
        is right_shoulder2."""
        env = envs.create("scratchitch", disability={"joint_idx": 13, "joint_restriction_factor": 0.5})
        sys = env.unwrapped.sys
        m = sys.mj_model  # the compiled model, untouched by the wrapper
        self.assertEqual(m.actuator(13).name, "right_elbow")
        np.testing.assert_array_equal(np.flatnonzero(np.asarray(env.disability_mask)), [13])
        elbow, shoulder2 = m.joint("right_elbow").id, m.joint("right_shoulder2").id
        np.testing.assert_allclose(np.asarray(sys.jnt_range[elbow]), 0.5 * m.jnt_range[elbow], rtol=1e-6)
        np.testing.assert_allclose(np.asarray(sys.jnt_range[shoulder2]), m.jnt_range[shoulder2], rtol=1e-6)


def _with_touch_weight(weight: float) -> dict:
    """README_PREFS with an explicit touch_penalty weight."""
    return {**README_PREFS, "preference_weights": {**README_PREFS["preference_weights"], "touch_penalty": weight}}


class TestPreferenceRewardWrapper(unittest.TestCase):
    def _terms(self, prefs: dict) -> dict:
        """The wrapper's three preference reward terms for a step that starts a new touch (contact force 1.0
        after 0.0, above the default touch threshold 0.3), at speed 0.1 and force 2.0 (both in range)."""
        prev = types.SimpleNamespace(info={"preference_tracking": {"last_contact_force": jp.array(0.0)}})
        try:
            terms = PreferenceRewardWrapper(None, prefs)._compute_preference_rewards(
                jp.array(0.1), jp.array(2.0), jp.array(1.0), prev)
        except KeyError as e:
            self.fail(f"preference_weights without {e} raised KeyError")
        return {k: float(v) for k, v in terms.items()}

    def test_readme_example_steps(self) -> None:
        """The README's feeding example (no touch_penalty weight) resets and steps."""
        env = assistax.make("feeding", preference_rewards=README_PREFS)
        key = jax.random.PRNGKey(0)
        _, state = jax.jit(env.reset)(key)
        actions = {a: jp.zeros(env.action_spaces[a].shape) for a in env.agents}
        try:
            _, _, rewards, _, _ = jax.jit(env.step)(key, state, actions)
        except KeyError as e:
            self.fail(f"step with preference_weights missing {e} raised KeyError")
        for agent, r in rewards.items():
            self.assertTrue(np.all(np.isfinite(np.asarray(r))), agent)

    def test_missing_weight_pays_what_agent_observes(self) -> None:
        """A weight left out of preference_weights is observed as 0.0, so it must also pay 0.0."""
        self.assertEqual(float(PreferenceRewardWrapper(None, README_PREFS)._pref_obs_vector[2]), 0.0)
        self.assertLess(self._terms(_with_touch_weight(-0.1))["touch_penalty_reward"], 0.0)  # the probe is a touch
        self.assertEqual(self._terms(README_PREFS), self._terms(_with_touch_weight(0.0)))


if __name__ == "__main__":
    unittest.main()
