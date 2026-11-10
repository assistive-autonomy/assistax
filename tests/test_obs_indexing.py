"""Observation indexing: joint angles/velocities must be read at qpos/qvel *addresses* (not joint ids),
and every agent's slice of the observation vector must match the env's actual observation layout."""
import unittest

import jax
import numpy as np

import assistax
from assistax import envs
from assistax.envs import base_env
from tests._rollout import (
    HUMAN_ROBOT_ENVS,
    PANDA_JOINTS,
    human_hinge_names,
    joint_dofadr,
    joint_qposadr,
    joint_range,
    obs_dicts,
    rollout,
)

# Hand-derived block sizes (first agent, second agent) in observation order:
# robot = tool/target/force entries + 7 Panda joints (armmanipulation: 7 angles + 7 velocities);
# human = tool/target/force entries + every humanoid hinge (17, or 19 with the neck).
SPLITS = {
    "teethbrushing": (23, 35),  # robot 3+4+3+6+7, human 3+4+3+6+19
    "feeding": (23, 35),  # same layout as teethbrushing
    "scratchitch": (30, 40),  # robot 3+4+1+3+3+3+6+7, human 3+4+1+3+3+3+6+17
    "bedbathing": (26, 36),  # robot 3+4+3+3+6+7, human 3+4+3+3+6+17
    "armmanipulation": (30, 38),  # robot 7+7+3+3+1+9, human 3+1+17+17
    "pushcoop": (27, 27),  # unchanged
}
N_HUMAN_HINGES = {"teethbrushing": 19, "feeding": 19, "scratchitch": 17, "bedbathing": 17, "armmanipulation": 17}
FEEDING_PREFS = {
    "preference_weights": {"speed_preference": 0.25, "force_preference": 0.35},
    "preference_ranges": {"speed_range": [0.06, 0.14], "force_range": [1.5, 3.5]},
    "reward_budget": 1.0,
}


def _armmanip_normalised(q: np.ndarray, rng: np.ndarray) -> np.ndarray:
    """armmanipulation's [-1, 1] joint normalisation. The env names jnt_range[:, 0] "upper" and
    jnt_range[:, 1] "lower" (swapped, giving a mirrored map); 0.1.0 keeps that convention."""
    upper, lower = rng[:, 0], rng[:, 1]
    return 2 * (q - lower) / (upper - lower) - 1


def _obs_size(env_name: str) -> int:
    """Length of the raw env's observation vector, from a traced (not compiled) reset."""
    env = envs._envs[env_name]()
    return int(jax.eval_shape(env.reset, jax.random.PRNGKey(0)).obs.shape[-1])


class TestJointObservations(unittest.TestCase):
    def test_robot_joint_obs_are_panda_joints(self) -> None:
        """robo_joint_angles (and armmanipulation's robo_joint_vel) equal qpos/qvel at the 7 Panda joints, found by name."""
        for name in HUMAN_ROBOT_ENVS:
            with self.subTest(env=name):
                env, states = rollout(name)
                m = env.sys.mj_model
                for state in states:
                    robo, _ = obs_dicts(env, name, state)
                    q = np.asarray(state.pipeline_state.qpos)[joint_qposadr(m, PANDA_JOINTS)]
                    if name == "armmanipulation":
                        q = _armmanip_normalised(q, joint_range(m, PANDA_JOINTS))
                        qd = np.asarray(state.pipeline_state.qd)[joint_dofadr(m, PANDA_JOINTS)]
                        np.testing.assert_allclose(np.asarray(robo["robo_joint_vel"]), qd, atol=1e-5)
                    np.testing.assert_allclose(np.asarray(robo["robo_joint_angles"]), q, atol=1e-5)

    def test_human_joint_obs_are_human_hinges(self) -> None:
        """human_joint_angles (and armmanipulation's velocities) equal qpos/qvel at every humanoid hinge joint."""
        for name in HUMAN_ROBOT_ENVS:
            with self.subTest(env=name):
                env, states = rollout(name)
                m = env.sys.mj_model
                hinges = human_hinge_names(m)
                self.assertEqual(len(hinges), N_HUMAN_HINGES[name])
                for state in states:
                    _, human = obs_dicts(env, name, state)
                    q = np.asarray(state.pipeline_state.qpos)[joint_qposadr(m, hinges)]
                    if name == "armmanipulation":
                        q = _armmanip_normalised(q, joint_range(m, hinges))
                        qd = np.asarray(state.pipeline_state.qd)[joint_dofadr(m, hinges)]
                        np.testing.assert_allclose(np.asarray(human["human_joint_vel"]), qd, atol=1e-5)
                    np.testing.assert_allclose(np.asarray(human["human_joint_angles"]), q, atol=1e-5)


class TestAgentObservationSplit(unittest.TestCase):
    def test_two_agent_split_partitions_obs(self) -> None:
        """Each agent's index range covers exactly its block of the obs vector, and `global` covers all of it."""
        for name, (r, h) in SPLITS.items():
            with self.subTest(env=name):
                mapping = base_env._agent_observation_mapping[name]
                first, second = base_env._agent_action_mapping[name]
                self.assertEqual(_obs_size(name), r + h)
                np.testing.assert_array_equal(mapping[first], np.arange(r))
                np.testing.assert_array_equal(mapping[second], np.arange(r, r + h))
                np.testing.assert_array_equal(mapping["global"], np.arange(r + h))

    def test_handover_split_shares_object_block(self) -> None:
        """Handover robots see their own 31-entry block plus the shared 25-entry object/goal/phase block."""
        mapping = base_env._agent_observation_mapping["handover"]
        shared = np.arange(62, 87)  # object pos/quat/vel/angvel, goals, phase one-hot: 3+4+3+3+3+3+6
        self.assertEqual(_obs_size("handover"), 87)
        np.testing.assert_array_equal(mapping["robot1"], np.concatenate([np.arange(0, 31), shared]))
        np.testing.assert_array_equal(mapping["robot2"], np.concatenate([np.arange(31, 62), shared]))
        np.testing.assert_array_equal(mapping["global"], np.arange(87))

    def test_human_block_is_wider_than_robot_block(self) -> None:
        """The human block is wider than the robot block. AHT appends preference obs after the `max`
        padding while zoo training appends them after each agent's block; the two layouts only coincide
        while the human block is the widest."""
        for name in HUMAN_ROBOT_ENVS:
            with self.subTest(env=name):
                mapping = base_env._agent_observation_mapping[name]
                self.assertGreater(mapping["human"].size, mapping["robot"].size)

    def test_preference_obs_appended_after_each_agent_block(self) -> None:
        """With preference rewards, each agent's mapping is its own block followed by the same 7 preference entries."""
        env = assistax.make("feeding", preference_rewards=FEEDING_PREFS)
        r, h = SPLITS["feeding"]
        pref = np.arange(r + h, r + h + 7)
        np.testing.assert_array_equal(env.agent_obs_mapping["robot"], np.concatenate([np.arange(r), pref]))
        np.testing.assert_array_equal(env.agent_obs_mapping["human"], np.concatenate([np.arange(r, r + h), pref]))


class TestHandoverVelocities(unittest.TestCase):
    def test_handover_velocity_obs_read_dof_addresses(self) -> None:
        """Handover joint and object velocities come from qvel at dof addresses; the goal follows a 3-wide angvel."""
        # Hand-derived layout (handover._get_obs): per robot block of 31 entries, joint pos [0:9], joint vel [9:18];
        # shared block from 62: object pos [62:65], quat [65:69], vel [69:72], angvel [72:75], handover goal [75:78].
        env, states = rollout("handover")
        m = env.sys.mj_model
        obj = m.joint("handover_object_joint").dofadr[0]
        goal = m.site("handover_goal").id
        for state in states:
            obs, qvel = np.asarray(state.obs), np.asarray(state.pipeline_state.qvel)
            for k, start in ((1, 0), (2, 31)):
                names = [f"panda{k}_joint{i}" for i in range(1, 8)] + [f"panda{k}_finger_joint{i}" for i in (1, 2)]
                np.testing.assert_allclose(obs[start + 9 : start + 18], qvel[joint_dofadr(m, names)], atol=1e-5)
            np.testing.assert_allclose(obs[69:72], qvel[obj : obj + 3], atol=1e-5)
            np.testing.assert_allclose(obs[72:75], qvel[obj + 3 : obj + 6], atol=1e-5)
            np.testing.assert_allclose(obs[75:78], np.asarray(state.pipeline_state.site_xpos[goal]), atol=1e-5)


class TestObservationSpaces(unittest.TestCase):
    def test_declared_obs_spaces_match_returned_obs(self) -> None:
        """observation_space(agent).shape equals the returned obs shape for `None` and `concat` padding
        (`max` is excluded: both sides are built from max_agent_obs_size, so it cannot disagree)."""
        cases = [(name, None) for name in base_env._agent_observation_mapping]
        cases += [("scratchitch", "concat"), ("pushcoop", "concat")]
        for name, method in cases:
            with self.subTest(env=name, homogenisation=method):
                env = assistax.make(name, homogenisation_method=method)
                obs = env.map_global_obs_to_agents(np.zeros(env.agent_obs_mapping["global"].size))
                for agent in env.agents:
                    self.assertEqual(env.observation_space(agent).shape, obs[agent].shape)


if __name__ == "__main__":
    unittest.main()
