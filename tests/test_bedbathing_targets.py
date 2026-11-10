"""Bed-bathing wipe targets are generated in the arm capsules' geom frames, so they must be mapped to
world coordinates with the capsule (geom) pose, not the arm body pose."""
import unittest

import jax.numpy as jp
import numpy as np
from brax import base
from brax.envs.base import PipelineEnv

from tests._rollout import rollout

CAPSULES = ("right_uarm", "right_larm")  # targets 0-25 on the upper arm, 26-51 on the lower arm


def _pose(ps: base.State, geom_id: int) -> tuple[np.ndarray, np.ndarray]:
    """World rotation (3x3) and position of a geom in an MJX pipeline state."""
    return np.asarray(ps.geom_xmat[geom_id]).reshape(3, 3), np.asarray(ps.geom_xpos[geom_id])


def _geom_frame_targets(env: PipelineEnv, ps: base.State) -> np.ndarray:
    """The env's local cylinder targets mapped through each capsule's geom pose (reference mapping)."""
    m = env.sys.mj_model
    out = []
    for name, local in zip(CAPSULES, (env.wiping_targets_uarm, env.wiping_targets_larm)):
        rot, pos = _pose(ps, m.geom(name).id)
        out.append(np.asarray(local) @ rot.T + pos)
    return np.concatenate(out)


class TestBedBathingTargets(unittest.TestCase):
    def test_get_targets_distances_use_capsule_frame(self) -> None:
        """get_targets' 52 distances match targets mapped through the arm capsules' geom pose."""
        env, states = rollout("bedbathing")
        wiper = env.sys.mj_model.site("wiper_centre").id
        for state in states:
            ps = state.pipeline_state
            _, distances = env.get_targets(ps, jp.ones(env.n_targets))
            want = np.linalg.norm(_geom_frame_targets(env, ps) - np.asarray(ps.site_xpos[wiper]), axis=1)
            np.testing.assert_allclose(np.asarray(distances), want, atol=1e-5)

    def test_step_distance_reward_uses_capsule_frame(self) -> None:
        """step's distance reward uses the closest unwiped target in the capsule geom frame."""
        env, states = rollout("bedbathing")
        wiper = env.sys.mj_model.site("wiper_centre").id
        for prev, state in zip(states[:-1], states[1:]):
            ps = state.pipeline_state
            d = np.linalg.norm(_geom_frame_targets(env, ps) - np.asarray(ps.site_xpos[wiper]), axis=1)
            closest = d[np.asarray(prev.info["contact_vector"]) != 0].min()  # targets not wiped before this step
            np.testing.assert_allclose(float(state.metrics["reward_dist"]), np.exp(-closest**2 / 0.1), rtol=1e-4)

    def test_targets_lie_on_arm_surface_and_outside_mattress(self) -> None:
        """At reset every target lies on its arm capsule's surface and none is over 5 mm inside the mattress."""
        env, states = rollout("bedbathing")
        self.assertTrue(hasattr(env, "_get_global_targets"), "env must expose the targets it scores against")
        m, ps = env.sys.mj_model, states[0].pipeline_state
        world = np.asarray(env._get_global_targets(ps))
        self.assertEqual(world.shape, (env.n_targets, 3))
        half = env.n_targets // 2
        for name, pts in zip(CAPSULES, (world[:half], world[half:])):
            g = m.geom(name).id
            rot, pos = _pose(ps, g)
            local = (pts - pos) @ rot
            radius, half_length = m.geom_size[g][:2]
            np.testing.assert_allclose(np.linalg.norm(local[:, :2], axis=1), radius, atol=1e-5, err_msg=name)
            self.assertTrue(np.all(np.abs(local[:, 2]) <= half_length + 1e-6), name)
        g = m.geom("bed_mattress").id
        rot, pos = _pose(ps, g)
        local = np.abs((world - pos) @ rot)
        depth = np.where(np.all(local < m.geom_size[g], axis=1), (m.geom_size[g] - local).min(axis=1), 0.0)
        self.assertLess(depth.max(), 0.005, f"targets inside the mattress, max depth {depth.max():.4f} m")


if __name__ == "__main__":
    unittest.main()
