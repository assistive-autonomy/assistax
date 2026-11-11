"""Feeding and tooth brushing: the human turns its head (neck yaw and pitch, as the paper says), and its back is
glued to the chair so that moving the head cannot tip it out of the seat (Leo, 2026-11-11)."""
import unittest

import numpy as np

from assistax import envs
from assistax.envs.base_env import _agent_action_mapping
from tests._rollout import rollout

TASKS = ("feeding", "teethbrushing")


def tilt_deg(xmat: np.ndarray, xmat0: np.ndarray) -> float:
    """Angle in degrees between a body's z axis now and at reset."""
    z, z0 = np.asarray(xmat).reshape(3, 3)[:, 2], np.asarray(xmat0).reshape(3, 3)[:, 2]
    return float(np.degrees(np.arccos(np.clip(z @ z0, -1.0, 1.0))))


class TestSeatedHuman(unittest.TestCase):
    def test_human_drives_neck_yaw_and_pitch(self) -> None:
        for task in TASKS:
            m = envs._envs[task]().sys.mj_model
            names = [m.actuator(int(i)).name for i in np.asarray(_agent_action_mapping[task]["human"])]
            self.assertEqual(names, ["neck_yaw", "neck_pitch"], task)

    def test_neck_has_spring_and_damper(self) -> None:
        """A passive spring (10 N m/rad) and damper (2 N m s/rad) hold the head up when the human does not act."""
        for task in TASKS:
            m = envs._envs[task]().sys.mj_model
            for joint in ("neck_z", "neck_y"):
                self.assertEqual(m.jnt_stiffness[m.joint(joint).id], 10.0, (task, joint))
                self.assertEqual(m.dof_damping[m.joint(joint).dofadr[0]], 2.0, (task, joint))

    def test_human_stays_seated_under_random_actions(self) -> None:
        """200 steps of uniform random actions: the back (torso) stays within 2 degrees of its reset orientation and
        the pelvis within 10 (it settles by ~5 degrees after reset on the passive abdomen joints). With abdomen
        control and a position-only seat weld, the human leaned and tipped by up to 27.7 degrees."""
        for task in TASKS:
            env, states = rollout(task, n_steps=200)
            m = env.sys.mj_model
            ps0 = states[0].pipeline_state
            for body, limit in (("torso", 2.0), ("pelvis", 10.0)):
                b = m.body(body).id
                worst = max(tilt_deg(s.pipeline_state.xmat[b], ps0.xmat[b]) for s in states[1:])
                self.assertLess(worst, limit, (task, body))


if __name__ == "__main__":
    unittest.main()
