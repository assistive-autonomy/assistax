"""Feeding and tooth brushing: the human turns its head (neck yaw and pitch, as the paper says), and its back is
glued to the chair so that moving the head cannot tip it out of the seat (Leo, 2026-11-11)."""
import unittest

import mujoco
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

    def test_head_turns_freely(self) -> None:
        """Full yaw effort for 1 s (plain MuJoCo, robot holding its pose) turns the head at least 30 degrees each
        way, as in the scene before the back weld (60 degrees). A back weld that presses the head into the headrest
        (660-860 N) let it turn only 4.6 degrees."""
        for task in TASKS:
            m = envs._envs[task]().sys.mj_model
            adr = m.joint("neck_z").qposadr[0]
            for sign in (1.0, -1.0):
                d = mujoco.MjData(m)
                mujoco.mj_resetDataKeyframe(m, d, m.key("init").id)
                mujoco.mj_forward(m, d)
                for r in [i for i in range(m.nu) if m.actuator(i).name.startswith("actuator")]:
                    q = d.qpos[m.jnt_qposadr[m.actuator_trnid[r, 0]]]
                    d.ctrl[r] = -m.actuator_biasprm[r, 1] * q / m.actuator_gainprm[r, 0]
                d.ctrl[m.actuator("neck_yaw").id] = sign * m.actuator_ctrlrange[m.actuator("neck_yaw").id, 1]
                for _ in range(int(1.0 / m.opt.timestep)):
                    mujoco.mj_step(m, d)
                self.assertGreater(sign * np.degrees(d.qpos[adr]), 30.0, (task, sign))

    def test_human_stays_seated_under_random_actions(self) -> None:
        """200 steps of uniform random actions. From the reset keyframe the back settles into its glued pose (the
        pose the original scene settles into, leaned 1 degree forward: ~4.6 degrees), then holds within 2 degrees
        (the weld is soft: random head and robot torques wobble it by ~1 degree, with no drift); the pelvis stays
        within 10 degrees. With abdomen control and a position-only seat weld, the human leaned and tipped by up to
        27.7 degrees."""
        settle = 25  # steps, 0.2 s
        for task in TASKS:
            env, states = rollout(task, n_steps=200)
            m = env.sys.mj_model
            ps0, pss = states[0].pipeline_state, states[settle].pipeline_state
            torso, pelvis = m.body("torso").id, m.body("pelvis").id
            self.assertLess(max(tilt_deg(s.pipeline_state.xmat[torso], ps0.xmat[torso]) for s in states[1:]), 6.0,
                            (task, "torso settling"))
            self.assertLess(max(tilt_deg(s.pipeline_state.xmat[torso], pss.xmat[torso]) for s in states[settle:]), 2.0,
                            (task, "torso after settling"))
            self.assertLess(max(tilt_deg(s.pipeline_state.xmat[pelvis], ps0.xmat[pelvis]) for s in states[1:]), 10.0,
                            (task, "pelvis"))


if __name__ == "__main__":
    unittest.main()
