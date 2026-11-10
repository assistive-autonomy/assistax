"""Wrapper behaviour that depends on MuJoCo indexing."""
import unittest

import numpy as np

from assistax import envs


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


if __name__ == "__main__":
    unittest.main()
