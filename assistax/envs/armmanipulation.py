from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
import mujoco 
from mujoco import mj_id2name, mj_name2id, mju_mat2Quat
from enum import IntEnum
from mujoco.mjx._src.support import contact_force
import numpy as np
# from brax import idcontacts

# TODO: Figure out how to actually use this function in the code
def contact_id(pipeline_state: State, id1: int, id2: int) -> int:
    """Returns the contact id between two geom ids."""
    mask = (pipeline_state.contact.geom == jp.array([id1, id2])) | (pipeline_state.contact.geom == jp.array([id2, id1])) 
    id = jp.all(mask, axis=0)   
    return id

class ArmManipulation(PipelineEnv):
    
    # TODO: Add docstring
    """
    Add docstring
    """

    # pyformat: enable

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-6,
        dist_reward_weight: float = 0.1,
        dist_scale: float = 0.1,
        # wiping_reward_weight: float = 1.0,
        reset_noise_scale=5e-3,
        backend="mjx",
        n_targets: int = 52,
        target_threshold: float = 0.05,
        **kwargs
    ):
        """Creates a BedBathing Environment.

        Args:
          ctrl_cost_weight: Weight for the control cost.
          reset_noise_scale: Scale of noise to add to reset states.
          backend: str, the physics backend to use
          **kwargs: Arguments that are passed to the base class.
        """
        self.path = epath.resource_path("brax") / "envs/assets/bed_scene_armmanip.xml"

        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        if backend == "mjx":
            self.sys = self.sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 1, # helps with stability, O(iterations x ls_iterations)
                    "opt.timestep": 0.001
                }
            )

        GEOM_IDX = mujoco.mjtObj.mjOBJ_GEOM
        BODY_IDX = mujoco.mjtObj.mjOBJ_BODY
        ACTUATOR_IDX = mujoco.mjtObj.mjOBJ_ACTUATOR
        SITE_IDX = mujoco.mjtObj.mjOBJ_SITE
        
        self.panda_actuators_ids = []
        self.humanoid_actuators_ids = []

        # TODO: Is this particioning of actuators useful in the training pipeline? 
        
        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if actuator_name.startswith("actuator"):
                self.panda_actuators_ids.append(i)
            else:
                self.humanoid_actuators_ids.append(i)


        self.panda_hplatform_idx = mj_name2id(mjmodel, GEOM_IDX, "hook_platform")
        self.panda_hend_idx = mj_name2id(mjmodel, GEOM_IDX, "hook_end")
        self.panda_hook_center_idx = mj_name2id(mjmodel, SITE_IDX, "platform_center")
        self.panda_hook_body_idx = mj_name2id(mjmodel, BODY_IDX, "hook")

        self.human_tuarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_upper_arm") # Right human arm tuarm = target arm upper arm
        self.human_tlarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_lower_arm") # Right human arm tlarm = target arm lower arm

        # ID of the location where we want the elbow to go
        self.hook_target_site = mj_name2id(mjmodel, SITE_IDX, "hook_target")
        self.arm_target_site = mj_name2id(mjmodel, SITE_IDX, "arm_target")

        self.human_tuarm_geom = mj_name2id(mjmodel, GEOM_IDX, "right_uarm")
        self.human_tlarm_geom = mj_name2id(mjmodel, GEOM_IDX, "right_larm")

        self.human_uarm_size = mjmodel.geom_size[self.human_tuarm_geom]
        self.human_larm_size = mjmodel.geom_size[self.human_tlarm_geom]

        self.human_uwaist_idx = mj_name2id(mjmodel, BODY_IDX, "uwaist")
        self.human_lwaist_idx = mj_name2id(mjmodel, BODY_IDX, "lwaist")
        
        # self.contact_force = jax.vmap(contact_force, in_axes=(None, 0, None, None))

        # self.TARGET_CONTACT_ID = 294
        
        # TODO: Update these indexes once XML is updated or write some sort of helper function to get these
        self.UARM_HPLATFORM_CONTACT_ID = 56
        self.LARM_HPLATFORM_CONTACT_ID = 60
        self.UARM_HEND_CONTACT_ID = 57
        self.LARM_HEND_CONTACT_ID = 62

        # TODO: Double check these or write a helper function to find these
        self.panda_joint_id_start = 18
        self.panda_joint_id_end = 24

        self.human_joint_id_start = 1
        self.human_joint_id_end = 18

        # Retrieve joint limits
        self.upper_joint_limits = mjmodel.jnt_range[:, 0]
        self.lower_joint_limits = mjmodel.jnt_range[:, 1]
        self.robot_upper_joint_limits = self.upper_joint_limits[self.panda_joint_id_start:self.panda_joint_id_end]
        self.robot_lower_joint_limits = self.lower_joint_limits[self.panda_joint_id_start:self.panda_joint_id_end]
        self.human_upper_joint_limits = self.upper_joint_limits[self.human_joint_id_start:self.human_joint_id_end]
        self.human_lower_joint_limits = self.lower_joint_limits[self.human_joint_id_start:self.human_joint_id_end]


        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._dist_reward_weight = dist_reward_weight
        
        self._dist_scale = dist_scale
        self._reset_noise_scale = reset_noise_scale
        # self.actuator_classes = self._get_actuator_classes(self.path)
        # self.humanoid_actuators, self.panda_actuators = self._identify_actuators(self.actuator_classes)

        # BedBathing specific indices
        # self.target_threshold = target_threshold
        # self.n_targets = n_targets
        # n_targets_per_arm = n_targets//2
        # self.wiping_targets_uarm = self._initialize_targets(n_targets_per_arm, mjmodel.geom_size[self.human_tuarm_geom][1], mjmodel.geom_size[self.human_tuarm_geom][0])
        # self.wiping_targets_larm = self._initialize_targets(n_targets_per_arm, mjmodel.geom_size[self.human_tlarm_geom][1], mjmodel.geom_size[self.human_tlarm_geom][0])

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        
        # contact_vector = jp.ones(self.n_targets)

        rng_pos, rng_vel = jax.random.split(rng, 2)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        init_q = self.sys.mj_model.keyframe("init").qpos
        #init_q = self.sys.init_q
        qpos = init_q + jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        # uarm_contact_id = contact_id(pipeline_state, self.human_tuarm_idx, self.panda_wiper_idx)
        # larm_contact_id = contact_id(pipeline_state, self.human_tlarm_idx, self.panda_wiper_idx)

        # self.LARM_TOOL_CONTACT_ID = jp.int32(larm_contact_id)
        # self.UARM_TOOL_CONTACT_ID = jp.int32(uarm_contact_id)

        robo_obs = self._get_robo_obs(pipeline_state)
        human_obs = self._get_human_obs(pipeline_state)

        # NOTE: IF THE LENGTH OF ANY OBSERVATIONS CHANGE, YOU MUST UPDATE THIS HERE:
        # jaxmarl/environments/mabrax/mappings.py#L111
        obs = jp.concatenate((
            # robot obs length = 6 + 6 + 3 + 3 + 1 + 9 = 28
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["tool_position"],
            # robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["tool_target_dist"].reshape((3,)),
            robo_obs["tool_target_dist_euclidean"].reshape((1,)),
            robo_obs["tool_target_dist_angular"].reshape((9,)),
            # human = 3 + 1 + 17 + 17 = 38
            human_obs["larm_waist_dist"].reshape((3,)),
            human_obs["larm_waist_dist_euclidean"].reshape((1,)),
            human_obs["human_joint_angles"], 
            human_obs["human_joint_vel"]       
          
        ))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_hook_dist": zero,
            "reward_rot": zero,
            "reward_waist_dist": zero,
            "reward_ctrl": zero,
            "weighted_reward_hook_dist": zero,
            "weighted_reward_waist_dist": zero,
            "weighted_reward_ctrl": zero,
            "weighted_reward_rot": zero
        }

        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        
        # global_targets_uarm = self._map_cylinder_points_to_global(self.wiping_targets_uarm, pipeline_state.xmat[self.human_tuarm_idx], pipeline_state.xpos[self.human_tuarm_idx])
        # global_targets_larm = self._map_cylinder_points_to_global(self.wiping_targets_larm, pipeline_state.xmat[self.human_tlarm_idx], pipeline_state.xpos[self.human_tlarm_idx])

        # global_targets = jp.vstack((global_targets_uarm, global_targets_larm))
        # distances_all = self._target_distances(global_targets, pipeline_state.site_xpos[self.panda_wiper_center_idx])
        # distances = self._mask_contacts(distances_all, self.contact_vector)

        ctrl_cost = -jp.sum(jp.square(action))
        robo_obs = self._get_robo_obs(pipeline_state)
        human_obs = self._get_human_obs(pipeline_state)
        
        # TODO: possibly replace uarm and larm positions with joint positions
        obs = jp.concatenate((
            # robot obs
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["tool_position"],
            # robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["tool_target_dist"].reshape((3,)),
            robo_obs["tool_target_dist_euclidean"].reshape((1,)),
            robo_obs["tool_target_dist_angular"].reshape((9,)),
            # human obs
            human_obs["larm_waist_dist"].reshape((3,)),
            human_obs["larm_waist_dist_euclidean"].reshape((1,)),
            human_obs["human_joint_angles"],
            human_obs["human_joint_vel"]       
        ))

        # elbow_stomach_dist = jp.exp((-jp.linalg.norm(human_obs["elbow_pos"] - human_obs["stomach_pos"]))**2/self._dist_scale)

        # tool_wrist_dist = jp.exp((-jp.linalg.norm(robo_obs["tool_position"] - robo_obs["wrist_pos"]))**2/self._dist_scale)


        # TODO: Add human preference rewards
        # COMPUTE REWARDS
        # 1) inverse distance reward for end-effector to reach itch target (tanh minimises dist faster)
        hook_arm_dist = robo_obs["tool_target_dist_euclidean"]
        r_hook_dist = (1 - jp.tanh(hook_arm_dist / self._dist_scale))
        self._dist_reward_weight = 1

        ang_dist = robo_obs["tool_target_dist_angular"]
        rot_scale = 0.1
        r_rot = jp.sqrt(jp.sum(ang_dist** 2)) 

        larm_waist_dist = human_obs["larm_waist_dist_euclidean"]
        r_waist_dist = (1 - jp.tanh(larm_waist_dist / self._dist_scale))

        waist_scale = 10
        reward = waist_scale*r_waist_dist + self._dist_reward_weight*r_hook_dist + self._ctrl_cost_weight*ctrl_cost + r_rot * rot_scale
        
        done = 0.0
        
        state.metrics.update(
            reward_hook_dist = r_hook_dist,
            reward_waist_dist = r_waist_dist,
            reward_ctrl = ctrl_cost,
            reward_rot= r_rot,
            weighted_reward_hook_dist = self._dist_reward_weight*r_hook_dist,
            weighted_reward_waist_dist = waist_scale*r_waist_dist,
            weighted_reward_ctrl = self._ctrl_cost_weight*ctrl_cost,
            weighted_reward_rot = r_rot * rot_scale
        )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )
        # return (robo_obs, human_obs)

    def xmat_to_quat(self, xmat):
        w = jp.sqrt(1 + xmat[0] + xmat[4] + xmat[8]) / 2
        x = (xmat[7] - xmat[5]) / (4*w)
        y = (xmat[2] - xmat[6]) / (4*w)
        z = (xmat[3] - xmat[1]) / (4*w)
        return jp.array([w, x, y, z])

    def _get_robo_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations."""

        # we want the 3d distance from panda hook to larm_lower
        tool_position = pipeline_state.site_xpos[self.panda_hook_center_idx]
        tool_orientation = pipeline_state.site_xmat[self.panda_hook_center_idx]
        # tool_orientation = self.xmat_to_quat(pipeline_state.site_xmat[self.panda_hook_center_idx].reshape(9))

        target_position = pipeline_state.site_xpos[self.hook_target_site]
        target_orientation = pipeline_state.site_xmat[self.hook_target_site]
        # target_orientation = self.xmat_to_quat(pipeline_state.site_xmat[self.hook_target_site].reshape(9))

        # calculate distances
        tool_target_dist = target_position - tool_position
        tool_target_dist_euclidean = jp.linalg.norm(tool_target_dist)
        # TEMPORARY: NEED TO FIX ROTATION!
        # tool_target_dist_angular = jp.zeros(9)
        tool_target_dist_angular = target_orientation - tool_orientation

        # # TODO: adjust this so the ._get_force_on_tool takes 3 args
        force_on_tool = self._get_force_on_tool(pipeline_state, self.UARM_HPLATFORM_CONTACT_ID, self.LARM_HPLATFORM_CONTACT_ID, self.UARM_HEND_CONTACT_ID, self.LARM_HEND_CONTACT_ID)

        # human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        robo_joint_angles = pipeline_state.qpos[self.panda_joint_id_start:self.panda_joint_id_end]
        normalised_robo_joint_angles = (robo_joint_angles - self.robot_lower_joint_limits) / (self.robot_upper_joint_limits - self.robot_lower_joint_limits)
        normalised_robo_joint_angles = 2 * normalised_robo_joint_angles - 1
        robo_joint_vel = pipeline_state.qd[self.panda_joint_id_start:self.panda_joint_id_end]
        # TODO: normalise joint velocities?

        # human_uarm_rot = pipeline_state.xmat[self.human_tuarm_idx]
        # human_larm_rot = pipeline_state.xmat[self.human_tlarm_idx]

        # uarm_radius, uarm_hlength, _  = self.sys.geom_size[self.human_tuarm_geom] 
        # larm_radius, larm_hlength, _ = self.sys.geom_size[self.human_tlarm_geom]

        # uarm_height_vector = jp.array([0, 0, uarm_hlength+uarm_radius])
        # larm_height_vector = jp.array([0, 0, larm_hlength+larm_radius])

        # shoulder_pos_norot = human_uarm_pos + uarm_height_vector
        
        # shoulder_pos = jp.dot(human_uarm_rot, shoulder_pos_norot)
        
        # wrist_pos_norot = human_larm_pos - larm_height_vector

        # wrist_pos = jp.dot(human_larm_rot, wrist_pos_norot)

        # elbow_pos_norot = human_larm_pos + larm_height_vector

        # elbow_pos = jp.dot(human_larm_rot, elbow_pos_norot)

        # stomach_pos = pipeline_state.xpos[self.human_uwaist_idx]
        # waist_pos = pipeline_state.xpos[self.human_lwaist_idx]

        return {
            # proprioception
            "robo_joint_angles": normalised_robo_joint_angles,
            "robo_joint_vel": robo_joint_vel,
            "tool_position": tool_position,
            # # tactile
            # "force_on_tool": force_on_tool,
            # ground truth
            "tool_target_dist": tool_target_dist,
            "tool_target_dist_euclidean": tool_target_dist_euclidean,
            "tool_target_dist_angular": tool_target_dist_angular
        }

       
    
    def _get_human_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations"""

        
        human_joint_angles = pipeline_state.qpos[self.human_joint_id_start:self.human_joint_id_end]
        normalised_human_joint_angles = (human_joint_angles - self.human_lower_joint_limits) / (self.human_upper_joint_limits - self.human_lower_joint_limits)
        normalised_human_joint_angles = 2 * normalised_human_joint_angles - 1
        human_joint_vel = pipeline_state.qd[self.human_joint_id_start:self.human_joint_id_end]

        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]

        # human_uarm_rot = pipeline_state.xmat[self.human_tuarm_idx]
        # human_larm_rot = pipeline_state.xmat[self.human_tlarm_idx]

        # uarm_radius, uarm_hlength, _  = self.sys.geom_size[self.human_tuarm_geom] 
        # larm_radius, larm_hlength, _ = self.sys.geom_size[self.human_tlarm_geom]

        # uarm_height_vector = jp.array([0, 0, uarm_hlength+uarm_radius])
        # larm_height_vector = jp.array([0, 0, larm_hlength+larm_radius])

        # shoulder_pos_norot = human_uarm_pos + uarm_height_vector
        
        # shoulder_pos = jp.dot(human_uarm_rot, shoulder_pos_norot)
        
        # wrist_pos_norot = human_larm_pos - larm_height_vector

        # wrist_pos = jp.dot(human_larm_rot, wrist_pos_norot)

        # elbow_pos_norot = human_larm_pos + larm_height_vector

        # elbow_pos = jp.dot(human_larm_rot, elbow_pos_norot)

        # stomach_pos = pipeline_state.xpos[self.human_uwaist_idx]
        # waist_pos = pipeline_state.xpos[self.human_lwaist_idx]

        # larm_waist_dist = waist_pos - human_larm_pos
        # larm_waist_dist_euclidean = jp.linalg.norm(larm_waist_dist)

        arm_pos = pipeline_state.site_xpos[self.hook_target_site]
        arm_target = pipeline_state.site_xpos[self.arm_target_site]
        larm_waist_dist = arm_target - arm_pos
        larm_waist_dist_euclidean = jp.linalg.norm(larm_waist_dist)

        # force_on_human = self._get_force_on_tool(pipeline_state, self.UARM_HPLATFORM_CONTACT_ID, self.LARM_HPLATFORM_CONTACT_ID, self.UARM_HEND_CONTACT_ID, self.LARM_HEND_CONTACT_ID)
        
        return {
            # "position": position,
            # "velocity": velocity,
            # "tool_position": tool_position,
            # "tool_orientation": tool_orientation,
            # # "distance_to_target": distance_to_target,
            # "tool_orientation": tool_orientation,
            # # "target_pos": target_pos,
            # "shoulder_pos": shoulder_pos,
            # "elbow_pos": elbow_pos,
            # "wrist_pos": wrist_pos,
            # "stomach_pos": stomach_pos,
            # "waist_pos": waist_pos,
            # "force_on_human": force_on_human,
            # "force_on_target": force_on_target,
            "human_joint_angles": normalised_human_joint_angles,
            "human_joint_vel": human_joint_vel,
            "larm_waist_dist": larm_waist_dist,
            "larm_waist_dist_euclidean": larm_waist_dist_euclidean,
            
        }
        #return jp.concatenate((position, velocity, distance_to_target, tool_orientation, target_pos, human_uarm_pos, human_larm_pos))
    
   
    def _get_force_on_tool(self, pipeline_state, uarm_tool1_id: int, larm_tool1_id:int, uarm_tool2_id: int, larm_tool2_id: int) -> jax.Array:
        """Return the force on the tool"""
        tool1_uarm = contact_force(self.sys, pipeline_state, uarm_tool1_id, False)
        tool1_larm = contact_force(self.sys, pipeline_state, larm_tool1_id, False)
        tool2_uarm = contact_force(self.sys, pipeline_state, uarm_tool2_id, False)
        tool2_larm = contact_force(self.sys, pipeline_state, larm_tool2_id, False)

        return jp.sum(jp.vstack((tool1_uarm, tool1_larm, tool2_uarm, tool2_larm)), axis=0)
