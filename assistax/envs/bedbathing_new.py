from typing import Tuple

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
from jax.scipy.spatial.transform import Rotation
import mujoco 
from mujoco import mj_id2name, mj_name2id
from enum import IntEnum
from mujoco.mjx._src.support import contact_force
import numpy as np
# from brax import idcontacts

def contact_id(pipeline_state: State, id1: int, id2: int) -> int:
    """Returns the contact id between two geom ids."""
    mask = (pipeline_state.contact.geom == jp.array([id1, id2])) | (pipeline_state.contact.geom == jp.array([id2, id1])) 
    id = jp.all(mask, axis=0)   
    return id

class BedBathing(PipelineEnv):
    
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
        wiping_reward_weight: float = 1.0,
        reset_noise_scale=5e-3,
        backend="mjx",
        n_targets: int = 52,
        target_threshold: float = 0.1,
        **kwargs
    ):
        """Creates a BedBathing Environment.

        Args:
          ctrl_cost_weight: Weight for the control cost.
          reset_noise_scale: Scale of noise to add to reset states.
          backend: str, the physics backend to use
          **kwargs: Arguments that are passed to the base class.
        """
        self.path = epath.resource_path("brax") / "envs/assets/bed_scene.xml"

        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        if backend == "mjx":
            self.sys = self.sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 1,
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


        self.panda_wiper_idx = mj_name2id(mjmodel, GEOM_IDX, "wiper_pad")
        self.panda_wiper_center_idx = mj_name2id(mjmodel, SITE_IDX, "wiper_centre")
        self.panda_wiper_body_idx = mj_name2id(mjmodel, BODY_IDX, "wiper")

        self.human_tuarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_upper_arm") # Right human arm tuarm = target arm upper arm
        self.human_tlarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_lower_arm") # Right human arm tlarm = target arm lower arm

        self.human_tuarm_geom = mj_name2id(mjmodel, GEOM_IDX, "right_uarm")
        self.human_tlarm_geom = mj_name2id(mjmodel, GEOM_IDX, "right_larm")

        self.human_uarm_size = mjmodel.geom_size[self.human_tuarm_geom]
        self.human_larm_size = mjmodel.geom_size[self.human_tlarm_geom]
        
        # self.contact_force = jax.vmap(contact_force, in_axes=(None, 0, None, None))

        # self.TARGET_CONTACT_ID = 294
        
        # TODO: Update these indexes once XML is updated or write some sort of helper function to get these
        self.UARM_TOOL_CONTACT_ID = 58
    
        self.LARM_TOOL_CONTACT_ID = 62

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
        self._wiping_reward_weight = wiping_reward_weight
        self._dist_scale = dist_scale
        self._reset_noise_scale = reset_noise_scale
        # self.actuator_classes = self._get_actuator_classes(self.path)
        # self.humanoid_actuators, self.panda_actuators = self._identify_actuators(self.actuator_classes)

        # BedBathing specific indices
        self.target_threshold = target_threshold
        self.n_targets = n_targets
        n_targets_per_arm = n_targets//2
        self.wiping_targets_uarm = self._initialize_targets(n_targets_per_arm, mjmodel.geom_size[self.human_tuarm_geom][1], mjmodel.geom_size[self.human_tuarm_geom][0])
        self.wiping_targets_larm = self._initialize_targets(n_targets_per_arm, mjmodel.geom_size[self.human_tlarm_geom][1], mjmodel.geom_size[self.human_tlarm_geom][0])

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        
        contact_vector = jp.ones(self.n_targets)

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

        # does this need to happen every step
        global_targets_uarm = self._map_cylinder_points_to_global(self.wiping_targets_uarm, pipeline_state.xmat[self.human_tuarm_idx], pipeline_state.xpos[self.human_tuarm_idx])
        global_targets_larm = self._map_cylinder_points_to_global(self.wiping_targets_larm, pipeline_state.xmat[self.human_tlarm_idx], pipeline_state.xpos[self.human_tlarm_idx])
        global_targets = jp.vstack((global_targets_uarm, global_targets_larm))

        # get 3d distances
        panda_wiper = pipeline_state.site_xpos[self.panda_wiper_center_idx]
        all_distances = panda_wiper - global_targets
        closest_distance_idx = jp.argmin(all_distances)
        closest_target = global_targets[closest_distance_idx]

        # vector for which contacts have already been activated
        contact_vector = jp.zeros(self.n_targets)

        robo_obs = self._get_robo_obs(pipeline_state, closest_target)
        human_obs = self._get_human_obs(pipeline_state)
        
        # NOTE: IF THE LENGTH OF ANY OBSERVATIONS CHANGE, YOU MUST UPDATE THIS HERE:
        # jaxmarl/environments/mabrax/mappings.py#L111
        obs = jp.concatenate((
            # robot obs length = 6 + 6 + 3 + 4 + 6 + 3 + 1 = 29
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["wiper_target_dist"].reshape((3,)),
            robo_obs["wiper_target_dist_euclidean"].reshape((1,)),  
            # human = 17 + 17 + 3 + 3 = 40
            human_obs["human_joint_angles"],
            human_obs["human_joint_vel"],   
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],  
                    
        ))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "reward_wiping": zero,
            "weighted_reward_dist": zero,
            "weighted_reward_ctrl": zero,
            "weighted_reward_wiping": zero,
            "contact_vector": jp.zeros(self.n_targets),
            "distances": jp.zeros(self.n_targets),
            "contacts_info": jp.zeros(self.n_targets)
        }
        info = {"contact_vector": contact_vector}
        return State(pipeline_state, obs, reward, done, metrics, info)

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


        # does this need to happen every step
        global_targets_uarm = self._map_cylinder_points_to_global(self.wiping_targets_uarm, pipeline_state.xmat[self.human_tuarm_idx], pipeline_state.xpos[self.human_tuarm_idx])
        global_targets_larm = self._map_cylinder_points_to_global(self.wiping_targets_larm, pipeline_state.xmat[self.human_tlarm_idx], pipeline_state.xpos[self.human_tlarm_idx])
        global_targets = jp.vstack((global_targets_uarm, global_targets_larm))

        # get 3d distances
        panda_wiper = pipeline_state.site_xpos[self.panda_wiper_center_idx]
        all_distances = panda_wiper - global_targets
        all_distances_euclidean = jp.linalg.norm(all_distances)

        # vector for which contacts have already been activated
        old_contact_vector = state.info["contact_vector"]

        # mask distances and get closest target
        masked_distances_euclidean = jp.where(old_contact_vector == 0, jp.inf, all_distances_euclidean)
        closest_distance_idx = jp.argmin(masked_distances_euclidean)
        closest_target = global_targets[closest_distance_idx]

        robo_obs = self._get_robo_obs(pipeline_state, closest_target)
        human_obs = self._get_human_obs(pipeline_state)
        
        obs = jp.concatenate((
            robo_obs["robo_joint_angles"],
            robo_obs["robo_joint_vel"],
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["wiper_target_dist"].reshape((3,)),
            robo_obs["wiper_target_dist_euclidean"].reshape((1,)),  
            human_obs["human_joint_angles"],   
            human_obs["human_joint_vel"],   
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],     
        ))


        # 1) distance to closest target
        closest_distance = masked_distances_euclidean[closest_distance_idx]
        r_dist = (1 - jp.tanh(closest_distance / self._dist_scale))

        # 2) reward for hitting new target
        new_contact_vector = self._update_contact_vector(masked_distances_euclidean, old_contact_vector, self.target_threshold, robo_obs["force_on_tool"])
        n_contacts = jp.count_nonzero(new_contact_vector==0)
        n_old_contacts = jp.count_nonzero(old_contact_vector==0)
        new_contacts = (n_contacts - n_old_contacts).astype(jp.float32)

        # TODO: Add human preference rewards
        reward = self._dist_reward_weight*r_dist + self._ctrl_cost_weight*ctrl_cost + self._wiping_reward_weight*new_contacts
        
        done = jp.all(new_contact_vector == 0.0).astype(jp.float32)

        contact_info = {"contact_vector": new_contact_vector}

        
        # also in resset
        state.metrics.update(
            reward_dist = r_dist,
            reward_ctrl = ctrl_cost,
            reward_wiping = new_contacts,
            weighted_reward_dist = self._dist_reward_weight*r_dist,
            weighted_reward_ctrl = self._ctrl_cost_weight*ctrl_cost,
            weighted_reward_wiping = self._wiping_reward_weight*new_contacts,
            contact_vector = new_contact_vector,
            distances = masked_distances_euclidean,
            contacts_info = state.info["contact_vector"],
        )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=state.info | contact_info,
        )
        # return (robo_obs, human_obs)

    def _get_robo_obs(self, pipeline_state: base.State, closest_target) -> jax.Array:
        """Returns the environment observations."""
      
        # proprioception
        robo_joint_angles = pipeline_state.qpos[self.panda_joint_id_start:self.panda_joint_id_end]
        normalised_robo_joint_angles = (robo_joint_angles - self.robot_lower_joint_limits) / (self.robot_upper_joint_limits - self.robot_lower_joint_limits)
        normalised_robo_joint_angles = 2 * normalised_robo_joint_angles - 1
        robo_joint_vel = pipeline_state.qd[self.panda_joint_id_start:self.panda_joint_id_end]
        tool_position = pipeline_state.site_xpos[self.panda_wiper_center_idx]
        tool_orientation = pipeline_state.xquat[self.panda_wiper_body_idx]

        # tactile
        force_on_tool = self._get_force_on_tool(pipeline_state, self.UARM_TOOL_CONTACT_ID, self.LARM_TOOL_CONTACT_ID)

        # ground truth       
        wiper_target_dist = closest_target - tool_position
        wiper_target_dist_euclidean = jp.linalg.norm(wiper_target_dist)

        return {
            # proprioception
            "robo_joint_angles": normalised_robo_joint_angles,
            "robo_joint_vel": robo_joint_vel,
            "tool_position": tool_position,
            "tool_orientation": tool_orientation,
            # # tactile
            "force_on_tool": force_on_tool,
            # ground truth
            "wiper_target_dist": wiper_target_dist,
            "wiper_target_dist_euclidean": wiper_target_dist_euclidean  
        }
       
    
    def _get_human_obs(self, pipeline_state: base.State) -> jax.Array:
        """Returns the environment observations"""
        
        # proprioception
        human_joint_angles = pipeline_state.qpos[self.human_joint_id_start:self.human_joint_id_end]
        normalised_human_joint_angles = (human_joint_angles - self.human_lower_joint_limits) / (self.human_upper_joint_limits - self.human_lower_joint_limits)
        normalised_human_joint_angles = 2 * normalised_human_joint_angles - 1
        human_joint_vel = pipeline_state.qd[self.human_joint_id_start:self.human_joint_id_end]
        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]

        # tactile
        force_on_human = self._get_force_on_tool(pipeline_state, self.UARM_TOOL_CONTACT_ID, self.LARM_TOOL_CONTACT_ID)

        # gt
        # arm_pos = pipeline_state.site_xpos[self.hook_target_site]
        # arm_target = pipeline_state.site_xpos[self.arm_target_site]
        # larm_waist_dist = arm_target - arm_pos
        # larm_waist_dist_euclidean = jp.linalg.norm(larm_waist_dist)

        return {
            "human_joint_angles": normalised_human_joint_angles,
            "human_joint_vel": human_joint_vel,
            "human_uarm_pos": human_uarm_pos,
            "human_larm_pos": human_larm_pos,
            # "force_on_human": force_on_human,
        }  

    
    def _get_force_on_tool(self, pipeline_state, uarm_tool_id: int, larm_id:int) -> jax.Array:
        """Return the force on the tool"""
        tool_uarm = contact_force(self.sys, pipeline_state, uarm_tool_id, False)
        tool_larm = contact_force(self.sys, pipeline_state, larm_id, False)

        return jp.sum(jp.vstack((tool_uarm, tool_larm)), axis=0)
    
    def _initialize_targets(self, num_points: int, height: float, radius: float) -> jax.Array:
        # Calculate angles for equally spaced points around the circumference
        angles = jp.linspace(0, 2 * jp.pi, num_points, endpoint=False)
        
        # Calculate heights for equally spaced points along the height of the cylinder
        heights = jp.linspace(-height, height, num_points)
        
        # Calculate the (x, y, z) coordinates for each point
        x = radius * jp.cos(angles)
        y = radius * jp.sin(angles)
        z = heights
        
        # Stack the coordinates 
        points = jp.stack((x, y, z), axis=-1)
        
        return points
    
    def _map_cylinder_points_to_global(self, points: jax.Array, rotation_matrix: jax.Array, global_position: jax.Array) -> jax.Array:
        # Rotate the cylinder points using the rotation matrix
        # TODO: check rotation matrix shape and see if we need .T
        rotated_points = jp.dot(points, rotation_matrix.T)
        # Translate the rotated points to their global position
        global_points = rotated_points + global_position
        return global_points
    
    def _target_distances(self, target_locations: jax.Array, point: jax.Array) -> jax.Array:
        # Calculate the squared distances from the point to each target location
        distances = jp.sqrt(jp.sum((target_locations - point) ** 2, axis=1))
        return distances
    
        
    def _mask_contacts(self, distances: jax.Array, contact_vector: jax.Array) -> jax.Array:
        # Assign a very high value (e.g., jnp.inf) to distances corresponding to touched points
        modified_distances = jp.where(contact_vector == 0, jp.inf, distances)
        return modified_distances
    
    def _update_contact_vector(self, distances: jax.Array, contact_vector: jax.Array, threshold: float, contact_forces: jax.Array) -> jax.Array:
            
            below_threshold = distances < threshold
    
            non_zero_forces = jp.any(contact_forces != 0.0)

            updated_contact_vector = jp.where(
                jp.logical_and(below_threshold, non_zero_forces),
                0,
                contact_vector)
            return updated_contact_vector
        
    
