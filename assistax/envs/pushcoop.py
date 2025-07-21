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

def contact_id(pipeline_state: State, id1: int, id2: int) -> int:
    """Returns the contact id between two geom ids."""
    mask = (pipeline_state.contact.geom == jp.array([id1, id2])) | (pipeline_state.contact.geom == jp.array([id2, id1])) 
    mask2 = jp.all(mask[0], axis=1)
    id = jp.where(mask2)  # this was missing in the original code my bad
    return id

class PushCoop(PipelineEnv):
    """PushCoop environment."""

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-6,
        dist_reward_weight: float = 1.0,
        ee_dist_scale: float = 0.1,
        t_dist_scale: float = 0.3,
        t_dist_weight: float = 0.5,
        t_contact_weight: float = 0.5,
        backend="mjx",
        reset_noise_scale=5e-3,
        **kwargs
    ):
        """Initializes the PushCoop environment.

        Args:
            ctrl_cost: Cost for control.
            dist_reward_weight: Weight for distance to target reward.
            dist_scale: Scale for distance.
            t_dist_weight: Weight for t distance.
            t_contact_weight: Weight for t contact distance.
        """
        self.path = epath.resource_path("assistax") / "envs/assets/push_coop.xml"

        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        if backend == "mjx":
            self.sys = self.sys.tree_replace(
                {
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )                
        
        GEOM_IDX = mujoco.mjtObj.mjOBJ_GEOM
        BODY_IDX = mujoco.mjtObj.mjOBJ_BODY
        ACTUATOR_IDX = mujoco.mjtObj.mjOBJ_ACTUATOR
        SITE_IDX = mujoco.mjtObj.mjOBJ_SITE      

        self.panda1_actuator_ids = []
        self.panda2_actuator_ids = []   

        self.panda1_joint_id_start = 7
        self.panda2_joint_id_start = 14
        self.panda1_joint_id_end = 14   
        self.panda2_joint_id_end = 21

        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if "panda1" in actuator_name:
                self.panda1_actuator_ids.append(i)
            elif "panda2" in actuator_name:
                self.panda2_actuator_ids.append(i)
            else:
                raise ValueError(f"Unknown actuator name: {actuator_name}")
            
        self.panda1_pusher_body_idx = mj_name2id(mjmodel, BODY_IDX, "panda1_pusher")
        self.panda1_pusher_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "panda1_pusher_stick")
        self.panda1_pusher_point_idx = mj_name2id(mjmodel, SITE_IDX, "panda1_pusher_point")

        self.panda2_pusher_body_idx = mj_name2id(mjmodel, BODY_IDX, "panda2_pusher")
        self.panda2_pusher_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "panda2_pusher_stick")
        self.panda2_pusher_point_idx = mj_name2id(mjmodel, SITE_IDX, "panda2_pusher_point")

        self.t_shape_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "t_main")
        self.t_shape_geom_idx2 = mj_name2id(mjmodel, GEOM_IDX, "t_cross")
        self.t_shape_body_idx = mj_name2id(mjmodel, BODY_IDX, "t_object")

        self.obstacle1_idx = mj_name2id(mjmodel, GEOM_IDX, "obs1")
        self.obstacle2_idx = mj_name2id(mjmodel, GEOM_IDX, "obs2")
        self.obstacle3_idx = mj_name2id(mjmodel, GEOM_IDX, "obs3")
        self.obstacle4_idx = mj_name2id(mjmodel, GEOM_IDX, "obs4")
        self.obstacle5_idx = mj_name2id(mjmodel, GEOM_IDX, "obs5")
        self.obstacle6_idx = mj_name2id(mjmodel, GEOM_IDX, "obs6")

        self.table_top_idx = mj_name2id(mjmodel, GEOM_IDX, "table_top")

        # self.panda1_sensor_idx = mj_name2id(mjmodel, GEOM_IDX, "panda1_touch")
        # self.panda2_sensor_idx = mj_name2id(mjmodel, GEOM_IDX, "panda2_touch")

        # contact ids when all obstacles are activated
        self.contact_id_tmain = [280, 281, 282, 283]
        self.contact_id_tcross = [284, 285, 286, 287]

        # panda contact ids when all obstacles are activated
        # self.panda1_contact_id_t = [794, 795, 792, 793]
        # self.panda2_contact_id_t = [796, 797, 798, 799]

        # panda contac ids contact with less obstacles
        self.panda1_contact_id_t = [672, 673, 674, 675]
        self.panda2_contact_id_t = [676, 677, 678, 679]

        # less obstacles T - obstacle contact ids

        self.contact_t_obs1 = [328, 329, 330, 331, 344, 345, 346, 347]
        self.contact_t_obs2 = [332, 333, 334, 335, 348, 349, 350, 351]
        self.contact_t_obs4 = [336, 337, 338, 339, 352, 353, 354, 355]
        self.contact_t_obs5 = [340, 341, 342, 343, 356, 357, 358, 359]

        # self.t_main_obs1 = [328, 329, 330, 331]
        # self.t_main_obs2 = [332, 333, 334, 335]
        # self.t_main_obs4 = [336, 337, 338, 339]
        # self.t_main_obs5 = [340, 341, 342, 343]

        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)
        self._ctrl_cost = ctrl_cost_weight
        self._dist_reward_weight = dist_reward_weight
        self._ee_dist_scale = ee_dist_scale
        self._t_dist_scale = t_dist_scale
        self._t_dist_weight = t_dist_weight
        self._t_contact_weight = t_contact_weight
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment.

        Args:
            rng: Random number generator.

        Returns:
            State: The initial state of the environment.
        """
        rng_pos, rng_vel, target_loc_rng = jax.random.split(rng, 3)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        init_q = self.sys.mj_model.keyframe("init").qpos
        qpos = init_q + jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)

        # Set the target position

        target_pos = self._initialize_target_pos(pipeline_state, target_loc_rng, self.table_top_idx)
        
        robo1_obs = self._get_robo1_obs(pipeline_state, target_pos)
        robo2_obs = self._get_robo2_obs(pipeline_state, target_pos)
        
        obs = jp.concatenate((
            robo1_obs["target_pos"],
            robo1_obs["pusher_pos"],
            robo1_obs["pusher_rot"],
            robo1_obs["pusher_forces"],
            robo1_obs["t_location"],
            robo1_obs["robo1_joint_angles"],
            robo1_obs["robot1_ee_dist"].reshape((1,)),
            # robo1_obs["obs1_forces"],
            # robo1_obs["obs2_forces"],
            # robo1_obs["obs4_forces"],
            # robo1_obs["obs5_forces"],
            robo1_obs["other_agent_ee_pos"],
            robo2_obs["target_pos"],
            robo2_obs["pusher_pos"],
            robo2_obs["pusher_rot"],
            robo2_obs["pusher_forces"], # .reshape((1,))
            robo2_obs["t_location"],
            robo2_obs["robo2_joint_angles"],
            robo2_obs["robot2_ee_dist"].reshape((1,)),
            # robo2_obs["obs1_forces"],
            # robo2_obs["obs2_forces"],
            # robo2_obs["obs4_forces"],
            # robo2_obs["obs5_forces"],
            robo2_obs["other_agent_ee_pos"],
        ))
        reward = jp.zeros(2) # for heterogenous rewards
        done, zero = jp.zeros(2)

        metrics = {
            "robo1_reward_dist": zero,
            "robo2_reward_dist": zero,
            "robo1_reward_ctrl": zero,
            "robo2_reward_ctrl": zero,
            "robo1_reward_t_contact": zero,
            "robo2_reward_t_contact": zero,
            "reward_t_dist": zero,
        }

        # info = {
        #     "robo1_dist_to_target": zero,
        #     "robo2_dist_to_target": zero,
        #     "robo1_t_contact": zero,
        #     "robo2_t_contact": zero,
        #     "t_dist": zero,
        #     "t_contact_id": zero,
        #     "t_contact_force": zero,
        #     "robo1_reward_dist": zero,
        #     "robo2_reward_dist": zero,
        #     "robo1_reward_ctrl": zero,
        #     "robo2_reward_ctrl": zero,
        #     "robo1_reward_t_contact": zero,
        #     "robo2_reward_t_contact": zero,
        #     "reward_t_dist": zero,
        #     "target_pos": target_pos,
        # }

        info = {
            "dist_to_target": zero,
            "t_contact": zero,
            "t_dist": zero,
            "t_contact_id": zero,
            "t_contact_force": zero,
            "target_pos": target_pos,
        }

        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        ctrl_cost = -jp.sum(jp.square(action))
        robo1_obs = self._get_robo1_obs(pipeline_state, state.info["target_pos"])
        robo2_obs = self._get_robo2_obs(pipeline_state, state.info["target_pos"])
        obs = jp.concatenate((
            robo1_obs["target_pos"],
            robo1_obs["pusher_pos"],
            robo1_obs["pusher_rot"],
            robo1_obs["pusher_forces"],
            robo1_obs["t_location"],
            robo1_obs["robo1_joint_angles"],
            robo1_obs["robot1_ee_dist"].reshape((1,)),
            # robo1_obs["obs1_forces"],
            # robo1_obs["obs2_forces"],
            # robo1_obs["obs4_forces"],
            # robo1_obs["obs5_forces"],
            robo1_obs["other_agent_ee_pos"],
            robo2_obs["target_pos"],
            robo2_obs["pusher_pos"],
            robo2_obs["pusher_rot"],
            robo2_obs["pusher_forces"], # .reshape((1,))
            robo2_obs["t_location"],
            robo2_obs["robo2_joint_angles"],
            robo2_obs["robot2_ee_dist"].reshape((1,)),
            # robo2_obs["obs1_forces"],
            # robo2_obs["obs2_forces"],
            # robo2_obs["obs4_forces"],
            # robo2_obs["obs5_forces"],
            robo2_obs["other_agent_ee_pos"],
        ))
        
        dist_target = - self._get_dist_target(pipeline_state, state.info)
        target_dist_reward = jp.exp(-dist_target**2 / self._t_dist_scale) 
        # dist1, dist2 = self._ee_dist_to_t(pipeline_state)
        dist1 = - robo1_obs["robot1_ee_dist"]
        dist2 = - robo2_obs["robot2_ee_dist"]
        dist1_reward = jp.exp(-dist1**2 / self._ee_dist_scale)
        dist2_reward = jp.exp(-dist2**2 / self._ee_dist_scale)

        # robo1_contact = jp.sum(robo1_obs["pusher_forces"]) > 0
        # robo2_contact = jp.sum(robo2_obs["pusher_forces"]) > 0

        # print(f"dist1: {dist1}, dist2: {dist2}, dist_target: {dist_target} \n target_dist_reward: {target_dist_reward}, dist1_reward: {dist1_reward}, dist2_reward: {dist2_reward}")

        # jax.debug.breakpoint()
        
        # print(f"t_at_target_reward: {t_at_target_reward}")

        # TODO: contact between the two robots should be penalized

        # done = self._get_t_floor_contact(pipeline_state) or (self._get_dist_target(pipeline_state, state.info) < 0.1) # add termination condition
        
        
        t_at_target_reward = (self._get_dist_target(pipeline_state, state.info) < 0.1) * 10
        failed_reward = self._get_t_floor_contact(pipeline_state) * -10

        done = ((t_at_target_reward + failed_reward) != 0)*1.0

        reward_robo1 = self._dist_reward_weight * target_dist_reward + self._t_dist_weight * dist1_reward + self._ctrl_cost * ctrl_cost + failed_reward + t_at_target_reward # took out this component 1.0 * robo1_contact
        reward_robo2 = self._dist_reward_weight * target_dist_reward + self._t_dist_weight * dist2_reward + self._ctrl_cost * ctrl_cost + failed_reward + t_at_target_reward
        reward = jp.array([reward_robo1, reward_robo2])
        
        # metrics = {
        #     "robo1_reward_dist": dist1_reward,
        #     "robo2_reward_dist": dist2_reward,
        #     "robo1_reward_ctrl": ctrl_cost,
        #     "robo2_reward_ctrl": ctrl_cost,
        #     "robo1_reward_t_contact": robo1_contact,
        #     "robo2_reward_t_contact": robo2_contact,
        #     "reward_t_dist": target_dist_reward,
        # }
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_robo1_obs(self, pipeline_state: base.State, target_pos) -> jax.Array:
        """Get the observation for robot 1."""
        pusher_pos = pipeline_state.site_xpos[self.panda1_pusher_point_idx]
        pusher_rot = pipeline_state.xquat[self.panda1_pusher_body_idx]

        pusher_forces = self._get_pusher_forces(self.panda1_contact_id_t, pipeline_state)

        t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]

        robo1_joint_angles = pipeline_state.qpos[self.panda1_joint_id_start:self.panda1_joint_id_end]

        ee_dist = jp.linalg.norm(pusher_pos - t_location)

        # obs1_forces, obs2_forces, obs4_forces, obs5_forces = self._get_t_obstacle_contact(pipeline_state)

        other_agent_ee_pos = pipeline_state.geom_xpos[self.panda2_pusher_geom_idx]

        return {
            "target_pos": target_pos,
            "pusher_pos": pusher_pos,
            "pusher_rot": pusher_rot,
            "pusher_forces": pusher_forces,
            "t_location": t_location,
            "robo1_joint_angles": robo1_joint_angles,
            "robot1_ee_dist": ee_dist,
            # "obs1_forces": obs1_forces,
            # "obs2_forces": obs2_forces,
            # "obs4_forces": obs4_forces,
            # "obs5_forces": obs5_forces,
            "other_agent_ee_pos": other_agent_ee_pos,
        }
    
    def _get_robo2_obs(self, pipeline_state: base.State, target_pos) -> jax.Array:
        """Get the observation for robot 2."""
        pusher_pos = pipeline_state.site_xpos[self.panda2_pusher_point_idx]
        pusher_rot = pipeline_state.xquat[self.panda2_pusher_body_idx]
        
        # pusher_forces = pipeline_state.sensordata[self.panda1_sensor_idx]

        pusher_forces = self._get_pusher_forces(self.panda2_contact_id_t, pipeline_state)

        t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]

        robo2_joint_angles = pipeline_state.qpos[self.panda2_joint_id_start:self.panda2_joint_id_end]

        ee_dist = jp.linalg.norm(pusher_pos - t_location)

        # obs1_forces, obs2_forces, obs4_forces, obs5_forces = self._get_t_obstacle_contact(pipeline_state)
        
        other_agent_ee_pos = pipeline_state.geom_xpos[self.panda1_pusher_geom_idx]

        return {
            "target_pos": target_pos,
            "pusher_pos": pusher_pos,
            "pusher_rot": pusher_rot,
            "pusher_forces": pusher_forces,
            "t_location": t_location,
            "robo2_joint_angles": robo2_joint_angles,
            "robot2_ee_dist": ee_dist,
            # "obs1_forces": obs1_forces,
            # "obs2_forces": obs2_forces,
            # "obs4_forces": obs4_forces,
            # "obs5_forces": obs5_forces,
            "other_agent_ee_pos": other_agent_ee_pos,
        }
    
    # # TODO do set random target position as self
    # def _initialize_target_pos(self, table_top_idx: int) -> jax.Array:
    #     """Initialize the target position."""
    #     table_top_pos = self.sys.mj_model.geom_pos[table_top_idx]
    #     table_top_size = self.sys.mj_model.geom_size[table_top_idx]
    #     table_top_height = table_top_pos[2] + table_top_size[2]

    #     # Set the target position to be above the table
    #     target_pos = jp.array([0.0, 0.0, table_top_height + 0.1])
    #     return target_pos
    
    def _get_dist_target(self, pipeline_state: base.State, info) -> jax.Array:
        """Get the distance to the target."""
        t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]
        target_pos = info["target_pos"]
        dist = jp.linalg.norm(target_pos - t_location)
        return dist
    
    # def _ee_dist_to_t(self, pipeline_state: base.State) -> jax.Array:
    #     """Get the distance from the end effector to the target."""
    #     t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]
    #     panda1_pusher_pos = pipeline_state.site_xpos[self.panda1_pusher_point_idx]
    #     panda2_pusher_pos = pipeline_state.site_xpos[self.panda2_pusher_point_idx]
    #     dist1 = jp.linalg.norm(t_location - panda1_pusher_pos)
    #     dist2 = jp.linalg.norm(t_location - panda2_pusher_pos)
    #     return dist1, dist2
    
    
    def _get_t_floor_contact(self, pipeline_state: base.State) -> jax.Array:
        """Get the contact between the T shape and the floor."""
        
        contact_forces = []
        for i in self.contact_id_tmain:
            force = contact_force(self.sys, pipeline_state, i, False)
            contact_forces.append(force)
        
        for i in self.contact_id_tcross:
            force = contact_force(self.sys, pipeline_state, i, False)
            contact_forces.append(force)
        
        contact_forces = jp.array(contact_forces)

        return (jp.sum(contact_forces) != 0).astype(float) 

    def _initialize_target_pos(self, pipeline_state, rng: jax.Array, table_top_idx: int) -> jax.Array:
        """Initialize a random target position on the far side of the table.
        
        Args:
            table_top_idx: Index of the table top geom
            
        Returns:
            jax.Array: A 3D position for the target
        """
        # Get table properties
        table_top_pos = pipeline_state.geom_xpos[table_top_idx]
        table_top_size = self.sys.geom_size[table_top_idx]
        table_rotation = pipeline_state.geom_xmat[table_top_idx]

        # table_height = table_top_pos[2] + table_top_size[2]  # Z coordinate of table surface
        
        # Calculate target area bounds (far side of the table, from obstacles)
        # The target area is on the negative x-side of the table (far from T-object's starting position)
        min_x = - 0.95 * table_top_size[0]  # Left 40% of the table
        max_x = - 0.5 * table_top_size[0]  # Up to 20% from the left edge
        
        # Y bounds - keep away from edges
        min_y = - 0.9 * table_top_size[1]  # Bottom 70% of table
        max_y = 0.9 * table_top_size[1]  # Top 70% of table
        
        # Function to generate random position
        
        x_key, y_key = jax.random.split(rng, 2)
        x = jax.random.uniform(x_key, (), minval=min_x, maxval=max_x)
        y = jax.random.uniform(y_key, (), minval=min_y, maxval=max_y)
        # Z is fixed at table height plus a small offset
        z = 0.01  # Slightly above table surface
        local_pos = jp.array([x, y, z])
        global_pos = table_top_pos + table_rotation @ local_pos
        
        return global_pos  
    
    def _get_pusher_forces(self, contact_ids, pipeline_state: base.State) -> jax.Array:
        """Get the combined forces on the pusher from all contacts with the T-shape.
        
        Args:
            contact_ids: List of contact IDs between pusher and T-shape
            pipeline_state: Current physics state
            
        Returns:
            jax.Array: Combined force vector [normal, shear_x, shear_y]
        """
        # Initialize with zeros in case there are no contacts
        if not contact_ids:
            return jp.zeros(3)
        
        # Collect forces from all contacts
        all_forces = []
        for i in contact_ids:
            # Get 6D force vector for this contact (3D force, 3D torque)
            # Shape: [6] - [force_x, force_y, force_z, torque_x, torque_y, torque_z]
            force = contact_force(self.sys, pipeline_state, i, False)
            all_forces.append(force)
        
        # Stack all force vectors
        # Shape: [num_contacts, 6]
        all_forces = jp.array(all_forces)
        
        # Sum along axis 0 (across all contacts) to get combined 6D force vector
        # Shape: [6]
        combined_forces = jp.sum(all_forces, axis=0)
        
        # Extract only the force components (first 3 values)
        # Shape: [3] - [force_x, force_y, force_z]
        force_components = combined_forces[:3]
        
        return force_components
    
    def _get_t_obstacle_contact(self, pipeline_state: base.State) -> jax.Array:
        """Get the contact between the T shape and the obstacles."""
        obs1_forces = []
        obs2_forces = []
        obs4_forces = []
        obs5_forces = []

        for i in self.contact_t_obs1:
            force = contact_force(self.sys, pipeline_state, i, False)
            obs1_forces.append(force)
        for i in self.contact_t_obs2:
            force = contact_force(self.sys, pipeline_state, i, False)
            obs2_forces.append(force)
        for i in self.contact_t_obs4:
            force = contact_force(self.sys, pipeline_state, i, False)
            obs4_forces.append(force)
        for i in self.contact_t_obs5:
            force = contact_force(self.sys, pipeline_state, i, False)
            obs5_forces.append(force)
        
        obs1_forces = jp.sum(jp.array(obs1_forces), axis=0)
        obs2_forces = jp.sum(jp.array(obs2_forces), axis=0)
        obs4_forces = jp.sum(jp.array(obs4_forces), axis=0)
        obs5_forces = jp.sum(jp.array(obs5_forces), axis=0)

        obs1_forces = obs1_forces[:3]
        obs2_forces = obs2_forces[:3]
        obs4_forces = obs4_forces[:3]
        obs5_forces = obs5_forces[:3]

        return obs1_forces, obs2_forces, obs4_forces, obs5_forces


    
    # def _t_to_closest_obs(self, pipeline_state: base.State) -> jax.Array:
    #     """Get the distance from the T shape to the closest obstacle."""
    #     t_location = pipeline_state.geom_xpos[self.t_shape_geom_idx]
    #     obs1_location = pipeline_state.geom_xpos[self.obstacle1_idx]
    #     obs2_location = pipeline_state.geom_xpos[self.obstacle2_idx]
    #     obs3_location = pipeline_state.geom_xpos[self.obstacle3_idx]
    #     obs4_location = pipeline_state.geom_xpos[self.obstacle4_idx]
    #     obs5_location = pipeline_state.geom_xpos[self.obstacle5_idx]
    #     obs6_location = pipeline_state.geom_xpos[self.obstacle6_idx]

    #     # Calculate distances to each obstacle
    #     dist_obs1 = jp.linalg.norm(t_location - obs1_location)
    #     dist_obs2 = jp.linalg.norm(t_location - obs2_location)
    #     dist_obs3 = jp.linalg.norm(t_location - obs3_location)
    #     dist_obs4 = jp.linalg.norm(t_location - obs4_location)
    #     dist_obs5 = jp.linalg.norm(t_location - obs5_location)
    #     dist_obs6 = jp.linalg.norm(t_location - obs6_location)

    #     # Find the minimum distance
    #     min_dist = jp.min(jp.array([dist_obs1, dist_obs2, dist_obs3, dist_obs4, dist_obs5, dist_obs6]))

    #     return min_dist
    
    # TODO: implement termination condition 
    
    # Could add this in for heterogenous rewards
    # def _get_t_contact(self, contact_id, pipeline_state: base.State) -> jax.Array:
        
