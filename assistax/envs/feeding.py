from typing import Tuple, Dict

from brax import base
from brax import math as bmath
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco 
from mujoco import mj_id2name, mj_name2id
from enum import IntEnum
from mujoco.mjx._src.support import contact_force
import numpy as np

# import xml.etree.ElementTree as ET

class GeomType(IntEnum):
    PLANE = 0
    HFIELD = 1
    SPHERE = 2
    CAPSULE = 3
    ELLIPSOID = 4
    CYLINDER = 5
    BOX = 6
    MESH = 7


class Feeding(PipelineEnv):
    
    # TODO: Add docstring
    """
    Add docstring
    """

    # pyformat: enable

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-6,
        dist_reward_weight: float = 2.0,
        dist_scale: float = 2.5,
        feeding_reward_weight: float = 1.0,
        target_spoon_speed: float = 0.1,
        target_spoon_force: float = 1.0,
        reset_noise_scale=5e-3,
        backend="mjx",
        **kwargs
    ):
        """Creates a Hopper environment.

        Args:
          ctrl_cost_weight: Weight for the control cost.
          reset_noise_scale: Scale of noise to add to reset states.
          backend: str, the physics backend to use
          **kwargs: Arguments that are passed to the base class.
        """
        self.path = epath.resource_path("assistax") / "envs/assets/feeding_scene.xml"

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

        self.n_agents = 2

        self.panda_actuators_ids = []
        self.humanoid_actuators_ids = []

        ACTUATOR_IDX = 19
        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if actuator_name.startswith("actuator"):
                self.panda_actuators_ids.append(i)
            else:
                self.humanoid_actuators_ids.append(i)

        GEOM_IDX = mujoco.mjtObj.mjOBJ_GEOM
        BODY_IDX = mujoco.mjtObj.mjOBJ_BODY
        ACTUATOR_IDX = mujoco.mjtObj.mjOBJ_ACTUATOR
        SITE_IDX = mujoco.mjtObj.mjOBJ_SITE

        self.panda_effector_idx = mj_name2id(mjmodel, BODY_IDX, "hand")

        self.panda_spoon_body_idx = mj_name2id(mjmodel, BODY_IDX, "spoon")
        self.panda_spoon_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "spoon_bowl")
        self.panda_spoon_rside_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "spoon_right_side")

        self.panda_spoon_centre = mj_name2id(mjmodel, SITE_IDX, "spoon_center")

        self.human_mouth = mj_name2id(mjmodel, SITE_IDX, "mouth")
        self.human_head_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "head") 

        self.SPOON_HEAD_CONTACT_ID = 186
        self.SPOON_RSIDE_CONTACT_ID = 187 

        self.panda_joint_id_start = 20
        self.panda_joint_id_end = 27

        self.human_joint_id_start = 1
        self.human_joint_id_end = 20

        
        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._dist_reward_weight = dist_reward_weight
        self._dist_scale = dist_scale
        self._feeding_reward_weight = feeding_reward_weight
        self._target_spoon_speed = target_spoon_speed
        self._target_spoon_force = target_spoon_force
        self._reset_noise_scale = reset_noise_scale

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng_pose, rng_scratch = jax.random.split(rng, 2)

        # Add small positional and velocity noise to initialisation
        rng_pos, rng_vel = jax.random.split(rng_pose, 2)
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        init_q = self.sys.mj_model.keyframe("init").qpos
        qpos = init_q + jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi)

        pipeline_state = self.pipeline_init(qpos, qvel)
        robo_obs = self._get_robo_obs(pipeline_state)
        human_obs = self._get_human_obs(pipeline_state)

        #print("Robo Obs Shapes:")
        #print([f"{i}: {robo_obs[i].shape}" for i in robo_obs.keys()])
        #print("Human Obs Shapes:")
        #print([f"{i}: {human_obs[i].shape}" for i in human_obs.keys()])
        #print("Robo obs indices:")
        #print(f"0 to {sum([robo_obs[i].shape[0] for i in robo_obs.keys()])}")
        #print("Human obs indices:")
        #print(f"From {sum([robo_obs[i].shape[0] for i in robo_obs.keys()])} to {sum([robo_obs[i].shape[0] for i in robo_obs.keys()]) + sum([human_obs[i].shape[0] for i in human_obs.keys()])}")

        
        obs = jp.concatenate((
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["target_position"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["robo_joint_angles"],
            human_obs["tool_position"],
            human_obs["tool_orientation"],
            human_obs["target_position"],
            human_obs["force_on_human"].reshape((6,)),
            human_obs["human_joint_angles"],           
        ))

        
        
        #print("Total obs shape:", obs.shape)

        info = {
            "ee_speed": 0.0, # add for preference tracking
            "ee_force": 0.0,
            "action_magnitude": 0.0,
        }
        
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "reward_orientation": zero,
            "reward_velocity": zero,
            "reward_force": zero,
        }
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        ctrl_cost = -jp.sum(jp.square(action))
        robo_obs = self._get_robo_obs(pipeline_state)
        human_obs = self._get_human_obs(pipeline_state)
        
        obs = jp.concatenate((
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["target_position"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["robo_joint_angles"],
            human_obs["tool_position"],
            human_obs["tool_orientation"],
            human_obs["target_position"],
            human_obs["force_on_human"].reshape((6,)),
            human_obs["human_joint_angles"],           
        ))

        # 1. Distance reward
        spoon_pos = robo_obs["tool_position"]
        mouth_pos = robo_obs["target_position"]
        distance = jp.linalg.norm(spoon_pos - mouth_pos)
        
        
        r_dist = jp.exp(-self._dist_scale * distance) # Spikier than the bolzmann esque reward from scratching

        #breakpoint()
        #jax.debug.print("Distance: {dist}, Distance_Reward: {r_dist}", dist=distance, r_dist=r_dist)
        # 2. Spoon Feeding Reward

        # 2.1 Roll orientation Reward
        # This shoud encourage the robot to tilt the spoon appropriately when near the mouth.
        # Define spoon local vector pointing up perpendicular up from the spoon bowl
        local_vec_spoon_up = jp.array([-1.0, 0.0, 0.0])

        # Vector pointing from spoon to mouth
        vec_spoon_to_mouth = mouth_pos - spoon_pos
        target_spoon_to_mouth = vec_spoon_to_mouth / (jp.linalg.norm(vec_spoon_to_mouth) + 1e-6) # Normalize 

        # Carrying i.e. pointing up target vector
        target_upright_gobal = jp.array([0.0, 0.0, 1.0])
        
        spoon_quat = robo_obs["tool_orientation"]
        spoon_up_world = bmath.rotate(local_vec_spoon_up, spoon_quat)     


        # TODO: See if we should add this in to further improve the feeding part. 
        # 2.2 Yaw orientation Reward
        # I.e. approaching the mouth with the front of the spoon
        
        vec_spoon_north = jp.array([0.0, 0.0, 1.0])
        vec_spoon_tip_to_world = bmath.rotate(vec_spoon_north, spoon_quat)
        
        r_aim = jp.dot(vec_spoon_tip_to_world, target_spoon_to_mouth)

        # Interpolate (Blend) the vectors for final rewqrd
        tilt_scale = 150.0 # Higher value means more sensitivity to beeing close to the mouth
        proximity_signal = jp.exp(-tilt_scale * distance**2)
        
        # 2.3 Blend the vectors
        # Far (0.0) -> Track Sky
        # Close (1.0) -> Track Mouth
        current_target_vec = (
            (1.0 - proximity_signal) * target_upright_gobal + 
            proximity_signal * target_spoon_to_mouth
        )
        
        # Normalize
        current_target_vec = current_target_vec / (jp.linalg.norm(current_target_vec) + 1e-6)
                
        # 2.4 Compare with World Up
        r_pour = jp.dot(current_target_vec, spoon_up_world)
        
        # Spoon feeding final orientation reward 
        r_orientation = 1.0 * r_pour + (0.3 + 0.7 * proximity_signal) * r_aim

        # 3. Velocity Reward
        spoon_vel = (
            pipeline_state.site_xpos[self.panda_spoon_centre] - pipeline_state0.site_xpos[self.panda_spoon_centre]
        ) / self.dt
        current_speed = jp.linalg.norm(spoon_vel)
        vel_scale = 150.0 #Higher later breaking lower earlier breaking
        braking_signal = jp.exp(-vel_scale * distance**2)

        # Far away: We WANT motion (0.2 m/s) to encourage approach.
        # At mouth: We WANT stopping (0.0 m/s) to prevent collision/vibration.
        target_speed_far = 0.20
        target_speed_close = 0.00
        
        # Smooth interpolation based on distance
        current_target_speed = (
            (1.0 - braking_signal) * target_speed_far + 
            braking_signal * target_speed_close
        )

        # vel_sigma controls strictness. 0.1 means +/- 10cm/s error is tolerated.
        vel_sigma = 0.1 
        
        # We square the difference to make a smooth Bell Curve peak at the target speed
        speed_error = current_speed - current_target_speed
        r_velocity = jp.exp(-jp.square(speed_error) / (vel_sigma**2))
        
        # 4. Contact with mouth reward
        right_side_spoon_force = self._get_force_on_tool(pipeline_state, self.SPOON_RSIDE_CONTACT_ID)

        contact_force_mouth = jp.linalg.norm(right_side_spoon_force)

        r_force = contact_force_mouth/self._target_spoon_force * jp.exp(-contact_force_mouth/self._target_spoon_force)

        # 5, Total Reward
        reward = (
            self._dist_reward_weight * r_dist +
            self._feeding_reward_weight * (r_orientation + r_velocity + r_force) +
            self._ctrl_cost_weight * ctrl_cost
        )

        done = 0.0
        state.metrics.update(
            reward_dist = r_dist,
            reward_ctrl = ctrl_cost,
            reward_orientation = r_orientation,
            reward_velocity = r_velocity,
            reward_force = r_force,
        )

        state.info.update(
            ee_speed=current_speed,
            ee_force=contact_force_mouth,
            action_magnitude=jp.linalg.norm(action),
        )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done
        )

    def _get_robo_obs(self, pipeline_state) -> Dict[str, jax.Array]:
        """Returns the environment observations."""
        tool_position = pipeline_state.site_xpos[self.panda_spoon_centre]
        tool_orientation = pipeline_state.xquat[self.panda_spoon_body_idx]
        force_on_spoon = self._get_force_on_tool(pipeline_state, self.SPOON_HEAD_CONTACT_ID)
        force_on_rside = self._get_force_on_tool(pipeline_state, self.SPOON_RSIDE_CONTACT_ID)
        total_force_on_spoon = jp.sum(jp.vstack([force_on_spoon, force_on_rside]), axis=0)
        robo_joint_angles = pipeline_state.qpos[self.panda_joint_id_start:self.panda_joint_id_end]
        target_position = pipeline_state.site_xpos[self.human_mouth]
        #breakpoint()
        #jax.debug.print("Tool Pos: {tp}, Target Pos: {tarp}", tp=tool_position, tarp=target_position)
        #jax.debug.print("Distance: {dist}", dist=jp.linalg.norm(tool_position - target_position))
        # distance_to_target = jp.linalg.norm(target_position - tool_position) # Maybe just for the reward

        return {
            "tool_position": tool_position,
            "tool_orientation": tool_orientation,
            "target_position": target_position,
            "force_on_tool": total_force_on_spoon,
            "robo_joint_angles": robo_joint_angles
        }
    

    # Forces is the only way human and robo obs are different
    def _get_human_obs(self, pipeline_state) -> Dict[str, jax.Array]:
        """Returns the environment observations."""
        tool_position = pipeline_state.site_xpos[self.panda_spoon_centre]
        tool_orientation = pipeline_state.xquat[self.panda_spoon_body_idx]
        force_on_spoon = self._get_force_on_tool(pipeline_state, self.SPOON_HEAD_CONTACT_ID)
        force_on_rside = self._get_force_on_tool(pipeline_state, self.SPOON_RSIDE_CONTACT_ID)
        total_force_on_spoon = jp.sum(jp.vstack([force_on_spoon, force_on_rside]), axis=0)
        target_position = pipeline_state.site_xpos[self.human_mouth] 
        #distance_to_target = jp.linalg.norm(target_position - tool_position)
        human_joint_angles = pipeline_state.qpos[self.human_joint_id_start:self.human_joint_id_end]
        
        return {
            "tool_position": tool_position,
            "tool_orientation": tool_orientation,
            "target_position": target_position,
            "force_on_human": total_force_on_spoon,
            "human_joint_angles": human_joint_angles
        }
    
    def _get_force_on_tool(self, pipeline_state, contact_id: int) -> jax.Array:
        force = contact_force(self.sys, pipeline_state, contact_id, False)
        return force 

    def get_sys_for_render(self, state):
        if state.info["scratch"]["arm"]:
            body_idx = self.human_tlarm_idx
            geom_idx = self.human_larm_geom_idx
            target_idx = self.human_larm_target_idx
        else:
            body_idx = self.human_tuarm_idx
            geom_idx = self.human_uarm_geom_idx
            target_idx = self.human_uarm_target_idx
        new_pos = self.sys.geom_pos[geom_idx] + bmath.rotate(
            state.info["scratch"]["pos"],
            bmath.relative_quat(
                self.sys.body_quat[body_idx],
                self.sys.geom_quat[geom_idx]
            )
        )
        return self.sys.replace(geom_pos=self.sys.geom_pos.at[target_idx].set(new_pos))
    