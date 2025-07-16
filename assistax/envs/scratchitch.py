from typing import Tuple

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


class ScratchItch(PipelineEnv):
    
    # TODO: Add docstring
    """
    Add docstring
    """

    # pyformat: enable

    def __init__(
        self,
        ctrl_cost_weight: float = 1e-6,
        dist_reward_weight: float = 1.0,
        dist_scale: float = 0.1,
        scratching_reward_weight: float = 4.0,
        target_scratcher_speed: float = 0.1,
        target_scratcher_force: float = 3.0,
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
        self.path = epath.resource_path("brax") / "envs/assets/wheelchair_scene.xml"

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

        self.panda_actuators_ids = []
        self.humanoid_actuators_ids = []

        ACTUATOR_IDX = 19
        for i in range(mjmodel.nu):
            actuator_name = mj_id2name(mjmodel, ACTUATOR_IDX, i)
            if actuator_name.startswith("actuator"):
                self.panda_actuators_ids.append(i)
            else:
                self.humanoid_actuators_ids.append(i)

        GEOM_IDX = 5 # 5 is the index of the geom tag in the xml file
        self.target_idx = mj_name2id(mjmodel, GEOM_IDX, "target")

        BODY_IDX = 1
        self.panda_effector_idx = mj_name2id(mjmodel, BODY_IDX, "hand")
        self.panda_scratcher_idx = mj_name2id(mjmodel, GEOM_IDX, "scratcher_stick")
        self.panda_scratcher_tip_idx = mj_name2id(mjmodel, mujoco.mjtObj.mjOBJ_SITE, "scratcher_point")
        self.panda_scratcher_body_idx = mj_name2id(mjmodel, BODY_IDX, "scratcher")

        self.human_tuarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_upper_arm") # Right human arm tuarm = target arm upper arm
        self.human_uarm_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "right_uarm1")
        self.human_uarm_target_idx = mj_name2id(mjmodel, GEOM_IDX, "target-u")
        self.human_tlarm_idx = mj_name2id(mjmodel, BODY_IDX, "right_lower_arm") # Right human arm tlarm = target arm lower arm
        self.human_larm_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "right_larm")
        self.human_larm_target_idx = mj_name2id(mjmodel, GEOM_IDX, "target-l")
        
        # self.contact_force = jax.vmap(contact_force, in_axes=(None, 0, None, None))

        self.UARM_TOOL_CONTACT_ID = 273
        self.LARM_TOOL_CONTACT_ID = 274

        self.panda_joint_id_start = 18
        self.panda_joint_id_end = 24

        self.human_joint_id_start = 1
        self.human_joint_id_end = 18

        
        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=self.sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._dist_reward_weight = dist_reward_weight
        self._dist_scale = dist_scale
        self._scratching_reward_weight = scratching_reward_weight
        self._target_scratcher_speed = target_scratcher_speed
        self._target_scratcher_force = target_scratcher_force
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

        # Generate scratch location position
        rng_scratch_arm, rng_scratch_loc = jax.random.split(rng_scratch, 2)
        # Choose arm part (upper/lower arm)
        scratch_arm = jax.random.bernoulli(rng_scratch_arm)
        scratch_arm_geom_idx = jax.lax.select(
            scratch_arm,
            self.human_larm_geom_idx,
            self.human_uarm_geom_idx
        )
        # Choose scratch position on arm
        rng_scratch_height, rng_scratch_angle = jax.random.split(rng_scratch_loc, 2)
        scratch_height = jax.random.uniform(rng_scratch_height, minval=-1.0, maxval=1.0)
        scratch_angle = jax.random.uniform(rng_scratch_angle, minval=0.0, maxval=2*jp.pi)
        arm_radius, arm_hlength, _ = self.sys.geom_size[scratch_arm_geom_idx]
        scratch_pos = jp.array([
            arm_radius*jp.cos(scratch_angle),
            arm_radius*jp.sin(scratch_angle),
            arm_hlength*scratch_height
        ]) # position of scratch *relative* to arm geom
        info = {
            "scratch": {
                "arm": scratch_arm,
                "arm_geom_idx": scratch_arm_geom_idx,
                "pos": scratch_pos,
            }
        }

        pipeline_state = self.pipeline_init(qpos, qvel)
        robo_obs = self._get_robo_obs(pipeline_state, info)
        human_obs = self._get_human_obs(pipeline_state, info)
        #obs = jp.concatenate((robo_obs, human_obs))
        obs = jp.concatenate((
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["distance_to_target"].reshape((1,)),
            robo_obs["target_position"],
            robo_obs["human_uarm_pos"],
            robo_obs["human_larm_pos"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["robo_joint_angles"],
            human_obs["tool_position"],
            human_obs["tool_orientation"],
            human_obs["distance_to_target"].reshape((1,)),
            human_obs["target_position"],
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],
            human_obs["force_on_human"].reshape((6,)),
            human_obs["human_joint_angles"],           
        ))
        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_dist": zero,
            "reward_ctrl": zero,
            "reward_scratching": zero
        }
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        ctrl_cost = -jp.sum(jp.square(action))
        robo_obs = self._get_robo_obs(pipeline_state, state.info)
        human_obs = self._get_human_obs(pipeline_state, state.info)
        obs = jp.concatenate((
            # robo_obs["position"],
            # robo_obs["velocity"],
            robo_obs["tool_position"],
            robo_obs["tool_orientation"],
            robo_obs["distance_to_target"].reshape((1,)),
            robo_obs["target_position"],
            robo_obs["human_uarm_pos"],
            robo_obs["human_larm_pos"],
            robo_obs["force_on_tool"].reshape((6,)),
            robo_obs["robo_joint_angles"],
            # human_obs["position"],
            # human_obs["velocity"],
            human_obs["tool_position"],
            human_obs["tool_orientation"],
            human_obs["distance_to_target"].reshape((1,)),
            human_obs["target_position"],
            human_obs["human_uarm_pos"],
            human_obs["human_larm_pos"],
            human_obs["force_on_human"].reshape((6,)),
            human_obs["human_joint_angles"],           
        ))
        
        dist = -robo_obs["distance_to_target"]
        r_dist = jp.exp(-dist**2/self._dist_scale)
        # This reward should mimick scratching but I'm not sure the scale is correct i.e. 0.005 might be too large or too small of a distance
        scratcher_vel = (
            pipeline_state.site_xpos[self.panda_scratcher_tip_idx] - pipeline_state0.site_xpos[self.panda_scratcher_tip_idx]
        ) / self.dt
        scratcher_speed = jp.linalg.norm(scratcher_vel)
        scratcher_force = jp.linalg.norm(human_obs["force_on_human"])
        # Chosen Boltzmann-like reward functions for scratcher speed and force, but we could swap with alternatives.
        r_scratching = (
                (r_dist < self._dist_scale)
                * scratcher_speed/self._target_scratcher_speed * jp.exp(-scratcher_speed/self._target_scratcher_speed)
                * scratcher_force/self._target_scratcher_force * jp.exp(-scratcher_force/self._target_scratcher_force)
        )
        reward = self._dist_reward_weight*r_dist + self._ctrl_cost_weight*ctrl_cost + self._scratching_reward_weight*r_scratching

        done = 0.0
        state.metrics.update(
            reward_dist = r_dist,
            reward_ctrl = ctrl_cost,
            reward_scratching = r_scratching
        )

        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done
        )

    # TODO: actually whether or not they make contact should not be used in the observation funciton but in the reward
    # observation should only include the distance from end effectors to the target which I can find with .site_xpos or .geom_xpos
    def _get_robo_obs(self, pipeline_state, state_info) -> jax.Array:
        """Returns the environment observations."""
        tool_position = pipeline_state.site_xpos[self.panda_scratcher_tip_idx]
        tool_orientation = pipeline_state.xquat[self.panda_scratcher_body_idx]
        force_on_tool = self._get_force_on_tool(pipeline_state, self.UARM_TOOL_CONTACT_ID, self.LARM_TOOL_CONTACT_ID)
        robo_joint_angles = pipeline_state.qpos[self.panda_joint_id_start:self.panda_joint_id_end]
        target_position = self._get_scratch_xpos(pipeline_state, state_info)
        distance_to_target = jp.linalg.norm(target_position - tool_position)
        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]

        return {
            "tool_position": tool_position,
            "tool_orientation": tool_orientation,
            "distance_to_target": distance_to_target,
            "target_position": target_position,
            "human_uarm_pos": human_uarm_pos,
            "human_larm_pos": human_larm_pos,
            "force_on_tool": force_on_tool,
            "robo_joint_angles": robo_joint_angles  
        }
    

    # TODO: Forces this is the only way human and robo obs are different
    def _get_human_obs(self, pipeline_state, state_info) -> jax.Array:
        """Returns the environment observations."""
        tool_position = pipeline_state.site_xpos[self.panda_scratcher_tip_idx]
        tool_orientation = pipeline_state.xquat[self.panda_scratcher_body_idx]        
        tool_orientation = pipeline_state.xquat[self.panda_effector_idx]
        target_position = self._get_scratch_xpos(pipeline_state, state_info)
        distance_to_target = jp.linalg.norm(target_position - tool_position)
        human_joint_angles = pipeline_state.qpos[self.human_joint_id_start:self.human_joint_id_end]
        human_uarm_pos = pipeline_state.xpos[self.human_tuarm_idx]
        human_larm_pos = pipeline_state.xpos[self.human_tlarm_idx]
        
        force_on_human = self._get_force_on_tool(pipeline_state, self.UARM_TOOL_CONTACT_ID, self.LARM_TOOL_CONTACT_ID)

        return {
            "tool_position": tool_position,
            "tool_orientation": tool_orientation,
            "distance_to_target": distance_to_target,
            "tool_orientation": tool_orientation,
            "target_position": target_position,
            "human_uarm_pos": human_uarm_pos,
            "human_larm_pos": human_larm_pos,
            "force_on_human": force_on_human,
            "human_joint_angles": human_joint_angles
        }
    
    def _get_scratch_xpos(self, pipeline_state, info):
        arm_geom_idx = info["scratch"]["arm_geom_idx"]
        scratch_pos = info["scratch"]["pos"]
        scratch_xpos = (
            pipeline_state.geom_xpos[arm_geom_idx]
            + pipeline_state.geom_xmat[arm_geom_idx]@scratch_pos
        )
        return scratch_xpos

    def _get_force_on_tool(self, pipeline_state, uarm_tool_id: int, larm_id:int) -> jax.Array:
        tool_uarm = contact_force(self.sys, pipeline_state, uarm_tool_id, False)
        tool_larm = contact_force(self.sys, pipeline_state, larm_id, False)
        return jp.sum(jp.vstack((tool_uarm, tool_larm)), axis=0)

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
