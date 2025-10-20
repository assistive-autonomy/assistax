from typing import Tuple, Dict, Optional
from enum import IntEnum
import jax
from jax import numpy as jp
import numpy as np
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import mujoco
from mujoco import mj_name2id
from mujoco.mjx._src.support import contact_force


class HandoverPhase(IntEnum):
    """Phases of the handover task"""
    APPROACH = 0      # Panda1 approaching object
    GRASP = 1        # Panda1 grasping object
    LIFT = 2         # Panda1 lifting object
    TRANSFER = 3     # Moving to handover location
    HANDOVER = 4     # Actual handover happening
    RECEIVE = 5      # Panda2 receiving object
    RETREAT = 6      # Panda1 retreating after handover
    PLACE = 7        # Panda2 placing object
    COMPLETE = 8     # Task complete


class CooperativeHandover(PipelineEnv):
    """
    Cooperative handover task between two Franka Panda robots.
    
    Task Description:
    - Panda1 (left) picks up an object from the left table
    - Both robots coordinate to meet at a handover point
    - Panda1 transfers the object to Panda2
    - Panda2 places the object on the right table
    
    The reward function encourages:
    - Smooth, coordinated motion
    - Stable grasping and handover
    - Minimal forces during transfer
    - Task completion
    """
    
    def __init__(
        self,
        # Task weights
        phase_progress_weight: float = 2.0,
        coordination_weight: float = 1.5,
        grasp_stability_weight: float = 1.0,
        smoothness_weight: float = 0.5,
        force_penalty_weight: float = 0.3,
        ctrl_cost_weight: float = 1e-4,
        
        # Task parameters
        grasp_threshold: float = 0.02,
        handover_zone_radius: float = 0.1,
        place_threshold: float = 0.05,
        max_contact_force: float = 50.0,
        gripper_force_range: Tuple[float, float] = (5.0, 30.0),
        
        # Phase transition thresholds
        lift_height: float = 0.1,
        handover_height: float = 0.35,
        coordination_distance: float = 0.15,
        
        # Simulation parameters
        reset_noise_scale: float = 5e-3,
        backend: str = "mjx",
        **kwargs
    ):
        """Initialize the handover environment."""
        
        self.path = epath.resource_path("assistax") / "envs/assets/cooperative_handover.xml"
        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        
        if backend == "mjx":
            self.sys = self.sys.tree_replace({
                "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                "opt.iterations": 1,
                "opt.ls_iterations": 4,
            })
        
        # Store weights
        self._phase_progress_weight = phase_progress_weight
        self._coordination_weight = coordination_weight
        self._grasp_stability_weight = grasp_stability_weight
        self._smoothness_weight = smoothness_weight
        self._force_penalty_weight = force_penalty_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        
        # Store parameters
        self._grasp_threshold = grasp_threshold
        self._handover_zone_radius = handover_zone_radius
        self._place_threshold = place_threshold
        self._max_contact_force = max_contact_force
        self._gripper_force_range = gripper_force_range
        self._lift_height = lift_height
        self._handover_height = handover_height
        self._coordination_distance = coordination_distance
        self._reset_noise_scale = reset_noise_scale
        
        # Get model indices
        self._setup_indices(mjmodel)
        
        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        
        super().__init__(sys=self.sys, backend=backend, **kwargs)
    
    def _setup_indices(self, mjmodel):
        """Setup all necessary model indices."""
        GEOM_IDX = mujoco.mjtObj.mjOBJ_GEOM
        BODY_IDX = mujoco.mjtObj.mjOBJ_BODY
        SITE_IDX = mujoco.mjtObj.mjOBJ_SITE
        JOINT_IDX = mujoco.mjtObj.mjOBJ_JOINT
        ACTUATOR_IDX = mujoco.mjtObj.mjOBJ_ACTUATOR
        
        # Handover object indices (we probably dont need all of these)
        self.object_bod_idx = mj_name2id(mjmodel, BODY_IDX, "handover_object")
        self.object_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "box_object")
        self.object_site_idx = mj_name2id(mjmodel, SITE_IDX, "handover_object_site")
        
        # Panda1 indices
        self.panda1_rfinger_bod_idx = mj_name2id(mjmodel, BODY_IDX, "panda1_right_finger")
        self.panda1_lfinger_bod_idx = mj_name2id(mjmodel, BODY_IDX, "panda1_left_finger")
        self.panda1_rfinger_col_idx = mj_name2id(mjmodel, GEOM_IDX, "panda1_right_finger_pad") 
        self.panda1_lfinger_col_idx = mj_name2id(mjmodel, GEOM_IDX, "panda1_left_finger_pad")
        self.panda1_hand_bod_idx = mj_name2id(mjmodel, BODY_IDX, "panda1_hand")
        self.panda1_grip_site_idx = mj_name2id(mjmodel, SITE_IDX, "panda1_grip_site")
        # Panda2 indices  
        self.panda2_rfinger_bod_idx = mj_name2id(mjmodel, BODY_IDX, "panda2_right_finger")
        self.panda2_lfinger_bod_idx = mj_name2id(mjmodel, BODY_IDX, "panda2_left_finger")
        self.panda2_rfinger_col_idx = mj_name2id(mjmodel, GEOM_IDX, "panda2_right_finger_pad") 
        self.panda2_lfinger_col_idx = mj_name2id(mjmodel, GEOM_IDX, "panda2_left_finger_pad")
        self.panda2_hand_bod_idx = mj_name2id(mjmodel, BODY_IDX, "panda2_hand")
        self.panda2_grip_site_idx = mj_name2id(mjmodel, SITE_IDX, "panda2_grip_site")       
        # Goal sites
        self.pickup_goal_idx = mj_name2id(mjmodel, SITE_IDX, "pickup_goal")
        self.handover_goal_idx = mj_name2id(mjmodel, SITE_IDX, "handover_goal")
        self.place_goal_idx = mj_name2id(mjmodel, SITE_IDX, "place_goal")
        
        # (Check these) probably not correct. Joint indices for both robots
        self.panda1_joint_start = 7  # After object's freejoint
        self.panda1_joint_end = 15   # 7 arm joints + 1 gripper
        self.panda2_joint_start = 15
        self.panda2_joint_end = 23
        
        # (Check these) Actuator indices
        self.panda1_actuators = list(range(0, 8))
        self.panda2_actuators = list(range(8, 16))
    
    def reset(self, rng: jax.Array) -> State:
        """Reset the environment to initial state."""
        rng_pos, rng_vel = jax.random.split(rng, 2)
        
        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        init_q = self.sys.mj_model.keyframe("init").qpos
        
        # Add noise to positions
        qpos = init_q + jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi
        )
        
        pipeline_state = self.pipeline_init(qpos, qvel)
        
        # Initial observations
        obs = self._get_obs(pipeline_state)
        
        # Initialize metrics
        reward, done = jp.zeros(2)
        metrics = {
            "phase": HandoverPhase.APPROACH,
            "reward_phase": 0.0,
            "reward_coordination": 0.0,
            "reward_grasp": 0.0,
            "reward_smooth": 0.0,
            "reward_force": 0.0,
            "reward_ctrl": 0.0,
            "grasp_stability": 0.0,
            "handover_progress": 0.0,
        }
        
        info = {
            "phase": HandoverPhase.APPROACH,
            "prev_action": jp.zeros(16),  # Store for smoothness calculation
            "handover_initiated": False,
            "object_transferred": False,
        }
        
        return State(pipeline_state, obs, reward, done, metrics, info)
    
    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        """Execute one environment step."""
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        
        # Apply action
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        
        # Get observations
        obs = self._get_obs(pipeline_state)
        
        # Determine current phase
        phase = self._determine_phase(pipeline_state, state.info)
        
        # Calculate reward components
        rewards = self._calculate_rewards(
            pipeline_state, action, state.info, phase
        )
        
        # Total reward
        reward = (
            self._phase_progress_weight * rewards["phase_progress"] +
            self._coordination_weight * rewards["coordination"] +
            self._grasp_stability_weight * rewards["grasp_stability"] +
            self._smoothness_weight * rewards["smoothness"] +
            self._force_penalty_weight * rewards["force_penalty"] +
            self._ctrl_cost_weight * rewards["ctrl_cost"]
        )
        
        # Check if done
        done = self._check_done(pipeline_state, phase)
        
        # Update metrics
        state.metrics.update(
            phase=phase,
            reward_phase=rewards["phase_progress"],
            reward_coordination=rewards["coordination"],
            reward_grasp=rewards["grasp_stability"],
            reward_smooth=rewards["smoothness"],
            reward_force=rewards["force_penalty"],
            reward_ctrl=rewards["ctrl_cost"],
            grasp_stability=rewards["grasp_stability"],
            handover_progress=rewards["handover_progress"],
        )
        
        # Update info
        new_info = state.info.copy()
        new_info["phase"] = phase
        new_info["prev_action"] = action
        new_info.update(self._update_task_flags(pipeline_state, phase, state.info)) # This upadates the True flags whether handover has happened or not.
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            info=new_info,
        )
    
    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Get observation vector."""
        # Object state
        object_pos = pipeline_state.xpos[self.object_body_idx]
        object_quat = pipeline_state.xquat[self.object_body_idx]
        object_vel = pipeline_state.qvel[0:6]  # First 6 DOF for freejoint
        
        # Panda1 state
        panda1_ee_pos = pipeline_state.site_xpos[self.panda1_ee_site_idx]
        panda1_joints = pipeline_state.qpos[self.panda1_joint_start:self.panda1_joint_end]
        panda1_joint_vel = pipeline_state.qvel[self.panda1_joint_start:self.panda1_joint_end]
        
        # Panda2 state
        panda2_ee_pos = pipeline_state.site_xpos[self.panda2_ee_site_idx]
        panda2_joints = pipeline_state.qpos[self.panda2_joint_start:self.panda2_joint_end]
        panda2_joint_vel = pipeline_state.qvel[self.panda2_joint_start:self.panda2_joint_end]
        
        # Goal positions
        pickup_goal = pipeline_state.site_xpos[self.pickup_goal_idx]
        handover_goal = pipeline_state.site_xpos[self.handover_goal_idx]
        place_goal = pipeline_state.site_xpos[self.place_goal_idx]
        
        # Relative positions
        panda1_to_object = object_pos - panda1_ee_pos
        panda2_to_object = object_pos - panda2_ee_pos
        ee_to_ee = panda2_ee_pos - panda1_ee_pos
        
        # Contact forces
        panda1_contact_force = self._get_gripper_contact_force(pipeline_state, 1)
        panda2_contact_force = self._get_gripper_contact_force(pipeline_state, 2)
        
        return jp.concatenate([
            object_pos,              # 3
            object_quat,             # 4
            object_vel,              # 6
            panda1_ee_pos,           # 3
            panda1_joints,           # 8
            panda1_joint_vel,        # 8
            panda2_ee_pos,           # 3
            panda2_joints,           # 8
            panda2_joint_vel,        # 8
            pickup_goal,             # 3
            handover_goal,           # 3
            place_goal,              # 3
            panda1_to_object,        # 3
            panda2_to_object,        # 3
            ee_to_ee,                # 3
            panda1_contact_force,    # 6
            panda2_contact_force,    # 6
        ])
    
    def _determine_phase(self, pipeline_state: base.State, info: Dict) -> HandoverPhase:
        """Determine current phase of handover task."""
        object_pos = pipeline_state.pos[self.object_bod_idx]
        object_height = object_pos[2]
        
        panda1_ee_pos = pipeline_state.site_xpos[self.panda1_grip_site_idx]
        panda2_ee_pos = pipeline_state.site_xpos[self.panda2_grip_site_idx]
        
        # Check distances
        panda1_to_object = jp.linalg.norm(object_pos - panda1_ee_pos)
        panda2_to_object = jp.linalg.norm(object_pos - panda2_ee_pos)
        ee_distance = jp.linalg.norm(panda2_ee_pos - panda1_ee_pos)
        
        handover_pos = pipeline_state.site_xpos[self.handover_goal_idx]
        place_pos = pipeline_state.site_xpos[self.place_goal_idx]
        
        dist_to_handover = jp.linalg.norm(object_pos - handover_pos)
        dist_to_place = jp.linalg.norm(object_pos - place_pos)
        
        current_phase = info["phase"]
        
        # Phase transition logic
        if current_phase == HandoverPhase.APPROACH:
            if panda1_to_object < self._grasp_threshold:
                return HandoverPhase.GRASP
                
        elif current_phase == HandoverPhase.GRASP:
            panda1_grasp = self._check_grasp(pipeline_state, 1)
            if panda1_grasp and object_height > 0.2:
                return HandoverPhase.LIFT
                
        elif current_phase == HandoverPhase.LIFT:
            if object_height > self._handover_height - 0.05:
                return HandoverPhase.TRANSFER
                
        elif current_phase == HandoverPhase.TRANSFER:
            if dist_to_handover < self._handover_zone_radius:
                return HandoverPhase.HANDOVER
                
        elif current_phase == HandoverPhase.HANDOVER:
            panda2_grasp = self._check_grasp(pipeline_state, 2)
            if panda2_grasp and panda2_to_object < panda1_to_object:
                return HandoverPhase.RECEIVE
                
        elif current_phase == HandoverPhase.RECEIVE:
            panda1_grasp = self._check_grasp(pipeline_state, 1)
            if not panda1_grasp and ee_distance > self._coordination_distance:
                return HandoverPhase.RETREAT
                
        elif current_phase == HandoverPhase.RETREAT:
            if dist_to_place < self._handover_zone_radius:
                return HandoverPhase.PLACE
                
        elif current_phase == HandoverPhase.PLACE:
            if dist_to_place < self._place_threshold and object_height < 0.25:
                return HandoverPhase.COMPLETE
        
        return current_phase
    
    def _calculate_rewards(
        self, 
        pipeline_state: base.State,
        action: jax.Array,
        info: Dict,
        phase: HandoverPhase
    ) -> Dict[str, float]:
        """Calculate all reward components."""
        
        rewards = {}
        
        # 1. Phase Progress Reward
        rewards["phase_progress"] = self._calculate_phase_reward(pipeline_state, phase)
        
        # 2. Coordination Reward
        rewards["coordination"] = self._calculate_coordination_reward(pipeline_state, phase)
        
        # 3. Grasp Stability Reward
        rewards["grasp_stability"] = self._calculate_grasp_stability_reward(pipeline_state, phase)
        
        # 4. Smoothness Reward (penalizes jerky movements)
        prev_action = info.get("prev_action", jp.zeros_like(action))
        action_diff = jp.linalg.norm(action - prev_action)
        rewards["smoothness"] = -action_diff
        
        # 5. Force Penalty (penalizes excessive forces)
        rewards["force_penalty"] = self._calculate_force_penalty(pipeline_state)
        
        # 6. Control Cost
        rewards["ctrl_cost"] = -jp.sum(jp.square(action))
        
        # 7. Handover Progress (for metrics)
        rewards["handover_progress"] = phase / HandoverPhase.COMPLETE
        
        return rewards
    
    def _calculate_phase_reward(self, pipeline_state: base.State, phase: HandoverPhase) -> float:
        """Calculate reward based on phase-specific objectives."""
        
        object_pos = pipeline_state.xpos[self.object_body_idx]
        panda1_ee_pos = pipeline_state.site_xpos[self.panda1_ee_site_idx]
        panda2_ee_pos = pipeline_state.site_xpos[self.panda2_ee_site_idx]
        
        if phase == HandoverPhase.APPROACH:
            # Reward for getting close to object
            dist = jp.linalg.norm(object_pos - panda1_ee_pos)
            return jp.exp(-5 * dist)
            
        elif phase == HandoverPhase.GRASP:
            # Reward for maintaining grasp
            grasp_quality = self._get_grasp_quality(pipeline_state, 1)
            return grasp_quality
            
        elif phase == HandoverPhase.LIFT:
            # Reward for lifting to correct height
            target_height = self._handover_height
            height_error = jp.abs(object_pos[2] - target_height)
            return jp.exp(-10 * height_error)
            
        elif phase == HandoverPhase.TRANSFER:
            # Reward for moving to handover zone
            handover_pos = pipeline_state.site_xpos[self.handover_goal_idx]
            dist = jp.linalg.norm(object_pos - handover_pos)
            return jp.exp(-5 * dist)
            
        elif phase == HandoverPhase.HANDOVER:
            # Reward for coordinated handover
            ee_dist = jp.linalg.norm(panda2_ee_pos - panda1_ee_pos)
            optimal_dist = 0.1  # Optimal distance for handover
            return jp.exp(-10 * jp.abs(ee_dist - optimal_dist))
            
        elif phase == HandoverPhase.RECEIVE:
            # Reward for stable transfer
            grasp2 = self._get_grasp_quality(pipeline_state, 2)
            grasp1 = self._get_grasp_quality(pipeline_state, 1)
            return grasp2 * (1 - grasp1)  # Panda2 grips while Panda1 releases
            
        elif phase == HandoverPhase.RETREAT:
            # Reward for Panda1 moving away
            ee_dist = jp.linalg.norm(panda2_ee_pos - panda1_ee_pos)
            return jp.minimum(ee_dist / 0.5, 1.0)
            
        elif phase == HandoverPhase.PLACE:
            # Reward for placing at target
            place_pos = pipeline_state.site_xpos[self.place_goal_idx]
            dist = jp.linalg.norm(object_pos - place_pos)
            return jp.exp(-5 * dist)
            
        elif phase == HandoverPhase.COMPLETE:
            # Large bonus for task completion
            return 10.0
            
        return 0.0
    
    def _calculate_coordination_reward(self, pipeline_state: base.State, phase: HandoverPhase) -> float:
        """Reward coordinated motion between robots."""
        
        panda1_ee_pos = pipeline_state.site_xpos[self.panda1_ee_site_idx]
        panda2_ee_pos = pipeline_state.site_xpos[self.panda2_ee_site_idx]
        
        # During handover phases, robots should coordinate their positions
        if phase in [HandoverPhase.TRANSFER, HandoverPhase.HANDOVER, HandoverPhase.RECEIVE]:
            # Both end-effectors should be near the handover zone
            handover_pos = pipeline_state.site_xpos[self.handover_goal_idx]
            
            dist1 = jp.linalg.norm(panda1_ee_pos - handover_pos)
            dist2 = jp.linalg.norm(panda2_ee_pos - handover_pos)
            
            # Reward when both are close to handover zone
            coordination = jp.exp(-3 * dist1) * jp.exp(-3 * dist2)
            
            # Also reward appropriate relative positioning
            ee_dist = jp.linalg.norm(panda2_ee_pos - panda1_ee_pos)
            relative_reward = jp.exp(-10 * jp.abs(ee_dist - 0.1))
            
            return coordination * relative_reward
        
        return 0.0
    
    def _calculate_grasp_stability_reward(self, pipeline_state: base.State, phase: HandoverPhase) -> float:
        """Reward stable grasping during critical phases."""
        
        if phase in [HandoverPhase.GRASP, HandoverPhase.LIFT, HandoverPhase.TRANSFER]:
            # Panda1 should maintain stable grasp
            grasp_quality = self._get_grasp_quality(pipeline_state, 1)
            object_vel = jp.linalg.norm(pipeline_state.qvel[0:3])  # Linear velocity
            stability = grasp_quality * jp.exp(-object_vel)
            return stability
            
        elif phase in [HandoverPhase.HANDOVER, HandoverPhase.RECEIVE]:
            # Both robots involved in grasp
            grasp1 = self._get_grasp_quality(pipeline_state, 1)
            grasp2 = self._get_grasp_quality(pipeline_state, 2)
            
            # During handover, want smooth transition
            total_grasp = grasp1 + grasp2
            return jp.minimum(total_grasp, 1.0)
            
        elif phase in [HandoverPhase.RETREAT, HandoverPhase.PLACE]:
            # Panda2 should maintain stable grasp
            grasp_quality = self._get_grasp_quality(pipeline_state, 2)
            object_vel = jp.linalg.norm(pipeline_state.qvel[0:3])
            stability = grasp_quality * jp.exp(-object_vel)
            return stability
            
        return 0.0
    
    def _calculate_force_penalty(self, pipeline_state: base.State) -> float:
        """Penalize excessive contact forces."""
        
        # Get contact forces for both grippers
        force1 = self._get_gripper_contact_force(pipeline_state, 1)
        force2 = self._get_gripper_contact_force(pipeline_state, 2)
        
        # Penalize forces above threshold
        force_magnitude1 = jp.linalg.norm(force1[:3])  # Only consider linear forces
        force_magnitude2 = jp.linalg.norm(force2[:3])
        
        penalty1 = jp.maximum(0, force_magnitude1 - self._max_contact_force)
        penalty2 = jp.maximum(0, force_magnitude2 - self._max_contact_force)
        
        return -(penalty1 + penalty2) / self._max_contact_force
    
    def _check_grasp(self, pipeline_state: base.State, robot_id: int) -> bool:
        """Check if robot has grasped the object."""
        
        # Get contact force between gripper and object
        force = self._get_gripper_contact_force(pipeline_state, robot_id)
        force_magnitude = jp.linalg.norm(force[:3])
        
        # Check if force is within grasp range
        in_range = jp.logical_and(
            force_magnitude > self._gripper_force_range[0],
            force_magnitude < self._gripper_force_range[1]
        )
        
        # Also check gripper closure
        if robot_id == 1:
            gripper_pos = pipeline_state.qpos[self.panda1_joint_end - 1]
        else:
            gripper_pos = pipeline_state.qpos[self.panda2_joint_end - 1]
            
        gripper_closed = gripper_pos < 0.03  # Gripper mostly closed
        
        return jp.logical_and(in_range, gripper_closed)
    
    def _get_grasp_quality(self, pipeline_state: base.State, robot_id: int) -> float:
        """Get grasp quality score [0, 1]."""
        
        # Get contact force
        force = self._get_gripper_contact_force(pipeline_state, robot_id)
        force_magnitude = jp.linalg.norm(force[:3])
        
        # Normalize force to [0, 1] based on ideal range
        min_force, max_force = self._gripper_force_range
        quality = jp.clip((force_magnitude - min_force) / (max_force - min_force), 0, 1)
        
        # Also consider gripper closure
        if robot_id == 1:
            gripper_pos = pipeline_state.qpos[self.panda1_joint_end - 1]
        else:
            gripper_pos = pipeline_state.qpos[self.panda2_joint_end - 1]
            
        closure_quality = 1 - (gripper_pos / 0.04)  # Normalized gripper closure
        
        return quality * closure_quality
    
    def _get_gripper_contact_force(self, pipeline_state: base.State, robot_id: int) -> jax.Array:
        """Get contact force between gripper and object."""
        
        # Find contact between gripper fingers and object
        if robot_id == 1:
            finger_indices = [self.panda1_finger1_idx, self.panda1_finger2_idx]
        else:
            finger_indices = [self.panda2_finger1_idx, self.panda2_finger2_idx]
        
        total_force = jp.zeros(6)
        
        for finger_idx in finger_indices:
            # Find contact ID between finger and object
            for contact_id in range(len(pipeline_state.contact.geom)):
                geoms = pipeline_state.contact.geom[contact_id]
                if (geoms[0] == finger_idx and geoms[1] == self.object_geom_idx) or \
                   (geoms[1] == finger_idx and geoms[0] == self.object_geom_idx):
                    force = contact_force(self.sys, pipeline_state, contact_id, False)
                    total_force += force
        
        return total_force
    
    def _update_task_flags(self, pipeline_state: base.State, phase: HandoverPhase, info: Dict) -> Dict:
        """Update task completion flags."""
        
        updates = {}
        
        # Check if handover has been initiated
        if phase >= HandoverPhase.HANDOVER and not info.get("handover_initiated", False):
            updates["handover_initiated"] = True
        
        # Check if object has been transferred
        if phase >= HandoverPhase.RECEIVE and not info.get("object_transferred", False):
            panda1_grasp = self._check_grasp(pipeline_state, 1)
            panda2_grasp = self._check_grasp(pipeline_state, 2)
            if panda2_grasp and not panda1_grasp:
                updates["object_transferred"] = True
        
        return updates
    
    def _check_done(self, pipeline_state: base.State, phase: HandoverPhase) -> float:
        """Check if episode is done."""
        
        # Success: task completed
        if phase == HandoverPhase.COMPLETE:
            return 1.0
        
        # Failure: object dropped
        object_height = pipeline_state.xpos[self.object_body_idx][2]
        if object_height < 0.1 and phase not in [HandoverPhase.APPROACH, HandoverPhase.GRASP]:
            return 1.0
        
        # Failure: robots collision (simplified check - distance too close)
        panda1_ee_pos = pipeline_state.site_xpos[self.panda1_ee_site_idx]
        panda2_ee_pos = pipeline_state.site_xpos[self.panda2_ee_site_idx]
        ee_distance = jp.linalg.norm(panda2_ee_pos - panda1_ee_pos)
        if ee_distance < 0.05 and phase not in [HandoverPhase.HANDOVER, HandoverPhase.RECEIVE]:
            return 1.0
        
        return 0.0