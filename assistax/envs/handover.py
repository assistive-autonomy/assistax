from typing import Tuple
from enum import IntEnum

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
import jax
from jax import numpy as jp
import mujoco 
from mujoco import mj_id2name, mj_name2id
from mujoco.mjx._src.support import contact_force


class HandoverPhase(IntEnum):
    """Enumeration of handover task phases."""
    APPROACH = 0
    GRASP = 1
    TRANSFER = 2
    HANDOVER = 3
    RETREAT = 4
    PLACE = 5


class CooperativeHandover(PipelineEnv):
    """
    Cooperative handover environment where two Panda robots pass an object.
    
    The task involves six phases:
    1. APPROACH: Panda1 moves towards the object
    2. GRASP: Panda1 grasps the object
    3. TRANSFER: Panda1 lifts and moves object to handover location
    4. HANDOVER: Panda2 approaches and both robots grip object
    5. RETREAT: Panda2 takes object while Panda1 releases
    6. PLACE: Panda2 places object at goal location
    """

    def __init__(
        self,
        # Reward weights
        dist_reward_weight: float = 1.0,
        grasp_reward_weight: float = 2.0,
        maintain_grip_weight: float = 1.0,
        handover_reward_weight: float = 3.0,
        place_reward_weight: float = 2.0,
        ctrl_cost_weight: float = 1e-4,
        drop_penalty: float = -10.0,
        collision_penalty: float = -5.0,
        phase_transition_bonus: float = 5.0, # Maybe we tune this for sparse rewards?  
        
        # Scaling factors
        dist_scale: float = 0.1,
        force_scale: float = 0.01,
        
        # Phase transition thresholds
        phase1_dist_threshold: float = 0.05,  # Distance for approach->grasp
        phase2_force_threshold: float = 0.5,   # Force for grasp->transfer
        phase3_dist_threshold: float = 0.15,   # Distance for transfer->handover
        phase4_dual_grip_threshold: float = 0.5, # Both gripping for handover->retreat
        phase5_dist_threshold: float = 0.08,   # Distance for retreat->place
        place_contact_threshold: float = 0.02, # Contact with table for completion
        
        # General parameters
        reset_noise_scale: float = 5e-3,
        backend: str = "mjx",
        **kwargs
    ):
        """Creates a CooperativeHandover Environment."""
        
        # Load XML model
        self.path = epath.resource_path("assistax") / "envs/assets/handover.xml"
        mjmodel = mujoco.MjModel.from_xml_path(str(self.path))
        self.sys = mjcf.load_model(mjmodel)
        
        if backend == "mjx":
            self.sys = self.sys.tree_replace({
                "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                "opt.iterations": 1,
                "opt.ls_iterations": 4,
            })

        # MuJoCo object type indices
        GEOM_IDX = mujoco.mjtObj.mjOBJ_GEOM
        BODY_IDX = mujoco.mjtObj.mjOBJ_BODY
        SITE_IDX = mujoco.mjtObj.mjOBJ_SITE
        
        # Panda1 (left robot) indices
        self.panda1_grip_site_idx = mj_name2id(mjmodel, SITE_IDX, "panda1_grip_site")
        self.panda1_hand_body_idx = mj_name2id(mjmodel, BODY_IDX, "panda1_hand")
        self.panda1_left_finger_geom = mj_name2id(mjmodel, GEOM_IDX, "panda1_leftfinger_collision1")
        self.panda1_right_finger_geom = mj_name2id(mjmodel, GEOM_IDX, "panda1_rightfinger_collision1")
        
        # Panda2 (right robot) indices
        self.panda2_grip_site_idx = mj_name2id(mjmodel, SITE_IDX, "panda2_grip_site")
        self.panda2_hand_body_idx = mj_name2id(mjmodel, BODY_IDX, "panda2_hand")
        self.panda2_left_finger_geom = mj_name2id(mjmodel, GEOM_IDX, "panda2_leftfinger_collision1")
        self.panda2_right_finger_geom = mj_name2id(mjmodel, GEOM_IDX, "panda2_rightfinger_collision1")
        
        # Object indices
        self.object_body_idx = mj_name2id(mjmodel, BODY_IDX, "handover_object")
        self.object_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "box_object")
        
        # Goal location indices
        self.handover_goal_idx = mj_name2id(mjmodel, SITE_IDX, "handover_goal")
        self.place_goal_idx = mj_name2id(mjmodel, SITE_IDX, "place_goal")
        self.pickup_goal_idx = mj_name2id(mjmodel, SITE_IDX, "pickup_goal")
        
        # Table indices for place detection
        self.table_right_geom = mj_name2id(mjmodel, GEOM_IDX, "table_right_top")

        # Floor geom index for contact detection
        self.floor_geom_idx = mj_name2id(mjmodel, GEOM_IDX, "floor")

        # Joint indices for observations
        # Object has 7 DOFs (3 pos + 4 quat) starting at index 0
        self.object_joint_start = 0
        self.object_joint_end = 7
        
        # Panda1 joints (7 arm + 2 fingers)
        self.panda1_joint_start = 7
        self.panda1_joint_end = 16  # 7 arm + 2 fingers
        
        # Panda2 joints (7 arm + 2 fingers)
        self.panda2_joint_start = 16
        self.panda2_joint_end = 25
        
        # Touch sensor indices
        self.panda1_left_inner_touch_idx = 0
        self.panda1_right_inner_touch_idx = 1
        self.panda1_left_outer_touch_idx = 2
        self.panda1_right_outer_touch_idx = 3
        self.panda2_left_inner_touch_idx = 4
        self.panda2_right_inner_touch_idx = 5
        self.panda2_left_outer_touch_idx = 6
        self.panda2_right_outer_touch_idx = 7

        # Contact IDs 
        self.object_floor_contact_id1 = 484
        self.object_floor_contact_id2 = 485
        self.object_floor_contact_id3 = 486
        self.object_floor_contact_id4 = 487
        # Store reward weights
        self._dist_reward_weight = dist_reward_weight
        self._grasp_reward_weight = grasp_reward_weight
        self._maintain_grip_weight = maintain_grip_weight
        self._handover_reward_weight = handover_reward_weight
        self._place_reward_weight = place_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._drop_penalty = drop_penalty
        self._collision_penalty = collision_penalty
        self._phase_transition_bonus = phase_transition_bonus
        
        # Store scaling factors
        self._dist_scale = dist_scale
        self._force_scale = force_scale
        
        # Store thresholds
        self._phase1_dist_threshold = phase1_dist_threshold
        self._phase2_force_threshold = phase2_force_threshold
        self._phase3_dist_threshold = phase3_dist_threshold
        self._phase4_dual_grip_threshold = phase4_dual_grip_threshold
        self._phase5_dist_threshold = phase5_dist_threshold
        self._place_contact_threshold = place_contact_threshold
        
        self._reset_noise_scale = reset_noise_scale
        
        n_frames = 4
        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)
        
        super().__init__(sys=self.sys, backend=backend, **kwargs)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng_pos, rng_vel = jax.random.split(rng, 2)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        init_q = self.sys.mj_model.keyframe("init").qpos if "init" in [
            self.sys.mj_model.key(i).name for i in range(self.sys.mj_model.nkey)
        ] else self.sys.init_q
        
        qpos = init_q + jax.random.uniform(
            rng_pos, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qvel = jax.random.uniform(
            rng_vel, (self.sys.qd_size(),), minval=low, maxval=hi
        )

        pipeline_state = self.pipeline_init(qpos, qvel)
        
        obs = self._get_obs(pipeline_state, HandoverPhase.APPROACH)
        
        reward, done = jp.zeros(2)
        metrics = {
            "reward_dist": 0.0,
            "reward_grasp": 0.0,
            "reward_maintain_grip": 0.0,
            "reward_handover": 0.0,
            "reward_place": 0.0,
            "reward_ctrl": 0.0,
            "penalty_drop": 0.0,
            "penalty_collision": 0.0,
            "phase_transition_bonus": 0.0,
            "phase": HandoverPhase.APPROACH,
        }
        
        info = {
            "phase": HandoverPhase.APPROACH,
            "prev_object_pos": pipeline_state.xpos[self.object_body_idx],
            "panda1_gripping": False,
            "panda2_gripping": False,
        }
        
        return State(pipeline_state, obs, reward, done, metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        
        # Get current phase
        current_phase = state.info["phase"]
        
        # Compute phase-specific rewards
        rewards_dict = self._compute_rewards(pipeline_state, action, current_phase, state.info)
        
        # Check for phase transitions
        new_phase, phase_transition_reward = self._check_phase_transition(
            pipeline_state, current_phase, state.info
        )
        
        # Total reward
        total_reward = (
            rewards_dict["dist"] +
            rewards_dict["grasp"] +
            rewards_dict["maintain_grip"] +
            rewards_dict["handover"] +
            rewards_dict["place"] +
            rewards_dict["ctrl"] +
            rewards_dict["drop"] +
            rewards_dict["collision"] +
            phase_transition_reward
        )
        
        # Check termination conditions
        done = self._check_done(pipeline_state, new_phase, rewards_dict)
        
        # Get observations
        obs = self._get_obs(pipeline_state, new_phase)
        
        # Update info
        panda1_gripping = self._check_gripper_contact(pipeline_state, robot_id=1)
        panda2_gripping = self._check_gripper_contact(pipeline_state, robot_id=2)
        
        new_info = {
            "phase": new_phase,
            "prev_object_pos": pipeline_state.xpos[self.object_body_idx],
            "panda1_gripping": panda1_gripping,
            "panda2_gripping": panda2_gripping,
        }
        
        # Update metrics
        state.metrics.update(
            reward_dist=rewards_dict["dist"],
            reward_grasp=rewards_dict["grasp"],
            reward_maintain_grip=rewards_dict["maintain_grip"],
            reward_handover=rewards_dict["handover"],
            reward_place=rewards_dict["place"],
            reward_ctrl=rewards_dict["ctrl"],
            penalty_drop=rewards_dict["drop"],
            penalty_collision=rewards_dict["collision"],
            phase_transition_bonus=phase_transition_reward,
            phase=new_phase,
        )
        
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=total_reward,
            done=done,
            info=new_info,
        )

    def _get_obs(self, pipeline_state: base.State, phase: int) -> jax.Array:
        """Constructs observation vector."""
        
        # Panda1 observations
        panda1_joint_pos = pipeline_state.qpos[self.panda1_joint_start:self.panda1_joint_end]
        panda1_joint_vel = pipeline_state.qvel[self.panda1_joint_start-7:self.panda1_joint_end-7] # Why don't we just double check this as well (or we might not need this actually)
        panda1_ee_pos = pipeline_state.site_xpos[self.panda1_grip_site_idx]
        panda1_hand_quat = pipeline_state.xquat[self.panda1_hand_body_idx]
        
        # Panda2 observations
        panda2_joint_pos = pipeline_state.qpos[self.panda2_joint_start:self.panda2_joint_end]
        panda2_joint_vel = pipeline_state.qvel[self.panda2_joint_start-7:self.panda2_joint_end-7]
        panda2_ee_pos = pipeline_state.site_xpos[self.panda2_grip_site_idx]
        panda2_hand_quat = pipeline_state.xquat[self.panda2_hand_body_idx]
        
        # Object observations
        object_pos = pipeline_state.xpos[self.object_body_idx]
        object_quat = pipeline_state.xquat[self.object_body_idx]
        object_vel = pipeline_state.qvel[self.object_joint_start:self.object_joint_start+3]
        object_angvel = pipeline_state.qvel[self.object_joint_start+3:self.object_joint_end]
        
        # Touch sensor readings
        panda1_touch = jp.array([
            pipeline_state.sensordata[self.panda1_left_inner_touch_idx],
            pipeline_state.sensordata[self.panda1_right_inner_touch_idx],
            pipeline_state.sensordata[self.panda1_left_outer_touch_idx],
            pipeline_state.sensordata[self.panda1_right_outer_touch_idx],
        ])
        
        panda2_touch = jp.array([
            pipeline_state.sensordata[self.panda2_left_inner_touch_idx],
            pipeline_state.sensordata[self.panda2_right_inner_touch_idx],
            pipeline_state.sensordata[self.panda2_left_outer_touch_idx],
            pipeline_state.sensordata[self.panda2_right_outer_touch_idx],
        ])
        
        # Goal positions
        handover_goal_pos = pipeline_state.site_xpos[self.handover_goal_idx]
        place_goal_pos = pipeline_state.site_xpos[self.place_goal_idx]
        
        # Phase as one-hot encoding
        phase_onehot = jp.zeros(6)
        phase_onehot = phase_onehot.at[phase].set(1.0) # This is interesting as well. 
        
        # Concatenate all observations
        obs = jp.concatenate([
            panda1_joint_pos,
            panda1_joint_vel,
            panda1_ee_pos,
            panda1_hand_quat,
            panda2_joint_pos,
            panda2_joint_vel,
            panda2_ee_pos,
            panda2_hand_quat,
            object_pos,
            object_quat,
            object_vel,
            object_angvel,
            panda1_touch,
            panda2_touch,
            handover_goal_pos,
            place_goal_pos,
            phase_onehot,
        ]) # These are actually also wrong (I need duplicates to split them up correctly (or actually maybe I could have them overlap somehow))
        
        return obs

    def _compute_rewards(
        self, 
        pipeline_state: base.State, 
        action: jax.Array,
        phase: int,
        info: dict
        ) -> dict:
        """Computes phase-specific rewards."""
        
        # Control cost (always applied)
        ctrl_reward = -self._ctrl_cost_weight * jp.sum(jp.square(action))
        
        # Drop penalty (check if object fell)
        dropped = self._get_object_dropped(pipeline_state, self.object_floor_contact_id1, self.object_floor_contact_id2, self.object_floor_contact_id3, self.object_floor_contact_id4) # Object below minimum height
        drop_reward = jp.where(dropped, self._drop_penalty, 0.0)
        
        # Collision penalty between robots
        robot_collision = self._check_robot_collision(pipeline_state)
        collision_reward = jp.where(robot_collision, self._collision_penalty, 0.0)
        
        # Compute all possible phase-specific rewards
        approach_reward = self._reward_approach(pipeline_state)
        grasp_reward = self._reward_grasp(pipeline_state, robot_id=1)
        transfer_reward = self._reward_transfer(pipeline_state)
        handover_reward = self._reward_handover(pipeline_state)
        retreat_reward = self._reward_retreat(pipeline_state)
        place_reward = self._reward_place(pipeline_state)
        
        maintain_grip_1 = self._reward_maintain_grip(pipeline_state, info, robot_id=1)
        maintain_grip_2 = self._reward_maintain_grip(pipeline_state, info, robot_id=2)
        
        # Use jax.lax.switch to select rewards based on phase
        # Phase 0 (APPROACH): dist only
        # Phase 1 (GRASP): dist + grasp
        # Phase 2 (TRANSFER): maintain_grip_1 + transfer
        # Phase 3 (HANDOVER): maintain_grip_1 + handover
        # Phase 4 (RETREAT): maintain_grip_2 + retreat
        # Phase 5 (PLACE): maintain_grip_2 + place
        
        dist_reward = jax.lax.switch(
            phase,
            [
                lambda: approach_reward,  # APPROACH
                lambda: approach_reward,  # GRASP
                lambda: transfer_reward,  # TRANSFER
                lambda: 0.0,              # HANDOVER
                lambda: retreat_reward,   # RETREAT
                lambda: 0.0,              # PLACE
            ]
        )
        
        grasp_reward_final = jax.lax.switch(
            phase,
            [
                lambda: 0.0,          # APPROACH
                lambda: grasp_reward, # GRASP
                lambda: 0.0,          # TRANSFER
                lambda: 0.0,          # HANDOVER
                lambda: 0.0,          # RETREAT
                lambda: 0.0,          # PLACE
            ]
        )
        
        maintain_grip_reward = jax.lax.switch(
            phase,
            [
                lambda: 0.0,            # APPROACH
                lambda: 0.0,            # GRASP
                lambda: maintain_grip_1, # TRANSFER
                lambda: maintain_grip_1, # HANDOVER
                lambda: maintain_grip_2, # RETREAT
                lambda: maintain_grip_2, # PLACE
            ]
        )
        
        handover_reward_final = jax.lax.switch(
            phase,
            [
                lambda: 0.0,             # APPROACH
                lambda: 0.0,             # GRASP
                lambda: 0.0,             # TRANSFER
                lambda: handover_reward, # HANDOVER
                lambda: 0.0,             # RETREAT
                lambda: 0.0,             # PLACE
            ]
        )
        
        place_reward_final = jax.lax.switch(
            phase,
            [
                lambda: 0.0,          # APPROACH
                lambda: 0.0,          # GRASP
                lambda: 0.0,          # TRANSFER
                lambda: 0.0,          # HANDOVER
                lambda: 0.0,          # RETREAT
                lambda: place_reward, # PLACE
            ]
        )
        
        rewards = {
            "dist": dist_reward,
            "grasp": grasp_reward_final,
            "maintain_grip": maintain_grip_reward,
            "handover": handover_reward_final,
            "place": place_reward_final,
            "ctrl": ctrl_reward,
            "drop": drop_reward,
            "collision": collision_reward,
        }
        
        return rewards

    def _reward_approach(self, pipeline_state: base.State) -> float: # Cururently this seems to be only used for robot1 
        """Reward for approaching object with Panda1."""
        ee_pos = pipeline_state.site_xpos[self.panda1_grip_site_idx]
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        dist = jp.linalg.norm(ee_pos - obj_pos)
        return self._dist_reward_weight * jp.exp(-dist**2 / self._dist_scale)

    def _reward_grasp(self, pipeline_state: base.State, robot_id: int) -> float:
        """Reward for grasping with appropriate force."""
        # Select touch sensors based on robot_id using jax.lax.switch
        touch_sensors = jax.lax.switch(
            robot_id - 1,  # Convert to 0-indexed
            [
                lambda: jp.array([
                    pipeline_state.sensordata[self.panda1_left_inner_touch_idx],
                    pipeline_state.sensordata[self.panda1_right_inner_touch_idx],
                ]),
                lambda: jp.array([
                    pipeline_state.sensordata[self.panda2_left_inner_touch_idx],
                    pipeline_state.sensordata[self.panda2_right_inner_touch_idx],
                ])
            ]
        )
        
        # Reward for contact on both fingers
        both_touching = jp.all(touch_sensors > 0.01)
        
        # Reward for appropriate force (not too weak, not too strong)
        mean_force = jp.mean(touch_sensors)
        force_in_range = (mean_force > 0.1) & (mean_force < 5.0)
        
        return self._grasp_reward_weight * (
            both_touching.astype(jp.float32) + force_in_range.astype(jp.float32)
        )

    def _reward_maintain_grip( # I don't thnk I like this definition currently. 
        self, 
        pipeline_state: base.State, 
        info: dict,
        robot_id: int
    ) -> float:
        """Reward for maintaining grip on object."""
        gripping = jax.lax.switch(
            robot_id - 1,
            [
                lambda: info["panda1_gripping"],
                lambda: info["panda2_gripping"]
            ]
        )
        
        # Select end effector position based on robot_id
        ee_pos = jax.lax.switch(
            robot_id - 1,
            [
                lambda: pipeline_state.site_xpos[self.panda1_grip_site_idx],
                lambda: pipeline_state.site_xpos[self.panda2_grip_site_idx]
            ]
        )
            
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        relative_motion = jp.linalg.norm(obj_pos - ee_pos) # I don't think this is correct. 
        stable = relative_motion < 0.1
        
        return self._maintain_grip_weight * (
            gripping.astype(jp.float32) + stable.astype(jp.float32)
        )

    def _reward_transfer(self, pipeline_state: base.State) -> float:
        """Reward for moving object to handover location."""
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        handover_pos = pipeline_state.site_xpos[self.handover_goal_idx]
        dist = jp.linalg.norm(obj_pos - handover_pos)
        return self._dist_reward_weight * jp.exp(-dist**2 / self._dist_scale)

    def _reward_handover(self, pipeline_state: base.State) -> float:
        """Reward for successful handover."""
        # Panda2 approaching object
        ee2_pos = pipeline_state.site_xpos[self.panda2_grip_site_idx]
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        dist = jp.linalg.norm(ee2_pos - obj_pos)
        approach_reward = jp.exp(-dist**2 / self._dist_scale)
        
        # Both robots gripping
        panda1_touch = jp.mean(jp.array([
            pipeline_state.sensordata[self.panda1_left_inner_touch_idx],
            pipeline_state.sensordata[self.panda1_right_inner_touch_idx],
        ]))
        panda2_touch = jp.mean(jp.array([
            pipeline_state.sensordata[self.panda2_left_inner_touch_idx],
            pipeline_state.sensordata[self.panda2_right_inner_touch_idx],
        ]))
        
        dual_grip_reward = (panda1_touch > 0.1).astype(jp.float32) * (panda2_touch > 0.1).astype(jp.float32)
        
        return self._handover_reward_weight * (approach_reward + dual_grip_reward)

    def _reward_retreat(self, pipeline_state: base.State) -> float:
        """Reward for moving object to place location."""
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        place_pos = pipeline_state.site_xpos[self.place_goal_idx]
        dist = jp.linalg.norm(obj_pos - place_pos)
        return self._dist_reward_weight * jp.exp(-dist**2 / self._dist_scale)

    def _reward_place(self, pipeline_state: base.State) -> float:
        """Reward for placing object at goal."""
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        place_pos = pipeline_state.site_xpos[self.place_goal_idx]
        
        # Distance reward
        dist = jp.linalg.norm(obj_pos - place_pos)
        dist_reward = jp.exp(-dist**2 / self._dist_scale)
        
        # Contact with table reward
        obj_z = obj_pos[2]
        table_z = 0.165  # Approximate table height
        on_table = jp.abs(obj_z - table_z) < self._place_contact_threshold
        
        # Upright orientation bonus
        obj_quat = pipeline_state.xquat[self.object_body_idx]
        upright = jp.abs(obj_quat[0]) > 0.9  # Close to no rotation
        
        return self._place_reward_weight * (
            dist_reward + 
            on_table.astype(jp.float32) + 
            upright.astype(jp.float32)
        )

    def _check_phase_transition(
        self, 
        pipeline_state: base.State,
        current_phase: int,
        info: dict
    ) -> Tuple[int, float]:
        """Checks and executes phase transitions."""
        
        # Compute all transition conditions
        # APPROACH -> GRASP
        panda1_ee_pos = pipeline_state.site_xpos[self.panda1_grip_site_idx]
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        panda1_dist_to_obj = jp.linalg.norm(panda1_ee_pos - obj_pos)

        approach_complete = panda1_dist_to_obj < self._phase1_dist_threshold

        # GRASP -> TRANSFER
        panda1_touch = jp.mean(jp.array([
            pipeline_state.sensordata[self.panda1_left_inner_touch_idx],
            pipeline_state.sensordata[self.panda1_right_inner_touch_idx],
        ]))
        grasp_complete = (panda1_touch > self._phase2_force_threshold) & (panda1_dist_to_obj > 0.05) # or should I replace this with the check contact function?
        
        # TRANSFER -> HANDOVER
        handover_pos = pipeline_state.site_xpos[self.handover_goal_idx]
        dist_to_handover = jp.linalg.norm(obj_pos - handover_pos)
        transfer_complete = dist_to_handover < self._phase3_dist_threshold
        
        # HANDOVER -> RETREAT
        panda2_ee_pos = pipeline_state.site_xpos[self.panda2_grip_site_idx]
        panda2_dist_to_obj = jp.linalg.norm(panda2_ee_pos - obj_pos)
        # Here we should actually just use self._check_gripper_contact function. 
        panda2_touch = jp.mean(jp.array([
            pipeline_state.sensordata[self.panda2_left_inner_touch_idx],
            pipeline_state.sensordata[self.panda2_right_inner_touch_idx],
        ]))
        both_gripping = (panda1_touch > self._phase4_dual_grip_threshold) & (
            panda2_touch > self._phase4_dual_grip_threshold
        ) & (
            panda2_dist_to_obj < 0.05
        )
        handover_complete = both_gripping
        
        # RETREAT -> PLACE
        place_pos = pipeline_state.site_xpos[self.place_goal_idx]
        dist_to_place = jp.linalg.norm(obj_pos - place_pos)
        retreat_complete = dist_to_place < self._phase5_dist_threshold
        
        # Use jax.lax.switch to determine new phase and bonus based on current phase
        def phase_0_transition():  # APPROACH
            return jp.where(approach_complete, HandoverPhase.GRASP, HandoverPhase.APPROACH), \
                   jp.where(approach_complete, self._phase_transition_bonus, 0.0)
        
        def phase_1_transition():  # GRASP
            return jp.where(grasp_complete, HandoverPhase.TRANSFER, HandoverPhase.GRASP), \
                   jp.where(grasp_complete, self._phase_transition_bonus, 0.0)
        
        def phase_2_transition():  # TRANSFER
            return jp.where(transfer_complete, HandoverPhase.HANDOVER, HandoverPhase.TRANSFER), \
                   jp.where(transfer_complete, self._phase_transition_bonus, 0.0)
        
        def phase_3_transition():  # HANDOVER
            return jp.where(handover_complete, HandoverPhase.RETREAT, HandoverPhase.HANDOVER), \
                   jp.where(handover_complete, self._phase_transition_bonus, 0.0)
        
        def phase_4_transition():  # RETREAT
            return jp.where(retreat_complete, HandoverPhase.PLACE, HandoverPhase.RETREAT), \
                   jp.where(retreat_complete, self._phase_transition_bonus, 0.0)
        
        def phase_5_transition():  # PLACE
            return HandoverPhase.PLACE, 0.0  # Stay in PLACE phase
        
        new_phase, transition_bonus = jax.lax.switch(
            current_phase,
            [
                phase_0_transition,
                phase_1_transition,
                phase_2_transition,
                phase_3_transition,
                phase_4_transition,
                phase_5_transition,
            ]
        )
        
        return new_phase, transition_bonus

    def _check_gripper_contact(self, pipeline_state: base.State, robot_id: int) -> bool:
        """Checks if gripper is in contact with object."""
        touch = jax.lax.switch(
            robot_id - 1,
            [
                lambda: jp.mean(jp.array([
                    pipeline_state.sensordata[self.panda1_left_inner_touch_idx],
                    pipeline_state.sensordata[self.panda1_right_inner_touch_idx],
                ])),
                lambda: jp.mean(jp.array([
                    pipeline_state.sensordata[self.panda2_left_inner_touch_idx],
                    pipeline_state.sensordata[self.panda2_right_inner_touch_idx],
                ]))
            ]
        )

        dist = jax.lax.switch(
            robot_id - 1,
            [
                lambda: jp.linalg.norm(
                    pipeline_state.site_xpos[self.panda1_grip_site_idx] - pipeline_state.xpos[self.object_body_idx]
                ),
                lambda: jp.linalg.norm(
                    pipeline_state.site_xpos[self.panda2_grip_site_idx] - pipeline_state.xpos[self.object_body_idx]
                )
            ]
        )
        
        return (touch > 0.1) & dist <0.05

    def _check_robot_collision(self, pipeline_state: base.State) -> bool:
        """Checks if robots are colliding with each other."""
        # Check distance between robot hands
        ee1_pos = pipeline_state.site_xpos[self.panda1_grip_site_idx]
        ee2_pos = pipeline_state.site_xpos[self.panda2_grip_site_idx]
        dist = jp.linalg.norm(ee1_pos - ee2_pos)
        
        # Collision if hands too close (excluding handover phase)
        collision_threshold = 0.08
        return dist < collision_threshold

    def _check_done(
        self, 
        pipeline_state: base.State, 
        phase: int,
        rewards: dict
    ) -> float:
        """Checks termination conditions."""
        
        # Check success condition (object placed at goal)
        obj_pos = pipeline_state.xpos[self.object_body_idx]
        place_pos = pipeline_state.site_xpos[self.place_goal_idx]
        dist = jp.linalg.norm(obj_pos - place_pos)
        obj_z = obj_pos[2]
        table_z = 0.165
        
        placed_successfully = (dist < self._place_contact_threshold) & (
            jp.abs(obj_z - table_z) < self._place_contact_threshold
        ) & (phase == HandoverPhase.PLACE)
        
        # Check failure conditions


        dropped = obj_z < 0.1  # Object dropped
        severe_collision = rewards["collision"] < -2.0  # Severe robot collision
        
        # Any termination condition met
        done = placed_successfully | dropped | severe_collision
        
        return done.astype(jp.float32)

    def _get_object_dropped(self, pipeline_state, object_floor_contact_id1, object_floor_contact_id2, object_floor_contact_id3, object_floor_contact_id4) -> bool:
        """Check if the object has made contact with the floor."""

        contact_force1 = contact_force(self.sys, pipeline_state, object_floor_contact_id1)
        contact_force2 = contact_force(self.sys, pipeline_state, object_floor_contact_id2)
        contact_force3 = contact_force(self.sys, pipeline_state, object_floor_contact_id3)
        contact_force4 = contact_force(self.sys, pipeline_state, object_floor_contact_id4)

        total_contact_force = jp.sum(jp.vstack([contact_force1, contact_force2, contact_force3, contact_force4]))
        return total_contact_force > 0.0  # Threshold could change based on what we consider "dropped"

    @property
    def action_size(self) -> int:
        """Returns the size of the action space."""
        # 8 actuators per robot (7 arm joints + 1 gripper)
        return 16

    @property  
    def observation_size(self) -> int:
        """Returns the size of the observation space."""
        # Panda1: 9 joint pos + 9 joint vel + 3 ee pos + 4 hand quat = 25
        # Panda2: 9 joint pos + 9 joint vel + 3 ee pos + 4 hand quat = 25
        # Object: 3 pos + 4 quat + 3 vel + 4 angvel = 14
        # Touch sensors: 4 (panda1) + 4 (panda2) = 8
        # Goal positions: 3 (handover) + 3 (place) = 6
        # Phase one-hot: 6
        # Total: 25 + 25 + 14 + 8 + 6 + 6 = 84
        return 84