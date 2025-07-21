"""The disability wrapper which applied persistent disabilities to the human agent in the various tasks."""

from typing import Callable, Dict, Optional, Tuple, Any

from brax.base import System
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp

class DisabilityWrapper(Wrapper):
    """Wrapper for applying persistent disabilities."""

    def __init__(
        self,
        env: Env,
        cfg: dict[str, Any],
    ):
        super().__init__(env)
        self.disability_jnt_idx = cfg["joint_idx"]
        self.disability_mask = (
            jp.zeros(self.env.action_size, bool).at[self.disability_jnt_idx].set(True)
        )
        self.joint_restriction_factor = cfg.get("joint_restriction_factor", 1.0)
        self.joint_strength = cfg.get("joint_strength", 1.0)
        self.tremor_magnitude = cfg.get("tremor_magnitude", 0.0)
        orig_joint_range = self.env.unwrapped.sys.jnt_range[self.disability_jnt_idx]
        new_joint_range = self.joint_restriction_factor * orig_joint_range
        self.env.unwrapped.sys = self.env.unwrapped.sys.replace(
            jnt_range=self.env.unwrapped.sys.jnt_range.at[self.disability_jnt_idx].set(
                new_joint_range
            )
        )

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        rng_act, rng_env = jax.random.split(rng)
        modified_action = self._modify_action(rng_act, action)
        # N.B. we might want to handle control cost before passing to step
        return self.env.step(rng_env, state, modified_action)

    def _modify_action(self, rng: jax.Array, action: jax.Array) -> jax.Array:
        tremor_action = self.joint_strength * (
            action + self.tremor_magnitude * jax.random.uniform(rng)
        )
        tremor_action = jp.where(self.disability_mask, tremor_action, action)
        return tremor_action