# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import, g-importing-member
"""Wrappers to support Brax training."""

from typing import Callable, Dict, Optional, Tuple, Any

from brax.base import System
from brax.envs.base import Env, PipelineEnv, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp

from .pixels.rendering_utils import PixelState, build_objects_for_cache
from .pixels import rendering_utils as ru

def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[Callable[[System], Tuple[System, System]]] = None,
) -> Wrapper:
    """Common wrapper pattern for all training agents.

    Args:
      env: environment to be wrapped
      episode_length: length of episode
      action_repeat: how many repeated actions to take per step
      randomization_fn: randomization function that produces a vectorized system
        and in_axes to vmap over

    Returns:
      An environment that is wrapped with Episode and AutoReset wrappers.  If the
      environment did not already have batch dimensions, it is additional Vmap
      wrapped.
    """
    env = EpisodeWrapper(env, episode_length, action_repeat)
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = AutoResetWrapper(env)
    return env


class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        return jax.vmap(self.env.reset)(rng)

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(rng, state, action)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["steps"] = jp.zeros(rng.shape[:-1])
        state.info["truncation"] = jp.zeros(rng.shape[:-1])
        return state

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(rng, state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info["steps"] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info["truncation"] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info["steps"] = steps
        return state.replace(done=done)


class AutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info["first_pipeline_state"] = state.pipeline_state
        state.info["first_obs"] = state.obs
        return state

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(rng, state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info["first_pipeline_state"], state.pipeline_state
        )
        obs = where_done(state.info["first_obs"], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


@struct.dataclass
class EvalMetrics:
    """Dataclass holding evaluation metrics for Brax.

    Attributes:
        episode_metrics: Aggregated episode metrics since the beginning of the
          episode.
        active_episodes: Boolean vector tracking which episodes are not done yet.
        episode_steps: Integer vector tracking the number of steps in the episode.
    """

    episode_metrics: Dict[str, jax.Array]
    active_episodes: jax.Array
    episode_steps: jax.Array


class EvalWrapper(Wrapper):
    """Brax env with eval metrics."""

    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics["reward"] = reset_state.reward
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(jp.zeros_like, reset_state.metrics),
            active_episodes=jp.ones_like(reset_state.reward),
            episode_steps=jp.zeros_like(reset_state.reward),
        )
        reset_state.info["eval_metrics"] = eval_metrics
        return reset_state

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        state_metrics = state.info["eval_metrics"]
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(f"Incorrect type for state_metrics: {type(state_metrics)}")
        del state.info["eval_metrics"]
        nstate = self.env.step(rng, state, action)
        nstate.metrics["reward"] = nstate.reward
        episode_steps = jp.where(
            state_metrics.active_episodes,
            nstate.info["steps"],
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info["eval_metrics"] = eval_metrics
        return nstate


class DomainRandomizationVmapWrapper(Wrapper):
    """Wrapper for domain randomization."""

    def __init__(
        self,
        env: Env,
        randomization_fn: Callable[[System], Tuple[System, System]],
    ):
        super().__init__(env)
        self._sys_v, self._in_axes = randomization_fn(self.sys)

    def _env_fn(self, sys: System) -> Env:
        env = self.env
        env.unwrapped.sys = sys
        return env

    def reset(self, rng: jax.Array) -> State:
        def reset(sys, rng):
            env = self._env_fn(sys=sys)
            return env.reset(rng)

        state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
        return state

    def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
        def step(sys, r, s, a):
            env = self._env_fn(sys=sys)
            return env.step(r, s, a)

        res = jax.vmap(step, in_axes=[self._in_axes, 0, 0, 0])(
            self._sys_v, rng, state, action
        )
        return res


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


class PixelWrapper(PipelineEnv):
    def __init__(
        self,
        env: Env,
        hw: int,
        frame_stack: int,
        return_float32: bool,
        cache_objects: bool,
        n_envs: int,
    ):
        super().__init__(sys=env.sys, backend=env.backend)
        self.env = env
        self.seed = None  # TODO: does the env hold a seed?
        self.hw = hw
        self.frame_stack = frame_stack
        self.return_float32 = return_float32
        self.cache_objects = cache_objects

        if cache_objects:
            self.cached_objects, self.vmappable_objects = build_objects_for_cache(
                self.env.sys, n_envs
            )
        else:
            self.cached_objects, self.vmappable_objects = None, None

        # print(f"cache:  {self.cached_objects}")
        # print(
        #    f"test: {self.cached_objects[0].instance.model.specular_map.shape} // {type(self.cached_objects[0].instance.model.specular_map)}"
        # )

        # print(
        #    f"test 2: {self.cached_objects[0].rot}  // {type(self.cached_objects[0].rot)}"
        # )
        # qqq
        # The VmapWrapper is already handling this. Will likely need to remove
        # self._reset_fn = jax.vmap(env.reset)
        # self._step_fn = jax.vmap(env.step)

        # TODO: implement mesh caching. I'm pretty sure we can literally just compute
        # the mesh attributes a single time and cache them. We don't even need a pipeline_state
        # from the environment to do this. This is the most expensive part. Then we can
        # just pass the cached mesh to the rendering function!!!

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def observation_size(self) -> Tuple[int]:
        return (self.hw, self.hw, 3 * self.frame_stack)

    def reset(self, rng: jp.ndarray) -> PixelState:
        raw_state = self.env.reset(rng)
        # before = self.env.sys.mj_model.geom_pos
        # self.env.step(
        #    jax.random.split(rng, raw_state.obs.shape[0]),
        #    raw_state,
        #    jax.random.uniform(rng, (raw_state.obs.shape[0], self.env.action_size)),
        # )
        # after = self.env.sys.mj_model.geom_pos
        # print(f"delta: {(before - after).sum()}")
        # qqq
        if self.cache_objects:
            if raw_state.pipeline_state.qpos.ndim > 1:
                frames = ru.render_pixels_with_cached_objs(
                    raw_state.pipeline_state,
                    self.cached_objects,
                    self.vmappable_objects,
                    self.hw,
                )
            else:
                frames = ru.render_pixels_with_cached_objs(
                    jax.tree_map(lambda x: jp.expand_dims(x, 0), raw_state.pipeline_state),
                    self.cached_objects,
                    self.vmappable_objects,
                    self.hw,
                )
                frames = frames[0]

        else:
            frames = ru.render_pixels(self.env.sys, raw_state.pipeline_state, self.hw)

        if not self.return_float32:
            frames = (frames * 255).astype(jp.uint8)

        # TODO: add frame stacking here
        return PixelState(
            raw_state.pipeline_state,
            raw_state.obs,
            frames,
            raw_state.reward,
            raw_state.done,
            #jax.random.split(rng, raw_state.obs.shape[0]),
            raw_state.metrics,
            raw_state.info,
        )

    def step(
        self, rng: jp.ndarray, states: jp.ndarray, actions: jp.ndarray
    ) -> PixelState:
        raw_state = self.env.step(rng, states, actions)

        if self.cache_objects:
            if raw_state.pipeline_state.qpos.ndim > 1:
                frames = ru.render_pixels_with_cached_objs(
                    raw_state.pipeline_state,
                    self.cached_objects,
                    self.vmappable_objects,
                    self.hw,
                )
            else:
                frames = ru.render_pixels_with_cached_objs(
                    jax.tree_map(lambda x: jp.expand_dims(x, 0), raw_state.pipeline_state),
                    self.cached_objects,
                    self.vmappable_objects,
                    self.hw,
                )
                frames = frames[0]

        else:
            frames = ru.render_pixels(self.env.sys, raw_state.pipeline_state, self.hw)

        if not self.return_float32:
            frames = (frames * 255).astype(jp.uint8)

        # TODO: add frame stacking here
        return PixelState(
            raw_state.pipeline_state,
            raw_state.obs,
            frames,
            raw_state.reward,
            raw_state.done,
            #rng,
            raw_state.metrics,
            raw_state.info,
        )
