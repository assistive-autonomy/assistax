"""Shared helpers for the regression tests: cached env rollouts and MuJoCo-name lookups."""
import functools
from typing import Sequence

import jax
import mujoco
import numpy as np
from brax.envs.base import PipelineEnv, State

from assistax import envs

PANDA_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
HUMAN_ROBOT_ENVS = ("scratchitch", "bedbathing", "armmanipulation", "feeding", "teethbrushing")


@functools.lru_cache(maxsize=None)
def rollout(env_name: str, n_steps: int = 5, seed: int = 0) -> tuple[PipelineEnv, tuple[State, ...]]:
    """Return the raw (unwrapped) env and the states of a jitted reset plus ``n_steps`` steps under
    uniform random actions in [-1, 1]. Cached per argument tuple so test modules share compiles."""
    env = envs._envs[env_name]()
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    key_reset, key = jax.random.split(jax.random.PRNGKey(seed))
    states = [reset(key_reset)]
    for _ in range(n_steps):
        key, key_act, key_step = jax.random.split(key, 3)
        action = jax.random.uniform(key_act, (env.action_size,), minval=-1.0, maxval=1.0)
        states.append(step(key_step, states[-1], action))
    return env, tuple(states)


def human_hinge_names(m: mujoco.MjModel) -> list[str]:
    """Humanoid hinge joints, i.e. every hinge that is not a Panda joint, in joint-id order."""
    return [
        m.joint(j).name
        for j in range(m.njnt)
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name not in PANDA_JOINTS
    ]


def joint_qposadr(m: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    """``qpos`` address of each named joint."""
    return np.array([m.joint(n).qposadr[0] for n in names])


def joint_dofadr(m: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    """``qvel``/``qd`` address of each named joint."""
    return np.array([m.joint(n).dofadr[0] for n in names])


def joint_range(m: mujoco.MjModel, names: Sequence[str]) -> np.ndarray:
    """``jnt_range`` rows of the named joints, shape ``(len(names), 2)``."""
    return np.array([m.jnt_range[m.joint(n).id] for n in names])


def obs_dicts(env: PipelineEnv, env_name: str, state: State) -> tuple[dict, dict]:
    """The env's robot and human observation dicts for ``state`` (scratchitch also needs ``state.info``)."""
    args = (state.pipeline_state, state.info) if env_name == "scratchitch" else (state.pipeline_state,)
    return env._get_robo_obs(*args), env._get_human_obs(*args)
