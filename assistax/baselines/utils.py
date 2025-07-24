import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import assistax
from assistax.wrappers.baselines import get_space_dim, LogEnvState, LogWrapper, LogCrossplayWrapper
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper, LoadEvalAgentWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Optional
import os 
import functools

# Tree Utilities 

def _tree_take(pytree, indices, axis=None):
    """
    Take elements from each leaf of a pytree along a specified axis.
    
    Args:
        pytree: JAX pytree (nested structure of arrays)
        indices: Indices to take from each array
        axis: Axis along which to take indices (None for flat indexing)
        
    Returns:
        Pytree with same structure but indexed arrays
    """
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)


def _tree_shape(pytree):
    """
    Get the shape of each leaf in a pytree.
    
    Args:
        pytree: JAX pytree (nested structure of arrays)
        
    Returns:
        Pytree with same structure but shapes instead of arrays
    """
    return jax.tree.map(lambda x: x.shape, pytree)


def _unstack_tree(pytree):
    """
    Unstack a pytree along the first axis, yielding a list of pytrees.
    
    Converts a pytree where each leaf has shape (N, ...) into a list of N pytrees
    where each leaf has shape (...).
    
    Args:
        pytree: JAX pytree with arrays of shape (N, ...)
        
    Returns:
        List of N pytrees, each with arrays of shape (...)
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]


def _stack_tree(pytree_list, axis=0):
    """
    Stack a list of pytrees along a specified axis.
    
    Args:
        pytree_list: List of pytrees with compatible structures
        axis: Axis along which to stack
        
    Returns:
        Single pytree with stacked arrays
    """
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )


def _concat_tree(pytree_list, axis=0):
    """
    Concatenate a list of pytrees along a specified axis.
    
    Args:
        pytree_list: List of pytrees with compatible structures
        axis: Axis along which to concatenate
        
    Returns:
        Single pytree with concatenated arrays
    """
    return jax.tree.map(
        lambda *leaf: jnp.concat(leaf, axis=axis),
        *pytree_list
    )


def _tree_split(pytree, n, axis=0):
    """
    Split a pytree into n parts along a specified axis.
    
    Args:
        pytree: JAX pytree to split
        n: Number of parts to split into
        axis: Axis along which to split
        
    Returns:
        List of n pytrees
    """
    leaves, treedef = jax.tree.flatten(pytree)
    split_leaves = zip(
        *jax.tree.map(lambda x: jnp.array_split(x, n, axis), leaves)
    )
    return [
        jax.tree.unflatten(treedef, leaves)
        for leaves in split_leaves
    ]

# ================================ EPISODE PROCESSING UTILITIES ================================

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    """
    Extract a complete episode from evaluation data.
    
    Takes the pipeline states for a specific evaluation run and returns only
    the timesteps before the episode ended (excluding done states).
    
    Args:
        pipeline_states: Environment pipeline states for all timesteps
        dones: Boolean array indicating episode termination
        time_idx: Time axis index (default: -1)
        eval_idx: Which evaluation episode to extract (default: 0)
        
    Returns:
        List of pipeline states for the complete episode
    """
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not (done)
    ]


def _compute_episode_returns(eval_info, time_axis=-2):
    """
    Compute undiscounted episode returns from evaluation information.
    
    Handles episode boundaries correctly by resetting cumulative rewards
    when episodes end and start new ones.
    
    Args:
        eval_info: Evaluation information containing rewards and done flags
        time_axis: Axis representing time dimension (default: -2)
        
    Returns:
        Undiscounted returns for each episode
    """
    done_arr = eval_info.done["__all__"]
    
    # Create mask for episode boundaries
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    
    # Sum rewards within episodes only
    undiscounted_returns = jax.tree.map(
        lambda r: (r * (1 - episode_done)).sum(axis=time_axis),
        eval_info.reward
    )
    return undiscounted_returns

def _compute_episode_returns_sweep(eval_info, common_reward=False, time_axis=-2):
    """
    Compute undiscounted episode returns from evaluation information.
    
    Handles episode boundaries correctly by resetting cumulative rewards
    when episodes end and start new ones. Also handles both individual agent
    rewards and common team rewards.
    
    Args:
        eval_info: Evaluation information containing rewards and done flags
        common_reward: Whether agents share a common reward signal
        time_axis: Axis representing time dimension (default: -2)
        
    Returns:
        Undiscounted returns for each agent/team
    """
    done_arr = eval_info.done["__all__"]
    
    # Create mask for episode boundaries
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    
    # Sum rewards within episodes only
    undiscounted_returns = jax.tree.map(
        lambda r: (r * (1 - episode_done)).sum(axis=time_axis),
        eval_info.reward
    )
    
    # Add aggregate return if not present
    if "__all__" not in undiscounted_returns:
        undiscounted_returns.update({
            "__all__": (sum(undiscounted_returns.values())
                        / (len(undiscounted_returns) if common_reward else 1))
        })
    
    return undiscounted_returns
