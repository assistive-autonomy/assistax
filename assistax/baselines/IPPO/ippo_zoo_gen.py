"""
IPPO Agent Zoo Generation for Population-Based Multi-Agent Research

This module generates diverse populations of trained IPPO agents and saves them to a centralized
"zoo" for use in future experiments. The zoo serves as a repository of agent policies that can
be loaded as teammates, opponents, or evaluation partners in multi-agent scenarios.

Key Features:
- Population-based training with multiple random seeds for diversity
- Automatic zoo management and agent storage using ZooManager
- Individual agent extraction and saving from multi-agent training
- Support for all four IPPO variants (FF/RNN x NPS/PS)
- Dynamic algorithm selection based on network architecture
- Systematic agent organization by scenario and agent ID


Usage:
    python ippo_zoo_gen.py [hydra options]
    
The script will train agents and automatically save each individual agent
(from each seed) to the specified zoo directory for later retrieval.
"""

import os
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
import safetensors.flax
import optax
import distrax
import assistax
from assistax.wrappers.baselines import  get_space_dim, LogEnvState, LogWrapper
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper
from assistax.wrappers.aht import ZooManager # check whether this should be  
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns,
    _tree_shape, _stack_tree, _concat_tree, _tree_split
    )


# ================================ TREE MANIPULATION UTILITIES ================================

def _tree_take(pytree, indices, axis=None):
    """
    Take elements from each leaf of a pytree along a specified axis.
    
    Essential for extracting individual agent parameters from multi-agent
    training states when saving to the zoo.
    
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
    
    Useful for debugging and understanding the structure of training states
    when extracting individual agents.
    
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
    where each leaf has shape (...). Useful for separating agents or seeds.
    
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


# ================================ MAIN ZOO GENERATION FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="ippo_zoo_gen")
def main(config):
    """
    Main function for generating a diverse population of IPPO agents and saving them to a zoo.
    
    This function:
    1. Dynamically imports the correct IPPO variant based on config
    2. Trains multiple agents across different random seeds for diversity
    3. Extracts individual agent parameters from multi-agent training
    4. Saves each agent to the zoo with unique identifiers
    5. Organizes agents by scenario and agent ID for easy retrieval
    
    The resulting zoo can be used for:
    - Loading trained teammates in future experiments
    - Creating diverse opponent populations
    - Evaluating agent performance against consistent baselines
    - Supporting population-based research studies
    
    Args:
        config: Hydra configuration object containing training and zoo parameters
    """
    config = OmegaConf.to_container(config, resolve=True)

    # ===== DYNAMIC ALGORITHM SELECTION =====
    # Import the appropriate IPPO variant based on network architecture configuration
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps import make_train
            print("Using: Feedforward Networks with No Parameter Sharing")
        case (False, True):
            from ippo_ff_ps import make_train
            print("Using: Feedforward Networks with Parameter Sharing")
        case (True, False):
            from ippo_rnn_nps import make_train
            print("Using: Recurrent Networks with No Parameter Sharing")
        case (True, True):
            from ippo_rnn_ps import make_train
            print("Using: Recurrent Networks with Parameter Sharing")

    # ===== TRAINING SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    print(f"Generating agent zoo with {config['TOTAL_TIMESTEPS']} timesteps")
    print(f"Num environments: {config['NUM_ENVS']}")
    print(f"Num seeds (for diversity): {config['NUM_SEEDS']}")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Zoo path: {config['ZOO_PATH']}")
    
    # ===== POPULATION TRAINING =====
    print("Starting population training for zoo generation...")
    with jax.disable_jit(config["DISABLE_JIT"]):
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        
        # Create training function (no need to save training states for zoo generation)
        train_jit = jax.jit(
            make_train(config, save_train_state=False),
            device=jax.devices()[config["DEVICE"]]
        )
        
        # Train across multiple seeds to create diverse agent population
        print("Training diverse agent population...")
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )
        
        # Extract final trained parameters
        final_train_state = out["runner_state"].train_state.params
        print(f"Training completed. Parameter shape: {_tree_shape(final_train_state)}")

        # ===== ZOO MANAGEMENT AND AGENT SAVING =====
        print("Initializing zoo and saving agents...")
        zoo = ZooManager(config["ZOO_PATH"])
        
        total_agents_saved = 0
        
        # Save each agent from each seed to create diverse population
        for agent_idx, agent_id in enumerate(env.agents):
            print(f"Processing agent: {agent_id} (index: {agent_idx})")
            
            for seed_idx in range(config["NUM_SEEDS"]):
                # Extract individual agent parameters from multi-agent training state
                # First extract seed, then extract agent from that seed
                agent_params = _tree_take(  # Extract specific agent
                    _tree_take(  # Extract specific seed
                        final_train_state,
                        seed_idx,
                        axis=0,  # Seed dimension
                    ),
                    agent_idx,
                    axis=0,  # Agent dimension (for non-parameter sharing)
                )
                
                # Save agent to zoo with configuration metadata
                zoo.save_agent(
                    config=config,
                    param_dict=agent_params,
                    scenario_agent_id=agent_id
                )
                
                total_agents_saved += 1
                
            print(f"  Saved {config['NUM_SEEDS']} variants of {agent_id}")
        
        # ===== COMPLETION SUMMARY =====
        print(f"\nZoo generation completed successfully!")
        print(f"Total agents saved: {total_agents_saved}")


if __name__ == "__main__":
    main()



