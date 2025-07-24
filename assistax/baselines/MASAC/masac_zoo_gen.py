"""
MASAC Zoo Generation Script

This module trains Multi-Agent Soft Actor-Critic (MASAC) agents and saves them to a "zoo" 
for later use in mixed training scenarios. The zoo is a collection of trained agent policies
that can be loaded and used as training partners, enabling diverse multi-agent interactions
and robust policy development.

Usage:
    python masac_zoo_gen.py [hydra options] ZOO_PATH=path/to/zoo
    
The script will train agents and automatically save them to the zoo with proper
organization by agent ID and training seed.
"""

import jax
import jax.numpy as jnp
import hydra
import assistax
from tqdm import tqdm
from omegaconf import OmegaConf
from assistax.wrappers.aht import ZooManager
from typing import Dict, Any
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns,
    _tree_shape, _stack_tree, _concat_tree, _tree_split
    )


# ================================ TREE MANIPULATION UTILITIES ================================

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
        if not done
    ]


def _compute_episode_returns(eval_info, common_reward=False, time_axis=-2):
    """
    Compute undiscounted episode returns from evaluation information.
    
    Handles episode boundaries correctly by resetting cumulative rewards
    when episodes end and start new ones. Supports both individual agent
    rewards and common team rewards.
    
    Args:
        eval_info: Evaluation information containing rewards and done flags
        common_reward: Whether to treat rewards as shared across agents
        time_axis: Axis representing time dimension (default: -2)
        
    Returns:
        Dictionary of undiscounted returns per agent (and "__all__" for total)
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
    
    # Add total reward if not present
    if "__all__" not in undiscounted_returns:
        undiscounted_returns.update({
            "__all__": (sum(undiscounted_returns.values())
                       / (len(undiscounted_returns) if common_reward else 1))
        })
    
    return undiscounted_returns


# ================================ HYPERPARAMETER SWEEP UTILITIES ================================

def _generate_sweep_axes(rng, config):
    """
    Generate hyperparameter configurations for sweep.
    
    Creates random samples for each hyperparameter marked for sweeping in the
    configuration. Uses log-uniform sampling for learning rates and tau to
    cover multiple orders of magnitude effectively.
    
    Note: This function is included for compatibility but not used in zoo generation.
    
    Args:
        rng: Random number generator key
        config: Configuration dictionary containing sweep specifications
        
    Returns:
        Dictionary containing sampled values and vmap axes for each hyperparameter
    """
    p_lr_rng, q_lr_rng, alpha_lr_rng, tau_rng = jax.random.split(rng, 4)
    sweep_config = config["SWEEP"]
    
    # ===== POLICY LEARNING RATE SWEEP =====
    if sweep_config.get("p_lr", False):
        p_lrs = 10**jax.random.uniform(
            p_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["p_lr"]["min"],
            maxval=sweep_config["p_lr"]["max"],
        )
        p_lr_axis = 0
    else:
        p_lrs = config["POLICY_LR"]
        p_lr_axis = None

    # ===== Q-NETWORK LEARNING RATE SWEEP =====
    if sweep_config.get("q_lr", False):
        q_lrs = 10**jax.random.uniform(
            q_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["q_lr"]["min"],
            maxval=sweep_config["q_lr"]["max"],
        )
        q_lr_axis = 0
    else:
        q_lrs = config["Q_LR"]
        q_lr_axis = None

    # ===== TEMPERATURE LEARNING RATE SWEEP =====
    if sweep_config.get("alpha_lr", False):
        alpha_lrs = 10**jax.random.uniform(
            alpha_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["alpha_lr"]["min"],
            maxval=sweep_config["alpha_lr"]["max"],
        )
        alpha_lr_axis = 0
    else:
        alpha_lrs = config["ALPHA_LR"]
        alpha_lr_axis = None

    # ===== TAU (SOFT UPDATE) SWEEP =====
    if sweep_config.get("tau", False):
        taus = 10**jax.random.uniform(
            tau_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["tau"]["min"],
            maxval=sweep_config["tau"]["max"],
        )
        tau_axis = 0
    else:
        taus = config["TAU"]
        tau_axis = None

    return {
        "p_lr": {"val": p_lrs, "axis": p_lr_axis},
        "q_lr": {"val": q_lrs, "axis": q_lr_axis},
        "alpha_lr": {"val": alpha_lrs, "axis": alpha_lr_axis},
        "tau": {"val": taus, "axis": tau_axis},
    }


# ================================ MAIN ZOO GENERATION ORCHESTRATION ================================

@hydra.main(version_base=None, config_path="config", config_name="masac_zoo_gen")
def main(config):
    """
    Main orchestration function for MASAC zoo generation.
    
    This function:
    1. Trains MASAC agents across multiple random seeds
    2. Extracts individual agent parameters from trained models
    3. Saves each agent to the zoo with proper metadata and organization
    4. Creates a diverse collection of trained agents for future use
    
    The zoo enables training against diverse partners, curriculum learning,
    and robust policy evaluation in multi-agent environments.
    
    Args:
        config: Hydra configuration object containing training and zoo parameters
    """
    config = OmegaConf.to_container(config, resolve=True)
    
    print(f"Starting MASAC zoo generation")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Number of seeds: {config['NUM_SEEDS']}")
    print(f"Zoo path: {config['ZOO_PATH']}")
    print(f"Total timesteps per agent: {config['TOTAL_TIMESTEPS']}")
    
    # ===== IMPORT ALGORITHM COMPONENTS =====
    from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
    
    # ===== ENVIRONMENT SETUP =====
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    print(f"Environment agents: {env.agents}")
    print(f"Total agents to train: {len(env.agents) * config['NUM_SEEDS']}")
    
    # ===== RANDOM NUMBER GENERATOR SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    # ===== TRAINING EXECUTION =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        print("Compiling training function...")
        train_jit = jax.jit(
            make_train(config, save_train_state=False),  # Don't save training states for zoo generation
            device=jax.devices()[config["DEVICE"]]
        )
        
        # Train agents across all seeds using vmap for efficiency
        out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
            train_rngs,
            config["POLICY_LR"],
            config["Q_LR"],
            config["ALPHA_LR"],
            config["TAU"],
        )
        
        # Extract final trained parameters
        final_train_state = out["runner_state"].train_states.actor.params
        print(f"Training completed! Final parameters shape: {jax.tree.leaves(_tree_shape(final_train_state))[0]}")
        
        # ===== ZOO MANAGEMENT SETUP =====
        print("Setting up zoo management...")
        zoo = ZooManager(config["ZOO_PATH"])
        
        # ===== SAVE AGENTS TO ZOO =====
        print("Saving trained agents to zoo...")
        total_agents_saved = 0
        
        # Iterate through each agent type in the environment
        for agent_idx, agent_id in enumerate(env.agents):
            print(f"\nProcessing agent: {agent_id} (index {agent_idx})")
            
            # Save agent from each training seed
            for seed_idx in range(config["NUM_SEEDS"]):
                print(f"  Saving agent from seed {seed_idx}...")
                
                # Extract parameters for this specific agent and seed
                # First extract seed, then extract agent from the seed's parameters
                agent_params = _tree_take(  # Extract agent parameters
                    _tree_take(  # Extract seed parameters
                        final_train_state,
                        seed_idx,
                        axis=0,  # Seed axis
                    ),
                    agent_idx,
                    axis=0,  # Agent axis
                )
                
                # Save agent to zoo with metadata
                zoo.save_agent(
                    config=config,
                    param_dict=agent_params,
                    scenario_agent_id=agent_id
                )
                
                total_agents_saved += 1
                print(f"    ✓ Saved {agent_id}_seed{seed_idx} to zoo")
        
        # ===== DISPLAY COMPLETION SUMMARY =====
        print("\n" + "="*60)
        print("ZOO GENERATION COMPLETED")
        print("="*60)
        print(f"Total agents saved to zoo: {total_agents_saved}")
        print(f"Agents per type: {config['NUM_SEEDS']}")
        print(f"Agent types: {', '.join(env.agents)}")
        print(f"Zoo location: {config['ZOO_PATH']}")
        print(f"")
        print("The zoo now contains trained agents that can be used for:")
        print("- Mixed training scenarios")
        print("- Curriculum learning")
        print("- Robust policy evaluation")
        print("- Population-based training")
        print("="*60)
        
        print("Zoo generation completed successfully!")


if __name__ == "__main__":
    main()

# import os
# import time
# from tqdm import tqdm
# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# from flax.linen.initializers import constant, orthogonal
# from flax.training.train_state import TrainState
# from flax.traverse_util import flatten_dict
# import safetensors.flax
# import optax
# import distrax
# import assistax
# from assistax.wrappers.baselines import get_space_dim, LogEnvState
# from assistax.wrappers.baselines import LogWrapper
# from assistax.wrappers.aht import ZooManager
# import hydra
# from omegaconf import OmegaConf
# from typing import Sequence, NamedTuple, Any, Dict
# from base64 import urlsafe_b64encode

# def _tree_take(pytree, indices, axis=None):
#     return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

# def _tree_shape(pytree):
#     return jax.tree.map(lambda x: x.shape, pytree)

# def _unstack_tree(pytree):
#     leaves, treedef = jax.tree_util.tree_flatten(pytree)
#     unstacked_leaves = zip(*leaves)
#     return [jax.tree_util.tree_unflatten(treedef, leaves)
#             for leaves in unstacked_leaves]

# def _stack_tree(pytree_list, axis=0):
#     return jax.tree.map(
#         lambda *leaf: jnp.stack(leaf, axis=axis),
#         *pytree_list
#     )

# def _concat_tree(pytree_list, axis=0):
#     return jax.tree.map(
#         lambda *leaf: jnp.concat(leaf, axis=axis),
#         *pytree_list
#     )

# def _tree_split(pytree, n, axis=0):
#     leaves, treedef = jax.tree.flatten(pytree)
#     split_leaves = zip(
#         *jax.tree.map(lambda x: jnp.array_split(x,n,axis), leaves)
#     )
#     return [
#         jax.tree.unflatten(treedef, leaves)
#         for leaves in split_leaves
#     ]

# def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
#     episodes = _tree_take(pipeline_states, eval_idx, axis=1)
#     dones = dones.take(eval_idx, axis=1)
#     return [
#         state
#         for state, done in zip(_unstack_tree(episodes), dones)
#         if not (done)
#     ]

# def _compute_episode_returns(eval_info, common_reward=False, time_axis=-2):
#     done_arr = eval_info.done["__all__"]
#     first_timestep = [slice(None) for _ in range(done_arr.ndim)]
#     first_timestep[time_axis] = 0
#     episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
#     episode_done = jnp.roll(episode_done, 1, axis=time_axis)
#     episode_done = episode_done.at[tuple(first_timestep)].set(False)
#     undiscounted_returns = jax.tree.map(
#         lambda r: (r*(1-episode_done)).sum(axis=time_axis),
#         eval_info.reward
#     )
#     if "__all__" not in undiscounted_returns:
#         undiscounted_returns.update({
#             "__all__": (sum(undiscounted_returns.values())
#                         /(len(undiscounted_returns) if common_reward else 1))
#         })
#     return undiscounted_returns

# def _generate_sweep_axes(rng, config):
#     p_lr_rng, q_lr_rng, alpha_lr_rng, tau_rng = jax.random.split(rng, 4)
#     sweep_config = config["SWEEP"]
#     if sweep_config.get("p_lr", False):
#         p_lrs = 10**jax.random.uniform(
#             p_lr_rng,
#             shape=(sweep_config["num_configs"],),
#             minval=sweep_config["p_lr"]["min"],
#             maxval=sweep_config["p_lr"]["max"],
#         )
#         p_lr_axis = 0
#     else:
#         p_lrs = config["POLICY_LR"]
#         p_lr_axis = None

#     if sweep_config.get("q_lr", False):
#         q_lrs = 10**jax.random.uniform(
#             q_lr_rng,
#             shape=(sweep_config["num_configs"],),
#             minval=sweep_config["q_lr"]["min"],
#             maxval=sweep_config["q_lr"]["max"],
#         )
#         q_lr_axis = 0
#     else:
#         q_lrs = config["Q_LR"]
#         q_lr_axis = None

#     if sweep_config.get("alpha_lr", False):
#         alpha_lrs = 10**jax.random.uniform(
#             alpha_lr_rng,
#             shape=(sweep_config["num_configs"],),
#             minval=sweep_config["alpha_lr"]["min"],
#             maxval=sweep_config["alpha_lr"]["max"],
#         )
#         alpha_lr_axis = 0
#     else:
#         alpha_lrs = config["ALPHA_LR"]
#         alpha_lr_axis = None

#     if sweep_config.get("tau", False):
#         taus = 10**jax.random.uniform(
#             tau_rng,
#             shape=(sweep_config["num_configs"],),
#             minval=sweep_config["tau"]["min"],
#             maxval=sweep_config["tau"]["max"],
#         )
#         tau_axis = 0
#     else:
#         taus = config["TAU"]
#         tau_axis = None


#     return {
#         "p_lr": {"val": p_lrs, "axis": p_lr_axis},
#         "q_lr": {"val": q_lrs, "axis": q_lr_axis},
#         "alpha_lr": {"val": alpha_lrs, "axis": alpha_lr_axis},
#         "tau": {"val": taus, "axis":tau_axis},
#     }

# @hydra.main(version_base=None, config_path="config", config_name="masac_zoo_gen")
# def main(config):

#     config = OmegaConf.to_container(config, resolve=True)
#     from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
#     rng = jax.random.PRNGKey(config["SEED"])
#     train_rng, eval_rng = jax.random.split(rng)
#     train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
#     with jax.disable_jit(config["DISABLE_JIT"]):
#         env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#         train_jit = jax.jit(
#             make_train(config, save_train_state=False),
#             device=jax.devices()[config["DEVICE"]]
#         )
#         out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
#             train_rngs,
#             config["POLICY_LR"],
#             config["Q_LR"],
#             config["ALPHA_LR"],
#             config["TAU"],
#         )
#         final_train_state = out["runner_state"].train_states.actor.params
#         zoo = ZooManager(config["ZOO_PATH"])
#         for agent_idx, agent_id in enumerate(env.agents):
#             for seed_idx in range(config["NUM_SEEDS"]):
#                 zoo.save_agent(
#                     config=config,
#                     param_dict=_tree_take( # agent
#                         _tree_take( # seed
#                             final_train_state,
#                             seed_idx,
#                             axis=0,
#                         ),
#                         agent_idx,
#                         axis=0,
#                     ),
#                     scenario_agent_id=agent_id
#                 )

# if __name__ == "__main__":
#     main()
