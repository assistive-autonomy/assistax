"""
MASAC Episode Rendering Script

This module creates interactive HTML visualizations of trained Multi-Agent Soft Actor-Critic (MASAC) 
agents in action. It loads trained models for multiple agents, evaluates them in the environment,
and generates interactive HTML renderings of representative episodes for analysis and demonstration.

Key Features:
- Loads trained models for multiple agents (e.g., "human" and "robot")
- Evaluates agents together in multi-agent episodes
- Automatically selects representative episodes (worst, median, best performance)
- Generates interactive HTML visualizations using Brax
- Provides performance-based episode naming for easy identification
- Supports any Brax-compatible environment for rendering

Rendering Process:
1. Load trained parameters for each agent type
2. Combine agent parameters into multi-agent evaluation setup
3. Run evaluation episodes with environment state logging
4. Rank episodes by total reward/performance
5. Extract worst, median, and best performing episodes
6. Generate interactive HTML files for each selected episode

Output Files:
- final_worst_r{reward}.html: Visualization of worst performing episode
- final_median_r{reward}.html: Visualization of median performing episode  
- final_best_r{reward}.html: Visualization of best performing episode

The HTML files contain interactive 3D visualizations that can be opened in any web browser,
showing the complete episode with physics simulation, agent movements, and environment dynamics.

Usage:
    python masac_render.py [hydra options] eval.path.human=path/to/human.safetensors eval.path.robot=path/to/robot.safetensors
    
Requires:
- Trained model parameters saved in SafeTensors format
- Brax-compatible environment for physics rendering
- Separate model files for each agent type to be visualized
"""

import jax
import jax.numpy as jnp
import hydra
import safetensors.flax
from flax.traverse_util import flatten_dict, unflatten_dict
from flax import struct
from omegaconf import OmegaConf
from typing import Dict, Any, Callable
from brax.io import html


# ================================ DATA STRUCTURES ================================

@struct.dataclass
class EvalNetworkState:
    """
    Lightweight network state container for evaluation and rendering.
    
    Contains only the essential components needed for policy evaluation:
    the network's apply function and trained parameters for all agents.
    
    Attributes:
        apply_fn: Neural network forward pass function
        params: Trained network parameters for all agents
    """
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict


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


# ================================ EPISODE PROCESSING UTILITIES ================================

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    """
    Extract a complete episode from evaluation data for rendering.
    
    Takes the pipeline states for a specific evaluation run and returns only
    the timesteps before the episode ended, which is needed for proper
    visualization of the complete episode trajectory.
    
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


def _select_representative_episodes(eval_results):
    """
    Select worst, median, and best performing episodes for rendering.
    
    Analyzes episode returns and selects representative episodes that
    showcase the range of agent performance for visualization purposes.
    
    Args:
        eval_results: Evaluation results containing episode data
        
    Returns:
        Tuple of (worst_idx, median_idx, best_idx, episode_returns)
    """
    # Compute episode boundaries and returns
    first_episode_done = jnp.cumsum(eval_results.done["__all__"], axis=0, dtype=bool)
    first_episode_rewards = eval_results.reward["__all__"] * (1 - first_episode_done)
    first_episode_returns = first_episode_rewards.sum(axis=0)
    
    # Sort episodes by performance
    episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
    
    # Select representative episodes
    worst_idx = episode_argsort.take(0, axis=-1)          # Worst performing
    best_idx = episode_argsort.take(-1, axis=-1)          # Best performing  
    median_idx = episode_argsort.take(                    # Median performing
        episode_argsort.shape[-1] // 2, axis=-1
    )
    
    return worst_idx, median_idx, best_idx, first_episode_returns


def _load_agent_parameters(config):
    """
    Load trained parameters for all agents from SafeTensors files.
    
    Loads and unflatterns parameters for each agent type specified in the
    configuration. Supports loading different models for different agent types.
    
    Args:
        config: Configuration dictionary containing model paths
        
    Returns:
        Dictionary mapping agent names to their loaded parameters
    """
    agent_params = {}
    
    for agent_name, model_path in config["eval"]["path"].items():
        print(f"Loading {agent_name} model from: {model_path}")
        
        params = unflatten_dict(
            safetensors.flax.load_file(model_path), 
            sep='/'
        )
        agent_params[agent_name] = params
        
        print(f"  ✓ Loaded {agent_name} parameters")
    
    return agent_params


# ================================ MAIN RENDERING ORCHESTRATION ================================

@hydra.main(version_base=None, config_path="config", config_name="masac_mabrax")
def main(config):
    """
    Main orchestration function for MASAC episode rendering.
    
    This function:
    1. Loads trained models for all specified agents
    2. Sets up multi-agent evaluation with environment state logging
    3. Runs evaluation episodes to collect trajectory data
    4. Selects representative episodes (worst, median, best)
    5. Generates interactive HTML visualizations using Brax
    
    The resulting HTML files provide interactive 3D visualizations that can be
    used for analysis, demonstration, and understanding agent behaviors.
    
    Args:
        config: Hydra configuration object containing model paths and parameters
    """
    config = OmegaConf.to_container(config, resolve=True)
    
    print(f"Starting MASAC episode rendering")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Number of evaluation episodes: {config['NUM_EVAL_EPISODES']}")
    
    # ===== IMPORT ALGORITHM COMPONENTS =====
    from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
    from masac_ff_nps import MultiSACActor as NetworkArch
    
    # ===== RANDOM NUMBER GENERATOR SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    
    # ===== MODEL LOADING =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        print("\nLoading trained agent models...")
        agent_params = _load_agent_parameters(config)
        
        if len(agent_params) == 0:
            raise ValueError("No agent models specified in config. Please provide model paths.")
        
        print(f"Loaded models for agents: {list(agent_params.keys())}")
        
        # ===== EVALUATION SETUP =====
        print("Setting up evaluation environment...")
        eval_env, run_eval = make_evaluation(config)
        
        # Configure evaluation to log environment state for rendering
        eval_log_config = EvalInfoLogConfig(
            env_state=True,      # REQUIRED: Log environment state for rendering
            done=True,           # Log episode termination
            action=False,        # Don't log actions (saves memory)
            reward=True,         # Log rewards for episode selection
            log_prob=False,      # Don't log action probabilities (saves memory)
            obs=False,           # Don't log observations (saves memory)
            info=False,          # Don't log additional info (saves memory)
            avail_actions=False, # Don't log available actions (saves memory)
        )
        
        # JIT compile evaluation function
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        
        # Initialize network architecture
        network = NetworkArch(config=config)
        
        # ===== PREPARE MULTI-AGENT PARAMETERS =====
        print("Preparing multi-agent evaluation setup...")
        
        # Extract first seed/checkpoint for each agent (for rendering)
        agent_params_for_eval = {}
        for agent_name, params in agent_params.items():
            agent_params_for_eval[agent_name] = _tree_take(
                params,
                0,  # Take first seed/checkpoint
                axis=0
            )
            print(f"  Selected parameters for {agent_name}")
        
        # Stack agent parameters in environment order
        # Note: This assumes specific agent ordering - adjust based on your environment
        if "robot" in agent_params_for_eval and "human" in agent_params_for_eval:
            stacked_params = _stack_tree([
                agent_params_for_eval["robot"], 
                agent_params_for_eval["human"]
            ])
            print("  Stacked parameters: robot + human")
        else:
            # General case: stack in alphabetical order
            agent_names = sorted(agent_params_for_eval.keys())
            stacked_params = _stack_tree([
                agent_params_for_eval[name] for name in agent_names
            ])
            print(f"  Stacked parameters: {' + '.join(agent_names)}")
        
        # Create evaluation network state
        final_eval_network_state = EvalNetworkState(
            apply_fn=network.apply,
            params=stacked_params,
        )
        
        # ===== EVALUATION EXECUTION =====
        print("Running evaluation episodes for rendering...")
        eval_final = eval_jit(eval_rng, final_eval_network_state, eval_log_config)
        
        # ===== EPISODE SELECTION =====
        print("Selecting representative episodes...")
        worst_idx, median_idx, best_idx, episode_returns = _select_representative_episodes(eval_final)
        
        print(f"Selected episodes:")
        print(f"  Worst episode (index {worst_idx}): return = {episode_returns[worst_idx]:.1f}")
        print(f"  Median episode (index {median_idx}): return = {episode_returns[median_idx]:.1f}")
        print(f"  Best episode (index {best_idx}): return = {episode_returns[best_idx]:.1f}")
        
        # ===== EPISODE EXTRACTION =====
        print("Extracting episode trajectories...")
        
        # Get episode termination data for extraction
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        
        # Extract complete episode trajectories
        worst_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, 
            first_episode_done,
            time_idx=-1, 
            eval_idx=worst_idx,
        )
        
        median_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, 
            first_episode_done,
            time_idx=-1, 
            eval_idx=median_idx,
        )
        
        best_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, 
            first_episode_done,
            time_idx=-1, 
            eval_idx=best_idx,
        )
        
        print(f"Extracted episode lengths:")
        print(f"  Worst: {len(worst_episode)} timesteps")
        print(f"  Median: {len(median_episode)} timesteps") 
        print(f"  Best: {len(best_episode)} timesteps")
        
        # ===== HTML RENDERING =====
        print("Generating interactive HTML visualizations...")
        
        # Generate filenames with performance information
        worst_filename = f"final_worst_r{int(episode_returns[worst_idx])}.html"
        median_filename = f"final_median_r{int(episode_returns[median_idx])}.html"
        best_filename = f"final_best_r{int(episode_returns[best_idx])}.html"
        
        # Save interactive HTML files
        html.save(worst_filename, eval_env.sys, worst_episode)
        html.save(median_filename, eval_env.sys, median_episode)
        html.save(best_filename, eval_env.sys, best_episode)
        
        # ===== DISPLAY COMPLETION SUMMARY =====
        print("\n" + "="*60)
        print("RENDERING COMPLETED")


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
# from flax.traverse_util import flatten_dict, unflatten_dict
# import safetensors.flax
# import optax
# import distrax
# import jaxmarl
# from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
# from jaxmarl.wrappers.baselines import LogWrapper
# import hydra
# from omegaconf import OmegaConf
# from typing import Sequence, NamedTuple, Any, Dict, Callable
# from flax import struct

# @struct.dataclass
# class EvalNetworkState:
#     apply_fn: Callable = struct.field(pytree_node=False)
#     params: Dict

# def _tree_take(pytree, indices, axis=None):
#     return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

# def _unstack_tree(pytree):
#     leaves, treedef = jax.tree_util.tree_flatten(pytree)
#     unstacked_leaves = zip(*leaves)
#     return [jax.tree_util.tree_unflatten(treedef, leaves)
#             for leaves in unstacked_leaves]

# def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
#     episodes = _tree_take(pipeline_states, eval_idx, axis=1)
#     dones = dones.take(eval_idx, axis=1)
#     return [
#         state
#         for state, done in zip(_unstack_tree(episodes), dones)
#         if not (done)
#     ]

# def _stack_tree(pytree_list, axis=0):
#     return jax.tree.map(
#         lambda *leaf: jnp.stack(leaf, axis=axis),
#         *pytree_list
#     )

# @hydra.main(version_base=None, config_path="config", config_name="masac_mabrax")
# def main(config):
#     config = OmegaConf.to_container(config, resolve=True)



#     from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
#     from masac_ff_nps import MultiSACActor as NetworkArch

#     rng = jax.random.PRNGKey(config["SEED"])
#     rng, eval_rng = jax.random.split(rng)
#     with jax.disable_jit(config["DISABLE_JIT"]):
#         human_params = unflatten_dict(
#             safetensors.flax.load_file(config["eval"]["path"]["human"]), sep='/'
#         )

#         robot_params = unflatten_dict(
#             safetensors.flax.load_file(config["eval"]["path"]["robot"]), sep='/'
#         )

#         agent_params = {'human': human_params, 'robot': robot_params}
#         eval_env, run_eval = make_evaluation(config)
#         eval_log_config = EvalInfoLogConfig(
#             env_state=True,
#             done=True,
#             action=False,
#             reward=True,
#             log_prob=False,
#             obs=False,
#             info=False,
#             avail_actions=False,
#         )
#         eval_jit = jax.jit(
#             run_eval,
#             static_argnames=["log_eval_info"],
#         )
#         network = NetworkArch(config=config)
#         # RENDER
#         robot = _tree_take(
#             agent_params["robot"],
#             0,
#             axis=0
#         )
#         human = _tree_take(
#             agent_params["human"],
#             0,
#             axis=0
#         )
#         final_eval_network_state = EvalNetworkState(
#             apply_fn=network.apply,
#             params=_stack_tree([robot, human]),
#         )

#         eval_final = eval_jit(eval_rng, final_eval_network_state, eval_log_config)

#         first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
#         first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
#         first_episode_returns = first_episode_rewards.sum(axis=0)
#         episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
#         worst_idx = episode_argsort.take(0,axis=-1)
#         best_idx = episode_argsort.take(-1, axis=-1)
#         median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)
#         from brax.io import html
#         worst_episode = _take_episode(
#             eval_final.env_state.env_state.pipeline_state, first_episode_done,
#             time_idx=-1, eval_idx=worst_idx,
#         )
#         median_episode = _take_episode(
#             eval_final.env_state.env_state.pipeline_state, first_episode_done,
#             time_idx=-1, eval_idx=median_idx,
#         )
#         best_episode = _take_episode(
#             eval_final.env_state.env_state.pipeline_state, first_episode_done,
#             time_idx=-1, eval_idx=best_idx,
#         )
#         html.save(f"final_worst_r{int(first_episode_returns[worst_idx])}.html", eval_env.sys, worst_episode)
#         html.save(f"final_median_r{int(first_episode_returns[median_idx])}.html", eval_env.sys, median_episode)
#         html.save(f"final_best_r{int(first_episode_returns[best_idx])}.html", eval_env.sys, best_episode)



# if __name__ == "__main__":
#     main()