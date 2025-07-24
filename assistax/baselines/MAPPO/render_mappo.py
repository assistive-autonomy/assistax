"""
Multi-Agent Proximal Policy Optimization (MAPPO) Rendering and Evaluation Script

This module provides evaluation and visualization capabilities for trained
MAPPO agents. It loads pre-trained models, runs evaluation episodes, and generates
HTML visualizations of the best, worst, and median performing episodes.
"""

import os
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
import safetensors.flax
import optax
import distrax
import assistax
from assistax.wrappers.baselines import get_space_dim, LogEnvState
from assistax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Callable
from flax import struct


# ================================ EVALUATION DATA STRUCTURES ================================

@struct.dataclass
class EvalNetworkState:
    """
    Network state structure for evaluation with pre-trained models.
    
    Simplified version of TrainState that only contains the information
    needed for inference: the apply function and trained parameters.
    
    Attributes:
        apply_fn: Network forward pass function (not part of PyTree)
        params: Trained network parameters
    """
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict

@struct.dataclass
class ActorNetworkState:
    """
    Actor-only network state for environments requiring only policy networks.
    
    Used when critic networks are not needed for evaluation (e.g., when
    only generating rollouts without value estimation).
    
    Attributes:
        actor: Actor network evaluation state
    """
    actor: EvalNetworkState


# ================================ TREE MANIPULATION UTILITIES ================================

def _tree_take(pytree, indices, axis=None):
    """
    Extract elements from PyTree along specified axis using indices.
    
    Essential for selecting specific episodes or agent parameters from
    batched structures during evaluation and rendering.
    
    Args:
        pytree: JAX PyTree to extract from
        indices: Array indices to extract
        axis: Axis along which to extract (None for flat indexing)
        
    Returns:
        PyTree with extracted elements
    """
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

def _unstack_tree(pytree):
    """
    Convert PyTree with stacked arrays to list of PyTrees.
    
    Used to separate batched episode data into individual episodes
    for rendering and analysis.
    
    Args:
        pytree: PyTree with arrays that have a stackable leading dimension
        
    Returns:
        List of PyTrees, one for each element in the leading dimension
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    """
    Extract incomplete episodes from evaluation pipeline states.
    
    Filters out completed episodes and extracts trajectory states for
    rendering. Only returns episodes that haven't terminated early,
    ensuring we get complete trajectory visualizations.
    
    Args:
        pipeline_states: Batched evaluation states from simulation pipeline
        dones: Episode termination flags indicating which episodes completed
        time_idx: Time index to check (-1 for last timestep)
        eval_idx: Evaluation episode index to extract
        
    Returns:
        List of pipeline states for incomplete episodes ready for rendering
    """
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not (done)
    ]

def _tree_shape(pytree):
    """
    Get shapes of all leaves in a PyTree for debugging and validation.
    
    Args:
        pytree: JAX PyTree to inspect
        
    Returns:
        PyTree with same structure but leaf shapes instead of values
    """
    return jax.tree.map(lambda x: x.shape, pytree)

def _stack_tree(pytree_list, axis=0):
    """
    Stack a list of PyTrees along specified axis.
    
    Used to combine individual agent parameters into multi-agent
    network structures for evaluation.
    
    Args:
        pytree_list: List of PyTrees with identical structure
        axis: Axis along which to stack
        
    Returns:
        Single PyTree with stacked arrays
    """
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )


# ================================ MAIN EVALUATION AND RENDERING ================================

@hydra.main(version_base=None, config_path="config", config_name="mappo")
def main(config):
    """
    Main function for MAPPO evaluation and trajectory rendering.
    
    This function orchestrates the complete evaluation and visualization pipeline:
    1. Dynamically imports the appropriate MAPPO implementation
    2. Loads pre-trained agent parameters from SafeTensors files
    3. Creates evaluation network state for multi-agent inference
    4. Runs evaluation episodes to collect trajectory data
    5. Analyzes episode performance to select representative samples
    6. Generates HTML visualizations for qualitative analysis
    
    The script automatically handles different environment configurations
    and network architectures based on the provided configuration.
    
    Args:
        config: Hydra configuration object containing evaluation settings
    """
    print("Starting MAPPO evaluation and rendering...")
    config = OmegaConf.to_container(config, resolve=True)

    # ===== DYNAMIC IMPORTS BASED ON ARCHITECTURE =====
    print("Loading MAPPO implementation...")
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from mappo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_ff_nps import MultiActor as NetworkArch
            arch_name = "FF + NPS"
        case (False, True):
            from mappo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_ff_ps import Actor as NetworkArch
            arch_name = "FF + PS"
        case (True, False):
            from mappo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_rnn_nps import MultiActorRNN as NetworkArch
            arch_name = "RNN + NPS"
        case (True, True):
            from mappo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_rnn_ps import ActorRNN as NetworkArch
            arch_name = "RNN + PS"
        case _:
            raise Exception(
                f"Unsupported MAPPO configuration: recurrent={config['network']['recurrent']}, "
                f"param_sharing={config['network']['agent_param_sharing']}"
            )
    
    print(f"Using {arch_name} architecture")
    
    # ===== RANDOM NUMBER GENERATION =====
    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    
    with jax.disable_jit(config["DISABLE_JIT"]):
        
        # ===== AGENT PARAMETER LOADING =====
        print("Loading pre-trained agent parameters...")
        if config['ENV_NAME'] == 'pushcoop':
            robot1_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["human"]), sep='/'
            )
            robot2_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["robot"]), sep='/'
            )
            agent_params = {'robot1': robot1_params, 'robot2': robot2_params}
        else:
            human_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["human"]), sep='/'
            )
            robot_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["robot"]), sep='/'
            )
            agent_params = {'human': human_params, 'robot': robot_params}

        # ===== EVALUATION SETUP =====
        print("Setting up evaluation environment...")
        eval_env, run_eval = make_evaluation(config)
        eval_log_config = EvalInfoLogConfig(
            env_state=True,        # Need environment states for rendering
            done=True,             # Need done flags for episode boundary detection
            action=False,          # Don't need actions for performance analysis
            value=False,           # Don't need value estimates for rendering
            reward=True,           # Need rewards for return computation
            log_prob=False,        # Don't need log probabilities
            obs=False,             # Don't need observations for rendering
            info=False,            # Don't need environment info
            avail_actions=False,   # Don't need available actions
        )
        
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        network = NetworkArch(config=config)

        # ===== EVALUATION NETWORK STATE CREATION =====
        if config['ENV_NAME'] == 'pushcoop':
            # Extract first seed/config parameters for both robots
            robot1 = _tree_take(
                agent_params["robot1"],
                0,
                axis=0
            )
            robot2 = _tree_take(
                agent_params["robot2"],
                0,
                axis=0
            )
            
            # Stack parameters for multi-agent network
            eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=_stack_tree([robot1, robot2]),
            )
        else:    
            # Extract first seed/config parameters for human and robot
            robot = _tree_take(
                agent_params["robot"],
                0,
                axis=0
            )
            human = _tree_take(
                agent_params["human"],
                0,
                axis=0
            )
            
            # Stack parameters for multi-agent network (robot first, then human)
            eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=_stack_tree([robot, human]),
            )
        
        # ===== EVALUATION EXECUTION =====
        print(f"Running evaluation with {config.get('NUM_EVAL_EPISODES', 'default')} episodes...")
        eval_start = time.time()
        eval_final = eval_jit(eval_rng, eval_network_state, eval_log_config)
        eval_time = time.time() - eval_start
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # ===== EPISODE PERFORMANCE ANALYSIS =====
        print("Analyzing episode performance...")
        
        # Compute cumulative done flags to identify episode boundaries
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        
        # Calculate episode returns by masking rewards after episode completion
        first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        
        # Print performance statistics
        mean_return = first_episode_returns.mean()
        std_return = first_episode_returns.std()
        min_return = first_episode_returns.min()
        max_return = first_episode_returns.max()
        
        print(f"Performance Summary:")
        print(f"  Mean return: {mean_return:.3f} ± {std_return:.3f}")
        print(f"  Return range: [{min_return:.3f}, {max_return:.3f}]")
        print(f"  Total episodes: {len(first_episode_returns)}")
        
        # ===== REPRESENTATIVE EPISODE SELECTION =====
        # Sort episodes by performance for representative sampling
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        worst_idx = episode_argsort.take(0, axis=-1)                           # Lowest return
        best_idx = episode_argsort.take(-1, axis=-1)                          # Highest return
        median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)  # Middle return
        
        print(f"Selected episodes for rendering:")
        print(f"  Worst episode (idx {worst_idx}): return = {first_episode_returns[worst_idx]:.2f}")
        print(f"  Median episode (idx {median_idx}): return = {first_episode_returns[median_idx]:.2f}")
        print(f"  Best episode (idx {best_idx}): return = {first_episode_returns[best_idx]:.2f}")

        # ===== TRAJECTORY RENDERING =====
        print("Generating HTML visualizations...")
        from assistax.render import html
        
        # Extract episode trajectories for rendering
        pipeline_states = eval_final.env_state.env_state.pipeline_state
        
        worst_episode = _take_episode(
            pipeline_states, first_episode_done,
            time_idx=-1, eval_idx=worst_idx,
        )
        median_episode = _take_episode(
            pipeline_states, first_episode_done,
            time_idx=-1, eval_idx=median_idx,
        )
        best_episode = _take_episode(
            pipeline_states, first_episode_done,
            time_idx=-1, eval_idx=best_idx,
        )
        
        # Generate HTML visualizations with descriptive filenames
        worst_return = int(first_episode_returns[worst_idx])
        median_return = int(first_episode_returns[median_idx])
        best_return = int(first_episode_returns[best_idx])
        
        html.save(f"final_worst_r{worst_return}.html", eval_env.sys, worst_episode)
        html.save(f"final_median_r{median_return}.html", eval_env.sys, median_episode)
        html.save(f"final_best_r{best_return}.html", eval_env.sys, best_episode)
        
        print(f"HTML files saved:")
        print(f"  final_worst_r{worst_return}.html")
        print(f"  final_median_r{median_return}.html")
        print(f"  final_best_r{best_return}.html")
        
        print("Evaluation and rendering completed successfully!")

if __name__ == "__main__":
    main()
