"""
Independent Proximal Policy Optimization (IPPO) Rendering and Evaluation Script

This module provides comprehensive evaluation and visualization capabilities for trained
IPPO agents using the JaxMARL framework. It loads pre-trained models, runs evaluation 
episodes, and generates HTML visualizations of the best, worst, and median performing 
episodes using Brax's rendering system.
"""

import os
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
    Network state structure for IPPO evaluation with pre-trained models.
    """
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict


# ================================ TREE MANIPULATION UTILITIES ================================

def _tree_take(pytree, indices, axis=None):
    """
    Extract elements from PyTree along specified axis using indices.

    """
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

def _unstack_tree(pytree):
    """
    Convert PyTree with stacked arrays to list of PyTrees.
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    """
    Extract episodes from evaluation pipeline states.
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
    """
    return jax.tree.map(lambda x: x.shape, pytree)

def _stack_tree(pytree_list, axis=0):
    """
    Stack a list of PyTrees along specified axis.
    """
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )


# ================================ MAIN EVALUATION AND RENDERING ================================

@hydra.main(version_base=None, config_path="config", config_name="ippo")
def main(config):
    """
    Main function for IPPO evaluation and trajectory rendering.
    
    Args:
        config: Hydra configuration object containing evaluation settings
    """
    print("Starting IPPO evaluation and rendering...")
    config = OmegaConf.to_container(config, resolve=True)

    # ===== DYNAMIC IMPORTS BASED ON ARCHITECTURE =====
    print("Loading IPPO implementation...")
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps import make_evaluation, EvalInfoLogConfig
            from ippo_ff_nps import MultiActorCritic as NetworkArch
            arch_name = "FF + NPS (Multi Actor-Critic)"
        case (False, True):
            from ippo_ff_ps import make_evaluation, EvalInfoLogConfig
            from ippo_ff_ps import ActorCritic as NetworkArch
            arch_name = "FF + PS (Shared Actor-Critic)"
        case (True, False):
            from ippo_rnn_nps import make_evaluation, EvalInfoLogConfig
            from ippo_rnn_nps import MultiActorCriticRNN as NetworkArch
            arch_name = "RNN + NPS (Multi Actor-Critic RNN)"
        case (True, True):
            from ippo_rnn_ps import make_evaluation, EvalInfoLogConfig
            from ippo_rnn_ps import ActorCriticRNN as NetworkArch
            arch_name = "RNN + PS (Shared Actor-Critic RNN)"
        case _:
            raise Exception(
                f"Unsupported IPPO configuration: recurrent={config['network']['recurrent']}, "
                f"param_sharing={config['network']['agent_param_sharing']}"
            )
    
    print(f"Using {arch_name} architecture")

    # ===== RANDOM NUMBER GENERATION =====
    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    
    with jax.disable_jit(config["DISABLE_JIT"]):
        
        # ===== AGENT PARAMETER LOADING =====
        print("Loading pre-trained agent parameters from SafeTensors...")
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
        print("Setting up JaxMARL evaluation environment...")
        eval_env, run_eval = make_evaluation(config)
        eval_log_config = EvalInfoLogConfig(
            env_state=True,        # Need environment states for Brax rendering
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
        print("Creating evaluation network state...")
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
            
            # Stack parameters for multi-agent IPPO network
            final_eval_network_state = EvalNetworkState(
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
            
            # Stack parameters for multi-agent IPPO network (robot first, then human)
            final_eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=_stack_tree([robot, human]),
            )
        
        # ===== EVALUATION EXECUTION =====
        print(f"Running IPPO evaluation with {config.get('NUM_EVAL_EPISODES', 'default')} episodes...")
        eval_start_time = jax.lax.start_timer()
        eval_final = eval_jit(eval_rng, final_eval_network_state, eval_log_config)
        eval_time = jax.lax.stop_timer(eval_start_time)
        print(f"Evaluation completed in {eval_time:.2f} seconds")

        # ===== EPISODE PERFORMANCE ANALYSIS =====
        print("Analyzing episode performance for representative selection...")
        
        # Compute cumulative done flags to identify episode boundaries
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        
        # Calculate episode returns by masking rewards after episode completion
        # For IPPO, agents have independent rewards but share environment episodes
        first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        
        # Print performance statistics
        mean_return = first_episode_returns.mean()
        std_return = first_episode_returns.std()
        min_return = first_episode_returns.min()
        max_return = first_episode_returns.max()
        
        print(f"IPPO Performance Summary:")
        print(f"  Mean return: {mean_return:.3f} ± {std_return:.3f}")
        print(f"  Return range: [{min_return:.3f}, {max_return:.3f}]")
        print(f"  Total episodes: {len(first_episode_returns)}")
        
        # ===== REPRESENTATIVE EPISODE SELECTION =====
        # Sort episodes by performance for representative sampling
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        worst_idx = episode_argsort.take(0, axis=-1)                           # Lowest return
        best_idx = episode_argsort.take(-1, axis=-1)                          # Highest return
        median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)  # Middle return
        
        print(f"Selected episodes for Brax rendering:")
        print(f"  Worst episode (idx {worst_idx}): return = {first_episode_returns[worst_idx]:.2f}")
        print(f"  Median episode (idx {median_idx}): return = {first_episode_returns[median_idx]:.2f}")
        print(f"  Best episode (idx {best_idx}): return = {first_episode_returns[best_idx]:.2f}")

        # ===== BRAX TRAJECTORY RENDERING =====
        print("Generating Brax HTML visualizations...")
        from assistax.render import html
        
        # Extract episode trajectories from Brax pipeline states for rendering
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
        
        # Generate HTML visualizations with descriptive filenames including returns
        worst_return = int(first_episode_returns[worst_idx])
        median_return = int(first_episode_returns[median_idx])
        best_return = int(first_episode_returns[best_idx])
        
        html.save(f"final_worst_r{worst_return}.html", eval_env.sys, worst_episode)
        html.save(f"final_median_r{median_return}.html", eval_env.sys, median_episode)
        html.save(f"final_best_r{best_return}.html", eval_env.sys, best_episode)
        
        print(f"Brax HTML files saved:")
        print(f"  final_worst_r{worst_return}.html")
        print(f"  final_median_r{median_return}.html")
        print(f"  final_best_r{best_return}.html")
        
        print("IPPO evaluation and rendering completed successfully!")


if __name__ == "__main__":
    main()