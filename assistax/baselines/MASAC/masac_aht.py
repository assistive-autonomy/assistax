"""
MASAC Ad Hoc Teamwork (AHT) Training Script

This module implements Ad Hoc Teamwork training for Multi-Agent Soft Actor-Critic (MASAC) agents.
AHT focuses on training agents that can coordinate effectively with previously unseen partners,
which is crucial for real-world multi-agent deployment where team composition may vary.

Key Features:
- Trains agents against a diverse population of zoo partners
- Implements train/test split for rigorous zero-shot coordination evaluation
- Evaluates performance against both seen (training) and unseen (test) partners
- Supports population-based training for robust policy development
- Comprehensive evaluation metrics for generalization assessment

Ad Hoc Teamwork Concept:
Ad Hoc Teamwork refers to the ability of an agent to collaborate effectively with teammates
it has never encountered before. This is essential for:
- Real-world deployment with unknown team compositions
- Robust multi-agent systems that adapt to partner diversity
- Zero-shot coordination capabilities
- Generalization beyond training distribution

Training Process:
1. Load zoo of trained partner agents (e.g., "human" behavioral policies)
2. Split zoo into training and test sets (typically 50/50 split)
3. Train new agents against the training set partners
4. Evaluate against both training partners (seen) and test partners (unseen)
5. Measure zero-shot coordination performance

Evaluation Metrics:
- Training Set Performance: How well the agent coordinates with seen partners
- Test Set Performance: How well the agent generalizes to unseen partners
- Generalization Gap: Difference between training and test performance

Usage:
    python masac_aht.py [hydra options] ZOO_PATH=path/to/zoo ALGORITHM=masac
    
The script requires a pre-populated zoo with partner agents to train against.
"""

import jax
import jax.numpy as jnp
import hydra
import safetensors.flax
from tqdm import tqdm
from flax.traverse_util import flatten_dict
from omegaconf import OmegaConf
import assistax
from assistax.wrappers.aht import ZooManager
from typing import Dict, Any, List


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
    
    Note: This function is included for compatibility but not used in AHT training.
    
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


# ================================ ZOO MANAGEMENT UTILITIES ================================

def _setup_zoo_splits(zoo, algorithm, scenario, partner_agent_id="human", split_ratio=0.5):
    """
    Set up training and test splits from zoo partners.
    
    Creates a random split of available zoo agents for training and evaluation.
    This enables rigorous testing of zero-shot coordination capabilities.
    
    Args:
        zoo: ZooManager instance
        algorithm: Algorithm name to filter zoo agents
        scenario: Scenario/environment name to filter zoo agents
        partner_agent_id: Agent ID of partners to load (default: "human")
        split_ratio: Fraction of agents to use for training (default: 0.5)
        
    Returns:
        Tuple of (train_agent_list, test_agent_list)
    """
    # Query zoo for relevant agents
    index_filtered = zoo.index.query(f'algorithm == "{algorithm}"'
                                   ).query(f'scenario == "{scenario}"'
                                   ).query(f'scenario_agent_id == "{partner_agent_id}"')
    
    if len(index_filtered) == 0:
        raise ValueError(f"No agents found in zoo for algorithm={algorithm}, scenario={scenario}, agent_id={partner_agent_id}")
    
    # Create random train/test split
    train_set = index_filtered.sample(frac=split_ratio)
    test_set = index_filtered.drop(train_set.index)
    
    train_agents = list(train_set.agent_uuid)
    test_agents = list(test_set.agent_uuid)
    
    print(f"Zoo split created:")
    print(f"  Total available agents: {len(index_filtered)}")
    print(f"  Training set: {len(train_agents)} agents")
    print(f"  Test set: {len(test_agents)} agents")
    print(f"  Split ratio: {split_ratio:.1%} train / {1-split_ratio:.1%} test")
    
    return train_agents, test_agents


# ================================ MAIN AHT TRAINING ORCHESTRATION ================================

@hydra.main(version_base=None, config_path="config", config_name="masac_aht")
def main(config):
    """
    Main orchestration function for MASAC Ad Hoc Teamwork training.
    
    This function implements the complete AHT training pipeline:
    1. Loads zoo of partner agents and creates train/test splits
    2. Trains new agents against the training set of partners
    3. Evaluates trained agents against both seen and unseen partners
    4. Saves comprehensive results for zero-shot coordination analysis
    
    The goal is to develop agents that can coordinate effectively with
    previously unseen partners, demonstrating robust generalization.
    
    Args:
        config: Hydra configuration object containing AHT training parameters
    """
    config = OmegaConf.to_container(config, resolve=True)
    
    print(f"Starting MASAC Ad Hoc Teamwork (AHT) Training")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Algorithm: {config['ALGORITHM']}")
    print(f"Zoo path: {config['ZOO_PATH']}")
    print(f"Number of seeds: {config['NUM_SEEDS']}")
    
    # ===== IMPORT ALGORITHM COMPONENTS =====
    from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
    
    # ===== ZOO SETUP AND PARTNER SPLIT =====
    print("\nSetting up zoo and partner splits...")
    zoo = ZooManager(config["ZOO_PATH"])
    
    # Create train/test split of zoo partners for rigorous evaluation
    train_agents, test_agents = _setup_zoo_splits(
        zoo=zoo,
        algorithm=config["ALGORITHM"],
        scenario=config["ENV_NAME"],
        partner_agent_id="human",  # Partner agent type to load
        split_ratio=0.5  # 50/50 split for training and testing
    )
    
    if len(train_agents) == 0 or len(test_agents) == 0:
        raise ValueError("Insufficient agents in zoo for train/test split. Need at least 2 agents.")
    
    # ===== ENVIRONMENT SETUP =====
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    print(f"Environment agents: {env.agents}")
    
    # ===== RANDOM NUMBER GENERATOR SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    # ===== TRAINING EXECUTION =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        print("\nCompiling training function...")
        print("Training against partner population from training set...")
        
        # Create training function that loads training set partners
        train_jit = jax.jit(
            make_train(
                config,
                save_train_state=False,  # Don't save training states for AHT
                load_zoo={"human": train_agents},  # Load training set partners
            ),
            device=jax.devices()[config["DEVICE"]]
        )
        
        print("Running training across all seeds...")
        print(f"Training against {len(train_agents)} partner agents...")
        
        # Train agents using vmap across seeds
        out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
            train_rngs,
            config["POLICY_LR"],
            config["Q_LR"],
            config["ALPHA_LR"],
            config["TAU"],
        )
        
        print("Training completed!")

        # ===== SAVE TRAINING METRICS =====
        print("Saving training metrics...")
        EXCLUDED_METRICS = ["train_state"]
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
        }, allow_pickle=True)

        # ===== SAVE MODEL PARAMETERS =====
        print("Saving model parameters...")
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state
        
        # Save all training states (for analysis across training)
        safetensors.flax.save_file(
            flatten_dict(all_train_states.params, sep='/'),
            "all_params.safetensors"
        )
        
        # Save final parameters (different format for parameter sharing vs independent)
        if config["network"]["agent_param_sharing"]:
            # For parameter sharing: single set of shared parameters
            safetensors.flax.save_file(
                flatten_dict(final_train_state.params, sep='/'),
                "final_params.safetensors"
            )
        else:
            # For independent parameters: split by agent
            split_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state.params)
            )
            for agent, params in zip(env.agents, split_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"{agent}.safetensors",
                )

        # ===== EVALUATION SETUP =====
        print("\nSetting up evaluation...")
        
        # Calculate evaluation batching for memory efficiency
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:2]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        
        print(f"Batch dimensions: {batch_dims}")
        print(f"Sequential evaluation batches: {n_sequential_evals}")

        def _flatten_and_split_trainstate(trainstate):
            """
            Flatten training states across batch dimensions and split for sequential evaluation.
            
            For AHT evaluation, we have 2 batch dimensions:
            - Random seeds
            - Training checkpoints
            
            Args:
                trainstate: Training state with shape (num_seeds, num_checkpoints, ...)
                
            Returns:
                List of training state chunks for sequential evaluation
            """
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
                trainstate
            )
            return _tree_split(flat_trainstate, n_sequential_evals)

        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)

        # ===== CREATE EVALUATION ENVIRONMENTS =====
        print("Creating evaluation environments...")
        
        # Training set evaluation (seen partners)
        eval_train_env, run_eval_train = make_evaluation(
            config, 
            load_zoo={"human": train_agents}
        )
        
        # Test set evaluation (unseen partners) 
        eval_test_env, run_eval_test = make_evaluation(
            config, 
            load_zoo={"human": test_agents}
        )
        
        # Configure evaluation logging for memory efficiency
        eval_log_config = EvalInfoLogConfig(
            env_state=False,     # Don't log environment state (saves memory)
            done=True,           # Log episode termination (needed for returns)
            action=False,        # Don't log actions (saves memory)
            reward=True,         # Log rewards (needed for returns)
            log_prob=False,      # Don't log action probabilities (saves memory)
            obs=False,           # Don't log observations (saves memory)
            info=False,          # Don't log additional info (saves memory)
            avail_actions=False, # Don't log available actions (saves memory)
        )
        
        # ===== JIT COMPILE EVALUATION FUNCTIONS =====
        print("Compiling evaluation functions...")
        eval_train_jit = jax.jit(run_eval_train, static_argnames=["log_eval_info"])
        eval_train_vmap = jax.vmap(eval_train_jit, in_axes=(None, 0, None))
        
        eval_test_jit = jax.jit(run_eval_test, static_argnames=["log_eval_info"])
        eval_test_vmap = jax.vmap(eval_test_jit, in_axes=(None, 0, None))

        # ===== EVALUATION EXECUTION =====
        print("\nRunning evaluation against training set partners (seen)...")
        evals_train = _concat_tree([
            eval_train_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Training set evaluation")
        ])
        evals_train = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals_train
        )

        print("Running evaluation against test set partners (unseen)...")
        evals_test = _concat_tree([
            eval_test_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Test set evaluation")
        ])
        evals_test = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals_test
        )

        # ===== COMPUTE PERFORMANCE METRICS =====
        print("Computing performance metrics...")
        
        # Training set performance (seen partners)
        train_first_episode_returns = _compute_episode_returns(evals_train)
        train_first_episode_returns = train_first_episode_returns["__all__"]
        train_mean_episode_returns = train_first_episode_returns.mean(axis=-1)
        
        # Test set performance (unseen partners) 
        test_first_episode_returns = _compute_episode_returns(evals_test)
        test_first_episode_returns = test_first_episode_returns["__all__"]
        test_mean_episode_returns = test_first_episode_returns.mean(axis=-1)

        # ===== SAVE EVALUATION RESULTS =====
        print("Saving evaluation results...")
        jnp.save("train_returns.npy", train_mean_episode_returns)
        jnp.save("test_returns.npy", test_mean_episode_returns)
        
        # ===== DISPLAY AHT RESULTS SUMMARY =====
        print("\n" + "="*70)
        print("AD HOC TEAMWORK (AHT) RESULTS SUMMARY")
        print("="*70)
        
        # Compute summary statistics
        train_performance = train_mean_episode_returns.mean()
        test_performance = test_mean_episode_returns.mean()
        generalization_gap = train_performance - test_performance
        
        print(f"Training Set Performance (Seen Partners):")
        print(f"  Mean return: {train_performance:.2f} ± {train_mean_episode_returns.std():.2f}")
        print(f"  Number of partners: {len(train_agents)}")
        print(f"")
        print(f"Test Set Performance (Unseen Partners):")
        print(f"  Mean return: {test_performance:.2f} ± {test_mean_episode_returns.std():.2f}")
        print(f"  Number of partners: {len(test_agents)}")
        print(f"")
        print(f"Zero-Shot Coordination Analysis:")
        print(f"  Generalization gap: {generalization_gap:.2f}")
        print(f"  Retention rate: {(test_performance/train_performance)*100:.1f}%")
        
        if generalization_gap < 0:
            print(f"  ✓ Positive transfer: Agent performs better on unseen partners!")
        elif generalization_gap < 0.1 * train_performance:
            print(f"  ✓ Good generalization: Minimal performance drop on unseen partners")
        else:
            print(f"  ⚠ Generalization gap: Significant performance drop on unseen partners")
        
        print(f"")
        print(f"Training Configuration:")
        print(f"  Environment: {config['ENV_NAME']}")
        print(f"  Seeds: {config['NUM_SEEDS']}")
        print(f"  Total timesteps: {config['TOTAL_TIMESTEPS']}")
        print(f"  Zoo path: {config['ZOO_PATH']}")
        print("="*70)
        
        print("Ad Hoc Teamwork training completed successfully!")
        print("Results saved:")
        print("  - train_returns.npy: Performance with seen partners")
        print("  - test_returns.npy: Performance with unseen partners")


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

# @hydra.main(version_base=None, config_path="config", config_name="masac_aht")
# def main(config):

#     config = OmegaConf.to_container(config, resolve=True)
#     from baselines.MASAC.masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
#     rng = jax.random.PRNGKey(config["SEED"])
#     train_rng, eval_rng = jax.random.split(rng)
#     train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
#     with jax.disable_jit(config["DISABLE_JIT"]):
#         zoo = ZooManager(config["ZOO_PATH"])
#         alg = config["ALGORITHM"]
#         scenario = config["ENV_NAME"]
#         index_filtered = zoo.index.query(f'algorithm == "{alg}"'
#                                  ).query(f'scenario == "{scenario}"'
#                                  ).query('scenario_agent_id == "human"')
#         train_set = index_filtered.sample(frac=0.5)
#         test_set = index_filtered.drop(train_set.index)
#         env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#         train_jit = jax.jit(
#             make_train(
#                 config,
#                 save_train_state=False,
#                 load_zoo={"human": list(train_set.agent_uuid)},
#             ),
#             device=jax.devices()[config["DEVICE"]]
#         )
#         out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
#             train_rngs,
#             config["POLICY_LR"],
#             config["Q_LR"],
#             config["ALPHA_LR"],
#             config["TAU"],
#         )

#         # SAVE TRAIN METRICS
#         EXCLUDED_METRICS = ["train_state"]
#         jnp.save("metrics.npy", {
#             key: val
#             for key, val in out["metrics"].items()
#             if key not in EXCLUDED_METRICS
#             },
#             allow_pickle=True
#         )

#         # SAVE PARAMS
#         env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#         all_train_states = out["metrics"]["train_state"]
#         final_train_state = out["runner_state"].train_state
#         safetensors.flax.save_file(
#             flatten_dict(all_train_states.params, sep='/'),
#             "all_params.safetensors"
#         )
#         if config["network"]["agent_param_sharing"]:
#             safetensors.flax.save_file(
#                 flatten_dict(final_train_state.params, sep='/'),
#                 "final_params.safetensors"
#             )
#         else:
#             # split by agent
#             split_params = _unstack_tree(
#                 jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state.params)
#             )
#             for agent, params in zip(env.agents, split_params):
#                 safetensors.flax.save_file(
#                     flatten_dict(params, sep='/'),
#                     f"{agent}.safetensors",
#                 )


#         # RUN EVALUATION
#         # Assume the first 2 dimensions are batch dims
#         batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:2]
#         n_sequential_evals = int(jnp.ceil(
#             config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
#             / config["GPU_ENV_CAPACITY"]
#         ))
#         def _flatten_and_split_trainstate(trainstate):
#             # We define this operation and JIT it for memory reasons
#             flat_trainstate = jax.tree.map(
#                 lambda x: x.reshape((x.shape[0]*x.shape[1],*x.shape[2:])),
#                 trainstate
#             )
#             return _tree_split(flat_trainstate, n_sequential_evals)
#         split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)

#         eval_train_env, run_eval_train = make_evaluation(config, load_zoo={"human": list(train_set.agent_uuid)})
#         eval_test_env, run_eval_test = make_evaluation(config, load_zoo={"human": list(test_set.agent_uuid)})
                
#         eval_log_config = EvalInfoLogConfig(
#             env_state=False,
#             done=True,
#             action=False,
#             value=False,
#             reward=True,
#             log_prob=False,
#             obs=False,
#             info=False,
#             avail_actions=False,
#         )
#         eval_train_jit = jax.jit(run_eval_train, static_argnames=["log_eval_info"])
#         eval_train_vmap = jax.vmap(eval_train_jit, in_axes=(None, 0, None))
#         eval_test_jit = jax.jit(run_eval_test, static_argnames=["log_eval_info"])
#         eval_test_vmap = jax.vmap(eval_test_jit, in_axes=(None, 0, None))
#         evals_train = _concat_tree([
#             eval_train_vmap(eval_rng, ts, eval_log_config)
#             for ts in tqdm(split_trainstate, desc="Evaluation batches")
#         ])
#         evals_train = jax.tree.map(
#             lambda x: x.reshape((*batch_dims, *x.shape[1:])),
#             evals_train
#         )
#         evals_test = _concat_tree([
#             eval_test_vmap(eval_rng, ts, eval_log_config)
#             for ts in tqdm(split_trainstate, desc="Evaluation batches")
#         ])
#         evals_test = jax.tree.map(
#             lambda x: x.reshape((*batch_dims, *x.shape[1:])),
#             evals_test
#         )

#         # COMPUTE RETURNS
#         train_first_episode_returns = _compute_episode_returns(evals_train)
#         train_first_episode_returns = train_first_episode_returns["__all__"]
#         train_mean_episode_returns = train_first_episode_returns.mean(axis=-1)
#         test_first_episode_returns = _compute_episode_returns(evals_test)
#         test_first_episode_returns = test_first_episode_returns["__all__"]
#         test_mean_episode_returns = test_first_episode_returns.mean(axis=-1)

#         # SAVE RETURNS
#         jnp.save("train_returns.npy", train_mean_episode_returns)
#         jnp.save("test_returns.npy", test_mean_episode_returns)

# if __name__ == "__main__":
#     main()
