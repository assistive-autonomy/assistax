"""
IPPO Hyperparameter Sweeping 

This module orchestrates large-scale hyperparameter sweeps for IPPO experiments across different
network architectures. It systematically explores hyperparameter spaces, manages experiment
organization, and handles efficient evaluation of multiple configurations simultaneously.

Key Features:
- Systematic hyperparameter space exploration (learning rate, entropy coefficient, clipping epsilon)
- Automatic experiment organization with unique directory creation
- Efficient nested vmapping for simultaneous sweep execution
- Dynamic algorithm selection based on network architecture configuration
- Comprehensive result saving (metrics, parameters, hyperparameters, returns)
- Memory-efficient evaluation across large parameter spaces
- Support for all four IPPO variants (FF/RNN x NPS/PS)

Differences from Single-Run Script:
- Generates hyperparameter configurations automatically
- Uses nested vmaps for sweep execution (seeds x hyperparameters)
- Creates unique directories for each experiment
- Handles 3D batch dimensions (hyperparams x seeds x envs)
- Focuses on systematic exploration rather than visualization
- Saves hyperparameter configurations separately

Usage:
    python ippo_sweep.py [hydra options]
    
The script will create a unique directory for each sweep configuration and save all
results systematically for later analysis.
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
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict
from base64 import urlsafe_b64encode


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
        if not (done)
    ]


def _compute_episode_returns(eval_info, common_reward=False, time_axis=-2):
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


# ================================ HYPERPARAMETER SWEEP UTILITIES ================================

def _generate_sweep_axes(rng, config):
    """
    Generate hyperparameter configurations for sweep experiments.
    
    Creates arrays of hyperparameter values to sweep over, sampling from
    log-uniform distributions for learning rate, entropy coefficient, and
    clipping epsilon based on configuration specifications.
    
    Args:
        rng: Random number generator key
        config: Configuration dictionary containing sweep specifications
        
    Returns:
        Dictionary containing hyperparameter values and their corresponding
        vmap axes for efficient parallel execution
    """
    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(rng, 3)
    sweep_config = config["SWEEP"]
    
    # Learning rate sweep configuration
    if sweep_config.get("lr", False):
        lrs = 10**jax.random.uniform(
            lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["lr"]["min"],
            maxval=sweep_config["lr"]["max"],
        )
        lr_axis = 0
    else:
        lrs = config["LR"]
        lr_axis = None

    # Entropy coefficient sweep configuration
    if sweep_config.get("ent_coef", False):
        ent_coefs = 10**jax.random.uniform(
            ent_coef_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["ent_coef"]["min"],
            maxval=sweep_config["ent_coef"]["max"],
        )
        ent_coef_axis = 0
    else:
        ent_coefs = config["ENT_COEF"]
        ent_coef_axis = None

    # Clipping epsilon sweep configuration
    if sweep_config.get("clip_eps", False):
        clip_epss = 10**jax.random.uniform(
            clip_eps_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["clip_eps"]["min"],
            maxval=sweep_config["clip_eps"]["max"],
        )
        clip_eps_axis = 0
    else:
        clip_epss = config["CLIP_EPS"]
        clip_eps_axis = None

    return {
        "lr": {"val": lrs, "axis": lr_axis},
        "ent_coef": {"val": ent_coefs, "axis": ent_coef_axis},
        "clip_eps": {"val": clip_epss, "axis": clip_eps_axis},
    }


# ================================ MAIN SWEEPING FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="ippo_sweep")
def main(config):
    """
    Main orchestration function for IPPO hyperparameter sweeping.
    
    This function:
    1. Creates a unique experiment directory based on configuration hash
    2. Dynamically imports the correct IPPO variant based on config
    3. Generates hyperparameter sweep configurations
    4. Runs training across all hyperparameter combinations using nested vmaps
    5. Saves all results systematically for later analysis
    6. Evaluates all trained models and computes performance metrics
    
    Args:
        config: Hydra configuration object containing all hyperparameters
    """
    # ===== EXPERIMENT ORGANIZATION =====
    # Create unique directory for this sweep configuration
    config_key = hash(config) % 2**62
    config_key = urlsafe_b64encode(
        config_key.to_bytes(
            (config_key.bit_length() + 8) // 8,
            "big", signed=False
        )
    ).decode("utf-8").replace("=", "")
    
    os.makedirs(config_key, exist_ok=True)
    print(f"Experiment directory: {config_key}")
    
    config = OmegaConf.to_container(config, resolve=True)

    # ===== DYNAMIC ALGORITHM SELECTION =====
    # Import the appropriate IPPO variant based on network architecture configuration
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Feedforward Networks with No Parameter Sharing")
        case (False, True):
            from ippo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Feedforward Networks with Parameter Sharing")
        case (True, False):
            from ippo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with No Parameter Sharing")
        case (True, True):
            from ippo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with Parameter Sharing")

    # ===== SWEEP SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    # Generate hyperparameter sweep configurations
    sweep = _generate_sweep_axes(sweep_rng, config)
    
    print(f"Hyperparameter sweep configurations:")
    print(f"  Learning rates: {sweep['lr']['val'] if sweep['lr']['axis'] is not None else 'Fixed'}")
    print(f"  Entropy coefficients: {sweep['ent_coef']['val'] if sweep['ent_coef']['axis'] is not None else 'Fixed'}")
    print(f"  Clipping epsilons: {sweep['clip_eps']['val'] if sweep['clip_eps']['axis'] is not None else 'Fixed'}")
    print(f"  Seeds: {config['NUM_SEEDS']}")
    
    # ===== TRAINING EXECUTION =====
    print("Starting hyperparameter sweep training...")
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        
        # Execute nested vmap for hyperparameter sweep
        # Outer vmap: across hyperparameter configurations
        # Inner vmap: across random seeds
        out = jax.vmap(
            jax.vmap(
                train_jit,
                in_axes=(0, None, None, None)  # Vmap over seeds
            ),
            in_axes=(
                None,  # Seeds (broadcast to all hyperparameter configs)
                sweep["lr"]["axis"],        # Learning rate axis
                sweep["ent_coef"]["axis"],  # Entropy coefficient axis
                sweep["clip_eps"]["axis"],  # Clipping epsilon axis
            )
        )(
            train_rngs,
            sweep["lr"]["val"],
            sweep["ent_coef"]["val"],
            sweep["clip_eps"]["val"],
        )

        # ===== SAVE TRAINING RESULTS =====
        print("Saving training metrics...")
        
        # Save training metrics (excluding large training states)
        EXCLUDED_METRICS = ["train_state"]
        jnp.save(f"{config_key}/metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )
        
        # Save hyperparameter configurations for analysis
        print("Saving hyperparameter configurations...")
        jnp.save(f"{config_key}/hparams.npy", {
            "lr": sweep["lr"]["val"],
            "ent_coef": sweep["ent_coef"]["val"],
            "clip_eps": sweep["clip_eps"]["val"],
            "num_steps": config["NUM_STEPS"],
            "num_envs": config["NUM_ENVS"],
            "update_epochs": config["UPDATE_EPOCHS"],
            "num_minibatches": config["NUM_MINIBATCHES"],
            }
        )

        # ===== SAVE MODEL PARAMETERS =====
        print("Saving model parameters...")
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state
        
        # Save all training states (for analysis across training)
        safetensors.flax.save_file(
            flatten_dict(all_train_states.params, sep='/'),
            f"{config_key}/all_params.safetensors"
        )
        
        # Save final parameters (different format for parameter sharing vs independent)
        if config["network"]["agent_param_sharing"]:
            # For parameter sharing: single set of shared parameters
            safetensors.flax.save_file(
                flatten_dict(final_train_state.params, sep='/'),
                f"{config_key}/final_params.safetensors"
            )
        else:
            # For independent parameters: split by agent
            # Note: Different axis manipulation for 3D sweep structure (hyperparams x seeds x agents)
            split_params = _unstack_tree(
                jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), final_train_state.params)
            )
            for agent, params in zip(env.agents, split_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"{config_key}/{agent}.safetensors",
                )

        # ===== EVALUATION SETUP =====
        print("Setting up evaluation...")
        
        # Calculate evaluation batching for memory efficiency
        # Note: 3D batch structure for sweep (hyperparams x seeds x envs)
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        
        def _flatten_and_split_trainstate(train_state):
            """
            Flatten training states across all batch dimensions and split for sequential evaluation.
            
            For sweep experiments, we have 3D batch structure (hyperparams x seeds x envs)
            that needs to be flattened for memory-efficient evaluation.
            """
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0] * x.shape[1] * x.shape[2], *x.shape[3:])),
                train_state
            )
            return _tree_split(flat_trainstate, n_sequential_evals)
        
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)

        # ===== EVALUATION EXECUTION =====
        print("Running evaluation...")
        eval_env, run_eval = make_evaluation(config)
        
        # Configure what information to log during evaluation
        eval_log_config = EvalInfoLogConfig(
            env_state=False,
            done=True,
            action=False,
            value=False,
            reward=True,
            log_prob=False,
            obs=False,
            info=False,
            avail_actions=False,
        )
        
        # JIT compile evaluation functions for efficiency
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
        
        # Run evaluation in batches for memory efficiency
        evals = _concat_tree([
            eval_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        
        # Reshape evaluation results back to original 3D batch structure
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # ===== COMPUTE PERFORMANCE METRICS =====
        print("Computing performance metrics...")
        first_episode_returns = _compute_episode_returns(evals)
        mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)

        # Save evaluation results
        jnp.save(f"{config_key}/returns.npy", mean_episode_returns)
        
        
        print("\nHyperparameter sweep completed successfully!")


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
#     lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(rng, 3)
#     sweep_config = config["SWEEP"]
#     if sweep_config.get("lr", False):
#         lrs = 10**jax.random.uniform(
#             lr_rng,
#             shape=(sweep_config["num_configs"],),
#             minval=sweep_config["lr"]["min"],
#             maxval=sweep_config["lr"]["max"],
#         )
#         lr_axis = 0
#     else:
#         lrs = config["LR"]
#         lr_axis = None

#     if sweep_config.get("ent_coef", False):
#         ent_coefs = 10**jax.random.uniform(
#             ent_coef_rng,
#             shape=(sweep_config["num_configs"],),
#             minval=sweep_config["ent_coef"]["min"],
#             maxval=sweep_config["ent_coef"]["max"],
#         )
#         ent_coef_axis = 0
#     else:
#         ent_coefs = config["ENT_COEF"]
#         ent_coef_axis = None

#     if sweep_config.get("clip_eps", False):
#         clip_epss = 10**jax.random.uniform(
#             clip_eps_rng,
#             shape=(sweep_config["num_configs"],),
#             minval=sweep_config["clip_eps"]["min"],
#             maxval=sweep_config["clip_eps"]["max"],
#         )
#         clip_eps_axis = 0
#     else:
#         clip_epss = config["CLIP_EPS"]
#         clip_eps_axis = None

#     return {
#         "lr": {"val": lrs, "axis": lr_axis},
#         "ent_coef": {"val": ent_coefs, "axis":ent_coef_axis},
#         "clip_eps": {"val": clip_epss, "axis":clip_eps_axis},
#     }


# @hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax")
# def main(config):
#     config_key = hash(config) % 2**62
#     config_key = urlsafe_b64encode(
#         config_key.to_bytes(
#             (config_key.bit_length()+8)//8,
#             "big", signed=False
#         )
#     ).decode("utf-8").replace("=", "")
#     os.makedirs(config_key, exist_ok=True)
#     config = OmegaConf.to_container(config, resolve=True)

#     # IMPORT FUNCTIONS BASED ON ARCHITECTURE
#     match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
#         case (False, False):
#             from ippo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
#         case (False, True):
#             from baselines.IPPO.ippo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
#         case (True, False):
#             from baselines.IPPO.ippo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
#         case (True, True):
#             from baselines.IPPO.ippo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig

#     rng = jax.random.PRNGKey(config["SEED"])
#     train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
#     train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
#     sweep = _generate_sweep_axes(sweep_rng, config)
#     with jax.disable_jit(config["DISABLE_JIT"]):
#         train_jit = jax.jit(
#             make_train(config, save_train_state=True),
#             device=jax.devices()[config["DEVICE"]]
#         )
#         out = jax.vmap(
#             jax.vmap(
#                 train_jit,
#                 in_axes=(0, None, None, None)
#             ),
#             in_axes=(
#                 None,
#                 sweep["lr"]["axis"],
#                 sweep["ent_coef"]["axis"],
#                 sweep["clip_eps"]["axis"],
#             )
#         )(
#             train_rngs,
#             sweep["lr"]["val"],
#             sweep["ent_coef"]["val"],
#             sweep["clip_eps"]["val"],
#         )

#         # SAVE TRAIN METRICS
#         EXCLUDED_METRICS = ["train_state"]
#         jnp.save(f"{config_key}/metrics.npy", {
#             key: val
#             for key, val in out["metrics"].items()
#             if key not in EXCLUDED_METRICS
#             },
#             allow_pickle=True
#         )
        
#         # SAVE SWEEP HPARAMS
#         jnp.save(f"{config_key}/hparams.npy", {
#             "lr": sweep["lr"]["val"],
#             "ent_coef": sweep["ent_coef"]["val"],
#             "clip_eps": sweep["clip_eps"]["val"],
#             "num_steps": config["NUM_STEPS"],
#             "num_envs": config["NUM_ENVS"],
#             "update_epochs": config["UPDATE_EPOCHS"],
#             "num_minibatches": config["NUM_MINIBATCHES"],
#             }
#         )

#         # SAVE PARAMS
#         env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#         all_train_states = out["metrics"]["train_state"]
#         final_train_state = out["runner_state"].train_state
#         safetensors.flax.save_file(
#             flatten_dict(all_train_states.params, sep='/'),
#             f"{config_key}/all_params.safetensors"
#         )
#         if config["network"]["agent_param_sharing"]:
#             safetensors.flax.save_file(
#                 flatten_dict(final_train_state.params, sep='/'),
#                 f"{config_key}/final_params.safetensors"
#             )
#         else:
#             # split by agent
#             split_params = _unstack_tree(
#                 jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), final_train_state.params)
#             )
#             for agent, params in zip(env.agents, split_params):
#                 safetensors.flax.save_file(
#                     flatten_dict(params, sep='/'),
#                     f"{config_key}/{agent}.safetensors",
#                 )

#         # RUN EVALUATION
#         # Assume the first 3 dimensions are batch dims
#         batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3]
#         n_sequential_evals = int(jnp.ceil(
#             config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
#             / config["GPU_ENV_CAPACITY"]
#         ))
#         def _flatten_and_split_trainstate(train_state):
#             # We define this operation and JIT it for memory reasons
#             flat_trainstate = jax.tree.map(
#                 lambda x: x.reshape((x.shape[0]*x.shape[1]*x.shape[2],*x.shape[3:])),
#                 train_state
#             )
#             return _tree_split(flat_trainstate, n_sequential_evals)
#         split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)

#         eval_env, run_eval = make_evaluation(config)
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
#         eval_jit = jax.jit(
#             run_eval,
#             static_argnames=["log_eval_info"],
#         )
#         eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
#         evals = _concat_tree([
#             eval_vmap(eval_rng, ts, eval_log_config)
#             for ts in tqdm(split_trainstate, desc="Evaluation batches")
#         ])
#         evals = jax.tree.map(
#             lambda x: x.reshape((*batch_dims, *x.shape[1:])),
#             evals
#         )
#         breakpoint()
#         # COMPUTE RETURNS
#         first_episode_returns = _compute_episode_returns(evals)
#         mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)

#         # SAVE RETURNS
#         jnp.save(f"{config_key}/returns.npy", mean_episode_returns)


# if __name__ == "__main__":
#     main()