"""
MAPPO Hyperparameter Sweeping 

This module orchestrates large-scale hyperparameter sweeps for MAPPO experiments across different
network architectures. It systematically explores hyperparameter spaces, manages experiment
organization, and handles efficient evaluation of multiple configurations simultaneously, 
ensuring parity with the IPPO sweep pipeline.

Usage:
    python mappo_sweep.py [hydra options]
"""

import os
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.traverse_util import flatten_dict
import safetensors.flax
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict
from base64 import urlsafe_b64encode
import assistax

from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode,
    _tree_shape, _stack_tree, _concat_tree, _tree_split
    )
from assistax.baselines.utils import _compute_episode_returns_sweep as _compute_episode_returns

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ' 
)

# ================================ HYPERPARAMETER SWEEP UTILITIES ================================

def _generate_sweep_axes(rng, config):
    """
    Generate hyperparameter configurations for sweep experiments.
    
    Creates arrays of hyperparameter values to sweep over, sampling from
    log-uniform distributions for learning rate, entropy coefficient, and
    clipping epsilon based on configuration specifications.
    """
    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(rng, 3)
    sweep_config = config["SWEEP"]
    
    # Helper to handle sweep vs fixed params
    def get_axis_info(param_name, rng_key):
        if sweep_config.get(param_name, False):
            vals = 10**jax.random.uniform(
                rng_key,
                shape=(sweep_config["num_configs"],),
                minval=sweep_config[param_name]["min"],
                maxval=sweep_config[param_name]["max"],
            )
            return {"val": vals, "axis": 0}
        return {"val": config[param_name.upper()], "axis": None}

    lr = get_axis_info("lr", lr_rng)
    ent_coef = get_axis_info("ent_coef", ent_coef_rng)
    clip_eps = get_axis_info("clip_eps", clip_eps_rng)

    return {
        "lr": lr,
        "ent_coef": ent_coef,
        "clip_eps": clip_eps,
    }

# ================================ MAIN SWEEPING FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="mappo_sweep")
def main(config):
    """
    Main orchestration function for MAPPO hyperparameter sweeping.
    """
    # ===== EXPERIMENT ORGANIZATION =====
    config_key = hash(config) % 2**62
    config_key = urlsafe_b64encode(
        config_key.to_bytes((config_key.bit_length() + 8) // 8, "big", signed=False)
    ).decode("utf-8").replace("=", "")
    
    os.makedirs(config_key, exist_ok=True)
    print(f"Experiment directory: {config_key}")
    
    config = OmegaConf.to_container(config, resolve=True)

    # ===== DYNAMIC ALGORITHM SELECTION =====
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from mappo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: MAPPO Feedforward - No Parameter Sharing")
        case (False, True):
            from mappo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: MAPPO Feedforward - Parameter Sharing")
        case (True, False):
            from mappo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: MAPPO Recurrent - No Parameter Sharing")
        case (True, True):
            from mappo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: MAPPO Recurrent - Parameter Sharing")

    # ===== SWEEP SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
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
        
        # Nested vmap for parallel sweep execution
        out = jax.vmap(
            jax.vmap(train_jit, in_axes=(0, None, None, None)),
            in_axes=(None, sweep["lr"]["axis"], sweep["ent_coef"]["axis"], sweep["clip_eps"]["axis"])
        )(
            train_rngs,
            sweep["lr"]["val"],
            sweep["ent_coef"]["val"],
            sweep["clip_eps"]["val"],
        )

        # ===== SAVE TRAINING RESULTS =====
        print("Saving training metrics...")
        EXCLUDED_METRICS = ["train_state"]
        jnp.save(f"{config_key}/metrics.npy", {
            key: val for key, val in out["metrics"].items() if key not in EXCLUDED_METRICS
        }, allow_pickle=True)
        
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
        # ===== EVALUATION PIPELINE =====
        print("Setting up evaluation...")
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state

        if config["SAVE_ALL_TRAIN_STATES"]:
            print("Saving model parameters...")
            safetensors.flax.save_file(
                flatten_dict(all_train_states.actor.params, sep='/'),
                f"{config_key}/all_params.safetensors"
            )
            
            if not config["network"]["agent_param_sharing"]:
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
        
        # Note: In MAPPO, we evaluate based on the Actor parameters
        # Batch structure: (hyperparams x seeds x envs)
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.actor.params))[:3]
        
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        
        def _flatten_and_split_trainstate(train_state):
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0] * x.shape[1] * x.shape[2], *x.shape[3:])),
                train_state
            )
            return _tree_split(flat_trainstate, n_sequential_evals)
        
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states.actor)

        print("Running evaluation...")
        eval_env, run_eval = make_evaluation(config)
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
        eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
        
        evals = _concat_tree([
            eval_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        
        # Reshape to original sweep structure
        evals = jax.tree.map(lambda x: x.reshape((*batch_dims, *x.shape[1:])), evals)

        # ===== COMPUTE PERFORMANCE METRICS =====
        print("Computing performance metrics...")
        first_episode_returns = _compute_episode_returns(evals)
        mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)

        jnp.save(f"{config_key}/returns.npy", mean_episode_returns)
        print(f"\nSweep completed successfully. Results in: {config_key}")


if __name__ == "__main__":
    main()
#"""
#MAPPO Hyperparameter Sweep Script
#
#This script performs comprehensive hyperparameter sweeps for MAPPO across different architectures:
#- Feedforward vs Recurrent (RNN) networks
#- Parameter sharing vs No parameter sharing
#- Multiple hyperparameters: learning rate, entropy coefficient, clip epsilon
#
#The script automatically:
#1. Generates hyperparameter configurations
#2. Runs training across all combinations using JAX vmap for efficiency
#3. Saves training metrics, hyperparameters, and model parameters
#4. Evaluates trained models and computes episode returns
#5. Organizes results in a structured directory
#
#Key Features:
#- Efficient parallel training across hyperparameter combinations
#- Automatic model selection based on network architecture
#- Memory-efficient evaluation with batching
#- Comprehensive result saving for analysis
#"""
#
#import os
#import time
#from typing import Dict, List, Tuple, Any, Optional
#from base64 import urlsafe_b64encode
#
#import jax
#import jax.numpy as jnp
#import flax.linen as nn
#from flax.traverse_util import flatten_dict
#import safetensors.flax
#import hydra
#from omegaconf import OmegaConf
#from tqdm import tqdm
#
#import assistax
#
#from assistax.baselines.utils import (
#    _tree_take, _unstack_tree, _take_episode,
#    _tree_shape, _stack_tree, _concat_tree, _tree_split
#    )
#from assistax.baselines.utils import _compute_episode_returns_sweep as _compute_episode_returns
#
## ============================================================================
## HYPERPARAMETER SWEEP CONFIGURATION
## ============================================================================
#
#def _generate_sweep_axes(rng: jax.Array, config: Dict) -> Dict[str, Dict[str, Any]]:
#    """
#    Generate hyperparameter sweep configurations.
#    
#    For each hyperparameter, either generates a range of values (if sweep is enabled)
#    or uses the single value from config.
#    
#    Args:
#        rng: Random number generator key
#        config: Configuration dictionary containing sweep settings
#        
#    Returns:
#        Dictionary containing sweep axes with values and vmap axes
#    """
#    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(rng, 3)
#    sweep_config = config["SWEEP"]
#    
#    # Generate learning rate sweep
#    if sweep_config.get("lr", False):
#        lrs = 10**jax.random.uniform(
#            lr_rng,
#            shape=(sweep_config["num_configs"],),
#            minval=sweep_config["lr"]["min"],
#            maxval=sweep_config["lr"]["max"],
#        )
#        lr_axis = 0
#    else:
#        lrs = config["LR"]
#        lr_axis = None
#
#    # Generate entropy coefficient sweep
#    if sweep_config.get("ent_coef", False):
#        ent_coefs = 10**jax.random.uniform(
#            ent_coef_rng,
#            shape=(sweep_config["num_configs"],),
#            minval=sweep_config["ent_coef"]["min"],
#            maxval=sweep_config["ent_coef"]["max"],
#        )
#        ent_coef_axis = 0
#    else:
#        ent_coefs = config["ENT_COEF"]
#        ent_coef_axis = None
#
#    # Generate clip epsilon sweep
#    if sweep_config.get("clip_eps", False):
#        clip_epss = 10**jax.random.uniform(
#            clip_eps_rng,
#            shape=(sweep_config["num_configs"],),
#            minval=sweep_config["clip_eps"]["min"],
#            maxval=sweep_config["clip_eps"]["max"],
#        )
#        clip_eps_axis = 0
#    else:
#        clip_epss = config["CLIP_EPS"]
#        clip_eps_axis = None
#
#    return {
#        "lr": {"val": lrs, "axis": lr_axis},
#        "ent_coef": {"val": ent_coefs, "axis": ent_coef_axis},
#        "clip_eps": {"val": clip_epss, "axis": clip_eps_axis},
#    }
#
#
## ============================================================================
## ARCHITECTURE SELECTION
## ============================================================================
#
#def _import_mappo_variant(config: Dict):
#    """
#    Import the appropriate MAPPO variant based on configuration.
#    
#    Args:
#        config: Configuration dictionary specifying architecture
#        
#    Returns:
#        Tuple of (make_train, make_evaluation, EvalInfoLogConfig) functions
#    """
#    recurrent = config["network"]["recurrent"]
#    param_sharing = config["network"]["agent_param_sharing"]
#    
#    match (recurrent, param_sharing):
#        case (False, False):
#            # Feedforward, No Parameter Sharing
#            from mappo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
#        case (False, True):
#            # Feedforward, Parameter Sharing
#            from mappo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
#        case (True, False):
#            # Recurrent, No Parameter Sharing
#            from mappo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
#        case (True, True):
#            # Recurrent, Parameter Sharing
#            from mappo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
#    
#    return make_train, make_evaluation, EvalInfoLogConfig
#
#
## ============================================================================
## RESULT SAVING FUNCTIONS
## ============================================================================
#
#def _save_training_metrics(config_key: str, metrics: Dict):
#    """Save training metrics excluding large objects like train_state."""
#    EXCLUDED_METRICS = ["train_state"]
#    filtered_metrics = {
#        key: val
#        for key, val in metrics.items()
#        if key not in EXCLUDED_METRICS
#    }
#    jnp.save(f"{config_key}/metrics.npy", filtered_metrics, allow_pickle=True)
#
#
#def _save_hyperparameters(config_key: str, sweep: Dict, config: Dict):
#    """Save hyperparameter configurations used in the sweep."""
#    hparams = {
#        "lr": sweep["lr"]["val"],
#        "ent_coef": sweep["ent_coef"]["val"],
#        "clip_eps": sweep["clip_eps"]["val"],
#        "num_steps": config["NUM_STEPS"],
#        "num_envs": config["NUM_ENVS"],
#        "update_epochs": config["UPDATE_EPOCHS"],
#        "num_minibatches": config["NUM_MINIBATCHES"],
#    }
#    jnp.save(f"{config_key}/hparams.npy", hparams)
#
#
#def _save_model_parameters(config_key: str, config: Dict, all_train_states, final_train_state):
#    """
#    Save model parameters for different architectures.
#    
#    For parameter sharing: saves single set of parameters
#    For no parameter sharing: saves separate parameters for each agent
#    """
#    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#    
#    # Save all training states (across sweep)
#    safetensors.flax.save_file(
#        flatten_dict(all_train_states.actor.params, sep='/'),
#        f"{config_key}/all_params.safetensors"
#    )
#    
#    if config["network"]["agent_param_sharing"]:
#        # Parameter sharing: single set of parameters
#        safetensors.flax.save_file(
#            flatten_dict(final_train_state.actor.params, sep='/'),
#            f"{config_key}/final_params.safetensors"
#        )
#    else:
#        # No parameter sharing: separate parameters per agent
#        split_params = _unstack_tree(
#            jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), final_train_state.actor.params)
#        )
#        for agent, params in zip(env.agents, split_params):
#            safetensors.flax.save_file(
#                flatten_dict(params, sep='/'),
#                f"{config_key}/{agent}.safetensors",
#            )
#
#
## ============================================================================
## EVALUATION PIPELINE
## ============================================================================
#
#def _run_evaluation_pipeline(config: Dict, config_key: str, all_train_states, eval_rng):
#    """
#    Run comprehensive evaluation pipeline for all trained models.
#    
#    Args:
#        config: Configuration dictionary
#        config_key: Unique identifier for this sweep
#        all_train_states: All training states from the sweep
#        eval_rng: Random number generator for evaluation
#        
#    Returns:
#        Mean episode returns across all configurations
#    """
#    # Get architecture-specific evaluation functions
#    _, make_evaluation, EvalInfoLogConfig = _import_mappo_variant(config)
#    
#    # Calculate evaluation batching for memory efficiency
#    batch_dims = jax.tree.leaves(_tree_shape(all_train_states.actor.params))[:3]
#    n_sequential_evals = int(jnp.ceil(
#        config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
#        / config["GPU_ENV_CAPACITY"]
#    ))
#    
#    def _flatten_and_split_trainstate(train_state):
#        """Flatten and split training states for batched evaluation."""
#        flat_trainstate = jax.tree.map(
#            lambda x: x.reshape((x.shape[0]*x.shape[1]*x.shape[2], *x.shape[3:])),
#            train_state
#        )
#        return _tree_split(flat_trainstate, n_sequential_evals)
#    
#    # JIT compile for memory efficiency
#    split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)
#    
#    # Setup evaluation
#    eval_env, run_eval = make_evaluation(config)
#    eval_log_config = EvalInfoLogConfig(
#        env_state=False,
#        done=True,
#        action=False,
#        value=False,
#        reward=True,
#        log_prob=False,
#        obs=False,
#        info=False,
#        avail_actions=False,
#    )
#    
#    # JIT compile evaluation function
#    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
#    eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
#    
#    # Run evaluation in batches
#    print("Running evaluation across all hyperparameter configurations...")
#    evals = _concat_tree([
#        eval_vmap(eval_rng, ts, eval_log_config)
#        for ts in tqdm(split_trainstate, desc="Evaluation batches")
#    ])
#    
#    # Reshape back to original batch dimensions
#    evals = jax.tree.map(
#        lambda x: x.reshape((*batch_dims, *x.shape[1:])),
#        evals
#    )
#    
#    # Compute episode returns
#    first_episode_returns = _compute_episode_returns(evals)
#    mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)
#    
#    # Save evaluation results
#    jnp.save(f"{config_key}/returns.npy", mean_episode_returns)
#    
#    return mean_episode_returns
#
#
## ============================================================================
## MAIN SWEEP FUNCTION
## ============================================================================
#
#@hydra.main(version_base=None, config_path="config", config_name="mappo_sweep")
#def main(config):
#    """
#    Main function for MAPPO hyperparameter sweep.
#    
#    This function orchestrates the entire sweep process:
#    1. Sets up unique experiment directory
#    2. Generates hyperparameter configurations
#    3. Runs training across all combinations
#    4. Saves results and model parameters
#    5. Evaluates all trained models
#    6. Computes and saves performance metrics
#    
#    Args:
#        config: Hydra configuration object
#    """
#    
#    # ========================================================================
#    # SETUP EXPERIMENT DIRECTORY
#    # ========================================================================
#    
#    # Create unique identifier for this sweep
#    config_key = hash(config) % 2**62
#    config_key = urlsafe_b64encode(
#        config_key.to_bytes(
#            (config_key.bit_length()+8)//8,
#            "big", signed=False
#        )
#    ).decode("utf-8").replace("=", "")
#    
#    # Create experiment directory
#    os.makedirs(config_key, exist_ok=True)
#    print(f"Experiment directory: {config_key}")
#    
#    # Convert config to container
#    config = OmegaConf.to_container(config, resolve=True)
#    
#    # ========================================================================
#    # IMPORT ARCHITECTURE-SPECIFIC FUNCTIONS
#    # ========================================================================
#    
#    make_train, make_evaluation, EvalInfoLogConfig = _import_mappo_variant(config)
#    
#    print(f"Using architecture: "
#          f"{'RNN' if config['network']['recurrent'] else 'FF'} + "
#          f"{'PS' if config['network']['agent_param_sharing'] else 'NPS'}")
#    
#    # ========================================================================
#    # GENERATE HYPERPARAMETER SWEEP
#    # ========================================================================
#    
#    rng = jax.random.PRNGKey(config["SEED"])
#    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
#    
#    # Generate multiple random seeds for robust evaluation
#    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
#    
#    # Generate hyperparameter sweep configurations
#    sweep = _generate_sweep_axes(sweep_rng, config)
#    
#    print(f"Sweep configurations:")
#    for param_name, param_info in sweep.items():
#        if param_info["axis"] is not None:
#            print(f"  {param_name}: {len(param_info['val'])} values")
#        else:
#            print(f"  {param_name}: {param_info['val']} (fixed)")
#    
#    # ========================================================================
#    # RUN TRAINING SWEEP
#    # ========================================================================
#    
#    print("Starting training sweep...")
#    
#    with jax.disable_jit(config["DISABLE_JIT"]):
#        # Create JIT-compiled training function
#        train_jit = jax.jit(
#            make_train(config, save_train_state=True),
#            device=jax.devices()[config["DEVICE"]]
#        )
#        
#        # Run training across all hyperparameter combinations
#        # Uses nested vmap for efficient parallel execution
#        out = jax.vmap(
#            jax.vmap(
#                train_jit,
#                in_axes=(0, None, None, None)  # Multiple seeds
#            ),
#            in_axes=(
#                None,                          # Same seeds across configs
#                sweep["lr"]["axis"],           # Sweep learning rates
#                sweep["ent_coef"]["axis"],     # Sweep entropy coefficients
#                sweep["clip_eps"]["axis"],     # Sweep clip epsilons
#            )
#        )(
#            train_rngs,
#            sweep["lr"]["val"],
#            sweep["ent_coef"]["val"],
#            sweep["clip_eps"]["val"],
#        )
#
#    
#    # ========================================================================
#    # SAVE TRAINING RESULTS
#    # ========================================================================
#    
#    _save_training_metrics(config_key, out["metrics"])
#    
#    # Save hyperparameter configurations
#    _save_hyperparameters(config_key, sweep, config)
#    
#    # Save model parameters
#    all_train_states = out["metrics"]["train_state"]
#    final_train_state = out["runner_state"].train_state
#
#    if config["SAVE_ALL_TRAIN_STATES"]:
#        _save_model_parameters(config_key, config, all_train_states, final_train_state)
#    
#    print("\nHyperparameter sweep completed successfully!")
#
#
#if __name__ == "__main__":
#    main()
#
#