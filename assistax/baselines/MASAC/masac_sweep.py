"""
MASAC Hyperparameter Sweep Runner

This module orchestrates comprehensive hyperparameter sweeps for Multi-Agent Soft Actor-Critic (MASAC).
It automatically generates random hyperparameter configurations, runs training across all combinations,
and evaluates the resulting policies to identify optimal hyperparameter settings.

Usage:
    python masac_sweep.py [hydra options]
    
The sweep configuration is specified in the config file under the "SWEEP" section.
Each hyperparameter to sweep should specify "min" and "max" log10 values.
"""

import os
from pathlib import Path
import jax
import jax.numpy as jnp
import hydra
import safetensors.flax
from tqdm import tqdm
from flax.traverse_util import flatten_dict
from omegaconf import OmegaConf
from base64 import urlsafe_b64encode
from typing import Dict, Any, Optional
import assistax
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode,
    _tree_shape, _stack_tree, _concat_tree, _tree_split, 
    print_memory_stats
    )
from assistax.baselines.sweep_util import scan_completed_sweeps, config_already_run_sac
from assistax.baselines.utils import _compute_episode_returns_sweep as _compute_episode_returns

# ================================ HYPERPARAMETER SWEEP UTILITIES ================================

def _generate_sweep_axes(rng, config):
    """
    Generate hyperparameter configurations for sweep.
    
    Creates random samples for each hyperparameter marked for sweeping in the
    configuration. Uses log-uniform sampling for learning rates and tau to
    cover multiple orders of magnitude effectively.
    
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
        print(f"Policy LR sweep: {p_lrs.min():.2e} - {p_lrs.max():.2e}") # TODO wierd printing probs nicer to just print all?
    else:
        p_lrs = config["POLICY_LR"]
        p_lr_axis = None
        print(f"Fixed Policy LR: {p_lrs}")

    # ===== Q-NETWORK LEARNING RATE SWEEP =====
    if sweep_config.get("q_lr", False):
        q_lrs = 10**jax.random.uniform(
            q_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["q_lr"]["min"],
            maxval=sweep_config["q_lr"]["max"],
        )
        q_lr_axis = 0
        print(f"Q-Network LR sweep: {q_lrs.min():.2e} - {q_lrs.max():.2e}")
    else:
        q_lrs = config["Q_LR"]
        q_lr_axis = None
        print(f"Fixed Q-Network LR: {q_lrs}")

    # ===== TEMPERATURE LEARNING RATE SWEEP =====
    if sweep_config.get("alpha_lr", False):
        alpha_lrs = 10**jax.random.uniform(
            alpha_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["alpha_lr"]["min"],
            maxval=sweep_config["alpha_lr"]["max"],
        )
        alpha_lr_axis = 0
        print(f"Temperature LR sweep: {alpha_lrs.min():.2e} - {alpha_lrs.max():.2e}")
    else:
        alpha_lrs = config["ALPHA_LR"]
        alpha_lr_axis = None
        print(f"Fixed Temperature LR: {alpha_lrs}")

    # ===== TAU (SOFT UPDATE) SWEEP =====
    if sweep_config.get("tau", False):
        taus = 10**jax.random.uniform(
            tau_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["tau"]["min"],
            maxval=sweep_config["tau"]["max"],
        )
        tau_axis = 0
        print(f"Tau sweep: {taus.min():.2e} - {taus.max():.2e}")
    else:
        taus = config["TAU"]
        tau_axis = None
        print(f"Fixed Tau: {taus}")

    return {
        "p_lr": {"val": p_lrs, "axis": p_lr_axis},
        "q_lr": {"val": q_lrs, "axis": q_lr_axis},
        "alpha_lr": {"val": alpha_lrs, "axis": alpha_lr_axis},
        "tau": {"val": taus, "axis": tau_axis},
    }


def _create_unique_directory(config):
    """
    Create a unique directory for this sweep run based on config hash.
    
    Uses the configuration hash to create a reproducible but unique
    directory name for storing sweep results.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        String path to the created directory
    """
    config_key = hash(config) % 2**62
    config_key = urlsafe_b64encode(
        config_key.to_bytes(
            (config_key.bit_length() + 8) // 8,
            "big", signed=False
        )
    ).decode("utf-8").replace("=", "")
    
    os.makedirs(config_key, exist_ok=True)
    print(f"Created sweep directory: {config_key}")
    return config_key


# ================================ MAIN SWEEP ORCHESTRATION ================================

@hydra.main(version_base=None, config_path="config", config_name="masac_sweep")
def main(config):
    """
    Main orchestration function for MASAC hyperparameter sweep.
    
    This function:
    1. Creates a unique directory for the sweep run
    2. Generates hyperparameter configurations for sweeping
    3. Runs training across all hyperparameter combinations
    4. Saves training metrics and model parameters
    5. Evaluates all trained models
    6. Saves comprehensive results for analysis
    
    Args:
        config: Hydra configuration object containing sweep parameters
    """
    # ===== SETUP UNIQUE DIRECTORY =====
    config_key = _create_unique_directory(config)
    config = OmegaConf.to_container(config, resolve=True)

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
   
    sweep = _generate_sweep_axes(sweep_rng, config)
    
    base_dir = Path.cwd().parent.parent
    completed = scan_completed_sweeps(base_dir)
    if config_already_run_sac(config, completed, sweep):
        print(f"✓ SKIPPING - already completed:")
        print(f"  rollout_length={config['ROLLOUT_LENGTH']}, batch_size={config['BATCH_SIZE']}, num_updates={config['NUM_SAC_UPDATES']}")
        return
    
    print(f"Starting MASAC hyperparameter sweep")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Number of configurations: {config['SWEEP']['num_configs']}")
    print(f"Number of seeds per config: {config['NUM_SEEDS']}")
    
    # ===== IMPORT ALGORITHM COMPONENTS =====
    from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
    
    # ===== RANDOM NUMBER GENERATOR SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    # ===== GENERATE HYPERPARAMETER SWEEP =====
    print("Generating hyperparameter configurations...")
    sweep = _generate_sweep_axes(sweep_rng, config)
    
    # ===== TRAINING EXECUTION =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        print("Compiling training function...")
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        
        print("Running training across all hyperparameter configurations...")
        # Run training with nested vmap across seeds and hyperparameter configurations
        out = jax.vmap(
            jax.vmap(
                train_jit,
                in_axes=(0, None, None, None, None)  # vmap over seeds
            ),
            in_axes=(  # vmap over hyperparameter configurations
                None,  # train_rngs (same for all configs)
                sweep["p_lr"]["axis"],      # policy learning rate
                sweep["q_lr"]["axis"],      # Q-network learning rate
                sweep["alpha_lr"]["axis"],  # temperature learning rate
                sweep["tau"]["axis"],       # soft update coefficient
            )
        )(
            train_rngs,
            sweep["p_lr"]["val"],
            sweep["q_lr"]["val"],
            sweep["alpha_lr"]["val"],
            sweep["tau"]["val"]
        )

        # ===== SAVE TRAINING METRICS =====
        print("Saving training metrics...")
        EXCLUDED_METRICS = ["actor_train_state", "q1_train_state", "q2_train_state"]
        saveable_metrics = {
            key: val.copy() 
            for key, val in out["metrics"].items() 
            if key not in EXCLUDED_METRICS
        }
        
        jnp.save(f"{config_key}/metrics.npy", {
            key: val
            for key, val in saveable_metrics.items()
            if key not in EXCLUDED_METRICS
        }, allow_pickle=True)

        # ===== SAVE HYPERPARAMETER CONFIGURATIONS =====
        print("Saving hyperparameter configurations...")
        jnp.save(f"{config_key}/hparams.npy", {
            "p_lr": sweep["p_lr"]["val"],
            "q_lr": sweep["q_lr"]["val"],
            "alpha_lr": sweep["alpha_lr"]["val"],
            "tau": sweep["tau"]["val"],
            "num_updates": config["NUM_UPDATES"],
            "total_timesteps": config["TOTAL_TIMESTEPS"],
            "num_envs": config["NUM_ENVS"],
            "num_sac_updates": config["NUM_SAC_UPDATES"],
            "batch_size": config["BATCH_SIZE"],
            "buffer_size": config["BUFFER_SIZE"],
            "rollout_length": config["ROLLOUT_LENGTH"],
            "explore_steps": config["EXPLORE_STEPS"],
        })

        # ===== SAVE MODEL PARAMETERS =====
        print("Saving model parameters...")
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        all_train_states = out["metrics"]["actor_train_state"]
        final_train_state = out["runner_state"].train_states.actor

        if config["SAVE_ALL_TRAIN_STATES"]:
            # Save all training states (for analysis across training)
            safetensors.flax.save_file(
                flatten_dict(all_train_states.params, sep='/'),
                f"{config_key}/all_params.safetensors"
            )

        if config["SAVE_FINAL_TRAIN_STATE"]:
            # Save final parameters (different format for parameter sharing vs independent)
            if config["network"]["agent_param_sharing"]:
                # For parameter sharing: single set of shared parameters
                safetensors.flax.save_file(
                    flatten_dict(final_train_state.params, sep='/'),
                    f"{config_key}/final_params.safetensors"
                )
            else:
                # For independent parameters: split by agent
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
        # Note: 3 batch dimensions for sweep (configs, seeds, checkpoints)
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        
        print(f"Batch dimensions: {batch_dims}")
        print(f"Sequential evaluation batches: {n_sequential_evals}")

        def _flatten_and_split_trainstate(train_state):
            """
            Flatten training states across all batch dimensions and split for sequential evaluation.
            
            For sweep evaluation, we have 3 batch dimensions:
            - Hyperparameter configurations
            - Random seeds  
            - Training checkpoints
            
            Args:
                train_state: Training state with shape (num_configs, num_seeds, num_checkpoints, ...)
                
            Returns:
                List of training state chunks for sequential evaluation
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
        
        # JIT compile evaluation functions
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
        
        # Reshape evaluation results back to original batch structure
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # ===== COMPUTE PERFORMANCE METRICS =====
        print("Computing performance metrics...")
        first_episode_returns = _compute_episode_returns(evals)
        mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)

        # ===== SAVE EVALUATION RESULTS =====
        print("Saving evaluation results...")
        jnp.save(f"{config_key}/returns.npy", mean_episode_returns)
        
        # ===== DISPLAY SUMMARY STATISTICS =====
        print("\n" + "="*60)
        print("SWEEP RESULTS SUMMARY")
        print("="*60)
        
        # Find best configuration
        best_config_idx = jnp.unravel_index(
            jnp.argmax(mean_episode_returns), 
            mean_episode_returns.shape
        )[0]
        
        best_return = mean_episode_returns.max()
        worst_return = mean_episode_returns.min()
        mean_return = mean_episode_returns.mean()
        
        print(f"Total configurations evaluated: {config['SWEEP']['num_configs']}")
        print(f"Seeds per configuration: {config['NUM_SEEDS']}")
        print(f"")
        print(f"Best configuration (index {best_config_idx}):")
        print(f"  Policy LR: {sweep['p_lr']['val'][best_config_idx] if sweep['p_lr']['axis'] is not None else sweep['p_lr']['val']:.2e}")
        print(f"  Q-Network LR: {sweep['q_lr']['val'][best_config_idx] if sweep['q_lr']['axis'] is not None else sweep['q_lr']['val']:.2e}")
        print(f"  Temperature LR: {sweep['alpha_lr']['val'][best_config_idx] if sweep['alpha_lr']['axis'] is not None else sweep['alpha_lr']['val']:.2e}")
        print(f"  Tau: {sweep['tau']['val'][best_config_idx] if sweep['tau']['axis'] is not None else sweep['tau']['val']:.2e}")
        print(f"  Return: {best_return:.2f}")
        print(f"")
        print(f"Performance Statistics:")
        print(f"  Best return: {best_return:.2f}")
        print(f"  Worst return: {worst_return:.2f}")
        print(f"  Mean return: {mean_return:.2f}")
        print(f"  Std return: {mean_episode_returns.std():.2f}")
        print(f"")
        print(f"Results saved to directory: {config_key}")
        print("="*60)
        
        print("Hyperparameter sweep completed successfully!")

        network_type = "MASAC_FF_NPS"

        if config["PRINT_MEMORY_STATS"]:
            print_memory_stats(f"MASAC Sweep: Final Network={network_type}, Env={config['ENV_NAME']}, Seeds={config['NUM_SEEDS']}, Num Envs={config['NUM_ENVS']}, Batch_size={config['BATCH_SIZE']}")



if __name__ == "__main__":
    main()
