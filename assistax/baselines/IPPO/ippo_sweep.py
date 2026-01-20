"""
IPPO Hyperparameter Sweeping 

This module orchestrates large-scale hyperparameter sweeps for IPPO experiments across different
network architectures. It systematically explores hyperparameter spaces, manages experiment
organization, and handles efficient evaluation of multiple configurations simultaneously.

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
from datetime import datetime
import wandb

from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode,
    _tree_shape, _stack_tree, _concat_tree, _tree_split, upload_eval_data_to_wandb, 
    log_all_metrics, upload_html_visualizations_to_wandb, upload_model_parameters_to_wandb,
    upload_mujoco_trajectories_to_wandb, upload_mujoco_videos_to_wandb, print_memory_stats,
    log_memory_to_csv,
    )

from assistax.baselines.utils import _compute_episode_returns_sweep as _compute_episode_returns

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ' # As recommended by MJX for better performance on NVIDIA GPUs
)

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
            network_type = "FF_NPS"
        case (False, True):
            from ippo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Feedforward Networks with Parameter Sharing")
            network_type = "FF_PS"
        case (True, False):
            from ippo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with No Parameter Sharing")
            network_type = "RNN_NPS"
        case (True, True):
            from ippo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with Parameter Sharing")
            network_type = "RNN_PS"
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
        #if config["PRINT_MEMORY_STATS"]:
        #    print_memory_stats(f"IPPO Sweep: Training Network={network_type}, Env={config['ENV_NAME']}, Seeds={config['NUM_SEEDS']}, Num Envs={config['NUM_ENVS']},  Num Steps={config['NUM_STEPS']}")
       
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"]) # this could be inefficient memory wise
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
        # Save all training states (for analysis across training)
        
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state

        if config["SAVE_ALL_TRAIN_STATES"]: 
            print("Saving model parameters...")
            
            safetensors.flax.save_file(
                flatten_dict(all_train_states.params, sep='/'),
                f"{config_key}/all_params.safetensors"
            )

        # Save final parameters (different format for parameter sharing vs independent)
        if config["SAVE_FINAL_TRAIN_STATE"]:
            
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
         
        #if config["PRINT_MEMORY_STATS"]:
        #    print_memory_stats(f"IPPO Sweep: Pre-Eval Network={network_type}, Env={config['ENV_NAME']}, Seeds={config['NUM_SEEDS']}, Num Envs={config['NUM_ENVS']},  Num Steps={config['NUM_STEPS']}")
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

        if config["PRINT_MEMORY_STATS"]:
            print_memory_stats(f"IPPO Sweep: Final Network={network_type}, Env={config['ENV_NAME']}, Seeds={config['NUM_SEEDS']}, Num Envs={config['NUM_ENVS']},  Num Steps={config['NUM_STEPS']}")
            log_memory_to_csv(config, "/home/leo/assistive-autonomy-github/assistax/memory_stats", f"{config['ALG']}_memory_stats.csv")
            



if __name__ == "__main__":
    main()
