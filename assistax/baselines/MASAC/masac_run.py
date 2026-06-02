"""
MASAC Training, Evaluation, and Visualization Runner

This module serves as the main orchestration script for running Multi-Agent Soft Actor-Critic
(MASAC) experiments. It handles training execution with separate actor and critic networks,
parameter saving for all network components, evaluation, result analysis, and optional
trajectory visualization.

Usage:
    python masac_run.py [hydra options]
    
The script trains MASAC agents with independent actor and dual Q-networks per agent,
saves all network parameters separately, runs comprehensive evaluation, and optionally
generates interactive HTML visualizations of agent behaviors.
"""

import os
import sys
import time
from datetime import datetime
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
import wandb
import assistax
from assistax.wrappers.baselines import get_space_dim, LogEnvState
from assistax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Callable
from flax import struct
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns,
    _tree_shape, _stack_tree, _concat_tree, _tree_split,
    upload_eval_data_to_wandb, log_all_metrics, upload_model_parameters_to_wandb,
    print_memory_stats
    )
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ' # As recommended by MJX for better performance on NVIDIA GPUs
)


# ================================ EVALUATION DATA STRUCTURES ================================

@struct.dataclass
class EvalNetworkState:
    """Network state for MASAC evaluation with pre-trained models."""
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict

# ================================ MAIN ORCHESTRATION FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="masac")
def main(config):
    """
    Main orchestration function for MASAC training and evaluation.
    
    This function:
    1. Imports MASAC implementation with separate actor and critic networks
    2. Runs training with SAC-specific hyperparameters (policy_lr, q_lr, alpha_lr, tau)
    3. Saves all network parameters separately (actor, q1, q2 networks)
    4. Evaluates trained agents and computes performance metrics
    5. Optionally creates interactive HTML visualizations via separate script
    
    Args:
        config: Hydra configuration object containing all hyperparameters
    """
    print("Starting MASAC training and evaluation...")
    config = OmegaConf.to_container(config, resolve=True)

    # ===== WANDB LOGGING =====
    now = datetime.now()
    param_sharing = config["network"]["agent_param_sharing"]
    ps_tag = "ps" if param_sharing else "nps"
    rec_config = config["network"]["recurrent"]
    rec_tag = "rnn" if rec_config else "ff"

    env_name = (
        config.get("ENV_NAME")
        if config.get("MAP_NAME") is None
        else config.get("MAP_NAME")
    )
    env_name = env_name.lower()
    alg_name = config.get("ALG").lower()
    name = f"{alg_name}_{ps_tag}_{rec_tag}_{env_name}_{config['EXP_ID']}_seed{config['SEED']}"
    tags = [config["EXP_ID"]] + config.get("EXP_TAGS") + [env_name] + [alg_name]
    config["EXP_TAGS"] = tags
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=tags,
        config=config,
        mode=config["WANDB_MODE"],
        reinit=True,
        name=name,
        save_code=True,
    )

    # ===== ALGORITHM IMPORTS =====
    # MASAC uses Multi SAC Actor with separate Q-networks
    from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
    from masac_ff_nps import MultiSACActor as NetworkArch
    print("Using: Multi-Agent Soft Actor-Critic with separate actor and dual Q-networks")

    # ===== TRAINING SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])

    print(f"Starting training with {config['TOTAL_TIMESTEPS']} timesteps")
    print(f"Num environments: {config['NUM_ENVS']}")
    print(f"Num seeds: {config['NUM_SEEDS']}")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"SAC Hyperparameters:")
    print(f"  Policy LR: {config['POLICY_LR']}")
    print(f"  Q LR: {config['Q_LR']}")
    print(f"  Alpha LR: {config['ALPHA_LR']}")
    print(f"  Tau (soft update): {config['TAU']}")

    # ===== TRAINING EXECUTION =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        
        train_jit = jax.jit(
            make_train(config, save_train_state=True, load_zoo=False),
            device=jax.devices()[config["DEVICE"]]
        )
        
        # Execute MASAC training across all seeds with SAC-specific hyperparameters
        print("Running MASAC training...")
        out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
            train_rngs,
            config["POLICY_LR"], config["Q_LR"], config["ALPHA_LR"], config["TAU"]
        )

        # ===== SAVE TRAINING METRICS =====
        print("Saving training metrics...")
        # Exclude large training states from metrics file (SAC has 3 networks)
        EXCLUDED_METRICS = ["actor_train_state", "q1_train_state", "q2_train_state"]
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
        }, allow_pickle=True)

        # ===== SAVE MODEL PARAMETERS =====
        print("Saving MASAC model parameters...")
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        # Extract training states for all three networks (actor, q1, q2)
        all_train_states_actor = out["metrics"]["actor_train_state"]
        all_train_states_q1 = out["metrics"]["q1_train_state"]
        all_train_states_q2 = out["metrics"]["q2_train_state"]
        
        final_train_state_actor = out["runner_state"].train_states.actor
        final_train_state_q1 = out["runner_state"].train_states.q1
        final_train_state_q2 = out["runner_state"].train_states.q2

        # Save all training states across training (for analysis)
        print("Saving complete training history for all networks...")
        actor_all_path = "actor_all_params.safetensors"
        safetensors.flax.save_file(
            flatten_dict(all_train_states_actor.params, sep='/'),
            actor_all_path
        )
        # Get absolute path for potential rendering script
        actor_all_path = os.path.abspath(actor_all_path)
        
        safetensors.flax.save_file(
            flatten_dict(all_train_states_q1.params, sep='/'),
            "q1_all_params.safetensors"
        )
        safetensors.flax.save_file(
            flatten_dict(all_train_states_q2.params, sep='/'),
            "q2_all_params.safetensors"
        )

        # Save final parameters (different format for parameter sharing vs independent)
        print("Saving final network parameters...")
        if config["network"]["agent_param_sharing"]:
            # For parameter sharing: single set of shared parameters for each network
            print("Saving shared parameters for all networks...")
            safetensors.flax.save_file(
                flatten_dict(final_train_state_actor.params, sep='/'),
                "actor_final_params.safetensors"
            )
            safetensors.flax.save_file(
                flatten_dict(final_train_state_q1.params, sep='/'),
                "q1_final_params.safetensors"
            )
            safetensors.flax.save_file(
                flatten_dict(final_train_state_q2.params, sep='/'),
                "q2_final_params.safetensors"
            )
        else:
            # For independent parameters: split by agent for each network
            print("Saving agent-specific parameters for all networks...")
            
            # Split and save actor parameters by agent
            split_actor_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_actor.params)
            )
            for agent, params in zip(env.agents, split_actor_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"actor_{agent}.safetensors",
                )

            # Split and save Q1 parameters by agent
            split_q1_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_q1.params)
            )
            for agent, params in zip(env.agents, split_q1_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q1_{agent}.safetensors",
                )

            # Split and save Q2 parameters by agent
            split_q2_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_q2.params)
            )
            for agent, params in zip(env.agents, split_q2_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q2_{agent}.safetensors",
                )

        # ===== EVALUATION SETUP =====
        print("Setting up MASAC evaluation...")
        
        # Calculate evaluation batching for memory efficiency
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states_actor.params))[:2]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        
        def _flatten_and_split_trainstate(trainstate):
            """
            Flatten training states across batch dimensions and split for sequential evaluation.
            
            This operation is JIT compiled for memory efficiency during evaluation.
            For MASAC, we only need the actor network for policy evaluation.
            """
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
                trainstate
            )
            return _tree_split(flat_trainstate, n_sequential_evals)

        # Use actor network for evaluation (Q-networks not needed for rollouts)
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states_actor)
        
        # ===== EVALUATION EXECUTION =====
        print("Running MASAC evaluation...")
        eval_env, run_eval = make_evaluation(config)
        
        # Configure what information to log during evaluation
        eval_log_config = EvalInfoLogConfig(
            env_state=False,       # Don't need environment states for performance metrics
            done=True,             # Need done flags for episode boundary detection
            action=False,          # Don't need actions for performance analysis
            reward=True,           # Need rewards for return computation
            log_prob=False,        # Don't need log probabilities
            obs=False,             # Don't need observations
            info=False,            # Don't need environment info
            avail_actions=False,   # Don't need available actions
            env_metrics=True,      # Need environment metrics for wandb logging
        )
        
        # JIT compile evaluation functions for efficiency
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
        
        # Run evaluation in batches for memory efficiency
        print("Executing evaluation batches...")
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
        print("Computing MASAC performance metrics...")
        first_episode_returns = _compute_episode_returns(evals)
        first_episode_returns = first_episode_returns["__all__"]
        mean_episode_returns = first_episode_returns.mean(axis=-1)

        # ===== SAVE EVALUATION RESULTS =====
        print("Saving evaluation results...")
        jnp.save("returns.npy", mean_episode_returns)

        # ===== WANDB LOGGING =====
        print("Logging metrics to wandb...")
        log_all_metrics(config, out, evals, env)
        upload_eval_data_to_wandb(evals, config, run)
        upload_model_parameters_to_wandb(all_train_states_actor, final_train_state_actor, config, env, run)

        if config.get("PRINT_MEMORY_STATS", False):
            print_memory_stats(f"MASAC: Env={config['ENV_NAME']}, Seeds={config['NUM_SEEDS']}, Num Envs={config['NUM_ENVS']}")

        print(f"MASAC Performance Summary:")
        print(f"  Mean episode return: {mean_episode_returns.mean():.2f} ± {mean_episode_returns.std():.2f}")
        print("MASAC training and evaluation completed successfully!")

        # ===== OPTIONAL VISUALIZATION =====
        if config["VIZ_POLICY"]:
            print("Launching trajectory visualization...")
            
            # Get current directory and script paths for visualization
            current_output_dir = os.getcwd()
            script_directory = os.path.dirname(os.path.abspath(__file__))
            render_script_path = os.path.join(script_directory, "render_masac.py")
            
            print(f"Executing visualization script: {render_script_path}")
            print(f"Using actor parameters: {actor_all_path}")
            print(f"Rendering {config['N_RENDER_EPISODES']} episodes...")
            
            # Execute separate rendering script with appropriate parameters
            # Note: Using os.execv to replace current process with rendering script
            os.execv(sys.executable, 
                    [sys.executable,
                    render_script_path,
                    f'eval.path={actor_all_path}',
                    f'hydra.run.dir={current_output_dir}',
                    f'NUM_EVAL_EPISODES={config["N_RENDER_EPISODES"]}',
                    ])


if __name__ == "__main__":
    main()
