"""
MAPPO Training, Evaluation, and Visualization Runner

This module serves as the main orchestration script for running MAPPO experiments across different
network architectures. It dynamically imports the appropriate MAPPO variant based on configuration
settings, handles training execution, parameter saving, evaluation, and result visualization.

Usage:
    python mappo_run.py [hydra options] e.g. network=ff_nps
    
The script will automatically select the correct algorithm variant based on the config
values for network architecture (recurrent/feedforward and parameter sharing).
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
from assistax.wrappers.baselines import get_space_dim, LogEnvState
from assistax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns,
    _tree_shape, _stack_tree, _concat_tree, _tree_split
    )

# ================================ MAIN ORCHESTRATION FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="mappo")
def main(config):
    """
    Main orchestration function for MAPPO training and evaluation.
    
    This function:
    1. Dynamically imports the correct MAPPO variant based on config
    2. Runs training with specified hyperparameters
    3. Saves model parameters and training metrics
    4. Evaluates trained agents and computes performance metrics
    5. Creates interactive HTML visualizations of episodes
    
    Args:
        config: Hydra configuration object containing all hyperparameters
    """
    config = OmegaConf.to_container(config, resolve=True)
    
    # ===== DYNAMIC ALGORITHM SELECTION =====
    # Import the appropriate MAPPO variant based on network architecture configuration
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from mappo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Feedforward Networks with No Parameter Sharing")
        case (False, True):
            from mappo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Feedforward Networks with Parameter Sharing")
        case (True, False):
            from mappo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with No Parameter Sharing")
        case (True, True):
            from mappo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with Parameter Sharing")

    # ===== TRAINING SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    print(f"Starting training with {config['TOTAL_TIMESTEPS']} timesteps")
    print(f"Num environments: {config['NUM_ENVS']}")
    print(f"Num seeds: {config['NUM_SEEDS']}")
    print(f"Environment: {config['ENV_NAME']}")
    
    # ===== TRAINING EXECUTION =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        
        # Execute training across all seeds (includes JIT compilation on first run)
        print("Running training...")
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )

        # ===== SAVE TRAINING METRICS =====
        print("Saving training metrics...")
        EXCLUDED_METRICS = ["train_state"]  # Exclude large training states from metrics file
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )

        # ===== SAVE MODEL PARAMETERS =====
        print("Saving model parameters...")
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state

        # Save all training states (for analysis across training)
        safetensors.flax.save_file(
            flatten_dict(all_train_states.actor.params, sep='/'),
            "all_params.safetensors"
        )
        
        # Save final parameters (different format for parameter sharing vs independent)
        if config["network"]["agent_param_sharing"]:
            # For parameter sharing: single set of shared parameters
            safetensors.flax.save_file(
                flatten_dict(final_train_state.actor.params, sep='/'),
                "final_params.safetensors"
            )
        else:
            # For independent parameters: split by agent
            split_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state.actor.params)
            )
            for agent, params in zip(env.agents, split_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"{agent}.safetensors",
                )

        # ===== EVALUATION SETUP =====
        print("Setting up evaluation...")
        
        # Calculate evaluation batching for memory efficiency
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.actor.params))[:2]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        
        def _flatten_and_split_trainstate(trainstate):
            """
            Flatten training states across batch dimensions and split for sequential evaluation.
            
            This operation is JIT compiled for memory efficiency during evaluation.
            """
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
                trainstate
            )
            return _tree_split(flat_trainstate, n_sequential_evals)

        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states.actor)
        
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

        # Reshape evaluation results back to original batch structure
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # ===== COMPUTE PERFORMANCE METRICS =====
        print("Computing performance metrics...")
        first_episode_returns = _compute_episode_returns(evals)
        first_episode_returns = first_episode_returns["__all__"]
        mean_episode_returns = first_episode_returns.mean(axis=-1)

        # Save evaluation results
        jnp.save("returns.npy", mean_episode_returns)
        print(f"Mean episode return: {mean_episode_returns.mean():.2f} ± {mean_episode_returns.std():.2f}")

        # ===== VISUALIZATION AND RENDERING =====
        print("Creating episode visualizations...")
        
        # Run episodes for rendering (saving env_state at each timestep)
        render_log_config = EvalInfoLogConfig(
            env_state=True,  # Need environment state for visualization
            done=True,
            action=False,
            value=False,
            reward=True,
            log_prob=False,
            obs=False,
            info=False,
            avail_actions=False,
        )
        
        # Evaluate final model for visualization
        eval_final = eval_jit(eval_rng, _tree_take(final_train_state.actor, 0, axis=0), render_log_config)
        
        # Compute episode returns and select representative episodes
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        first_episode_rewards = eval_final.reward["__all__"] * (1 - first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        
        # Select worst, median, and best performing episodes
        worst_idx = episode_argsort.take(0, axis=-1)
        best_idx = episode_argsort.take(-1, axis=-1)
        median_idx = episode_argsort.take(episode_argsort.shape[-1] // 2, axis=-1)

        # Extract episode data for visualization
        from assistax.render import html
        
        worst_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=worst_idx,
        )
        median_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=median_idx,
        )
        best_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=best_idx,
        )
        
        # Generate interactive HTML visualizations
        html.save("final_worst.html", eval_env.sys, worst_episode)
        html.save("final_median.html", eval_env.sys, median_episode)
        html.save("final_best.html", eval_env.sys, best_episode)
        
        print("Visualizations saved:")
        print("  - final_worst.html: Worst performing episode")
        print("  - final_median.html: Median performing episode") 
        print("  - final_best.html: Best performing episode")
        
        print("\nTraining and evaluation completed successfully!")


if __name__ == "__main__":
    main()


