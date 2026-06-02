"""
ISAC Hyperparameter Sweep Script

This script performs comprehensive hyperparameter sweeps for ISAC:

Key ISAC Hyperparameters:
=========================
- **Policy Learning Rate**: Actor network optimization rate
- **Q Learning Rate**: Critic networks (Q1, Q2) optimization rate
- **Alpha Learning Rate**: Entropy temperature tuning rate
- **Tau**: Soft target network update coefficient
"""

import os
import time
from typing import Dict, Any
from base64 import urlsafe_b64encode
from pathlib import Path

import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from flax.traverse_util import flatten_dict
import safetensors.flax

import assistax

from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode,
    _tree_shape, _stack_tree, _concat_tree, _tree_split,
    print_memory_stats
    )

from assistax.baselines.sweep_utils import (
    scan_completed_sweeps,
    config_already_run_sac,
    )

from assistax.baselines.utils import _compute_episode_returns_sweep as _compute_episode_returns


# ============================================================================
# HYPERPARAMETER SWEEP CONFIGURATION
# ============================================================================

def _generate_sweep_axes(rng: jax.Array, config: Dict) -> Dict[str, Dict[str, Any]]:
    """
    Generate ISAC hyperparameter sweep configurations.

    For each ISAC hyperparameter, either generates a range of values or uses
    the single value from config.

    Args:
        rng: Random number generator key
        config: Configuration dictionary containing sweep settings

    Returns:
        Dictionary containing sweep axes with values and vmap axes
    """
    p_lr_rng, q_lr_rng, alpha_lr_rng, tau_rng = jax.random.split(rng, 4)
    sweep_config = config["SWEEP"]

    # Generate policy learning rate sweep
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

    # Generate Q-function learning rate sweep
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

    # Generate entropy learning rate sweep
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

    # Generate tau (soft update) sweep
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


# ============================================================================
# MAIN SWEEP FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="isac_sweep")
def main(config):
    """
    Main function for ISAC hyperparameter sweep.

    Orchestrates the complete sweep process for off-policy ISAC algorithm:
    1. Sets up unique experiment directory
    2. Generates ISAC-specific hyperparameter configurations
    3. Runs training across all combinations
    4. Saves results and model parameters
    5. Evaluates all trained models
    6. Computes and saves performance metrics
    """

    # ========================================================================
    # SETUP EXPERIMENT DIRECTORY
    # ========================================================================

    config_key = hash(config) % 2**62
    config_key = urlsafe_b64encode(
        config_key.to_bytes(
            (config_key.bit_length()+8)//8,
            "big", signed=False
        )
    ).decode("utf-8").replace("=", "")

    config = OmegaConf.to_container(config, resolve=True)

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])

    sweep = _generate_sweep_axes(sweep_rng, config)

    base_dir = Path.cwd().parent.parent
    completed = scan_completed_sweeps(base_dir)
    if config_already_run_sac(config, completed, sweep):
        print(f"✓ SKIPPING - already completed:")
        print(f"  rollout_length={config['ROLLOUT_LENGTH']}, batch_size={config['BATCH_SIZE']}")
        return


    os.makedirs(config_key, exist_ok=True)

    print(f"ISAC Hyperparameter Sweep")
    print(f"Experiment ID: {config_key}")

    # ========================================================================
    # IMPORT ISAC FUNCTIONS AND GENERATE SWEEP
    # ========================================================================

    from isac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig

    sweep_info = []
    for param_name, param_info in sweep.items():
        if param_info["axis"] is not None:
            sweep_info.append(f"{param_name}: {len(param_info['val'])} values")
        else:
            sweep_info.append(f"{param_name}: {param_info['val']} (fixed)")

    print(f"Sweep configuration: {', '.join(sweep_info)}")

    # ========================================================================
    # RUN TRAINING SWEEP
    # ========================================================================

    start_time = time.time()

    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )

        # Run ISAC training across all hyperparameter combinations
        out = jax.vmap(
            jax.vmap(
                train_jit,
                in_axes=(0, None, None, None, None)  # Multiple seeds
            ),
            in_axes=(
                None,                                 # Same seeds across configs
                sweep["p_lr"]["axis"],               # Policy learning rates
                sweep["q_lr"]["axis"],               # Q-function learning rates
                sweep["alpha_lr"]["axis"],           # Entropy learning rates
                sweep["tau"]["axis"],                # Soft update coefficients
            )
        )(
            train_rngs,
            sweep["p_lr"]["val"],
            sweep["q_lr"]["val"],
            sweep["alpha_lr"]["val"],
            sweep["tau"]["val"]
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # ====================================================================
        # SAVE TRAINING RESULTS
        # ====================================================================

        # Save training metrics
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

        # Save hyperparameter configurations
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

        # Save model parameters
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        all_train_states = out["metrics"]["actor_train_state"]
        final_train_state = out["runner_state"].train_states.actor

        if config["SAVE_ALL_TRAIN_STATES"]:
            safetensors.flax.save_file(
                flatten_dict(all_train_states.params, sep='/'),
                f"{config_key}/all_params.safetensors"
            )

        if config["SAVE_FINAL_TRAIN_STATE"]:
            if config["network"]["agent_param_sharing"]:
                safetensors.flax.save_file(
                    flatten_dict(final_train_state.params, sep='/'),
                    f"{config_key}/final_params.safetensors"
                )
            else:
                split_params = _unstack_tree(
                    jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), final_train_state.params)
                )
                for agent, params in zip(env.agents, split_params):
                    safetensors.flax.save_file(
                        flatten_dict(params, sep='/'),
                        f"{config_key}/{agent}.safetensors",
                    )

        # ====================================================================
        # RUN EVALUATION
        # ====================================================================

        eval_start_time = time.time()

        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))

        def _flatten_and_split_trainstate(train_state):
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0]*x.shape[1]*x.shape[2], *x.shape[3:])),
                train_state
            )
            return _tree_split(flat_trainstate, n_sequential_evals)

        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)

        eval_env, run_eval = make_evaluation(config)
        eval_log_config = EvalInfoLogConfig(
            env_state=False,
            done=True,
            action=False,
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

        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        first_episode_returns = _compute_episode_returns(evals)
        mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)

        jnp.save(f"{config_key}/returns.npy", mean_episode_returns)

        evaluation_time = time.time() - eval_start_time

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    total_time = time.time() - start_time

    # Performance statistics
    best_return = float(jnp.max(mean_episode_returns))
    worst_return = float(jnp.min(mean_episode_returns))
    mean_return = float(jnp.mean(mean_episode_returns))

    print(f"\nSweep completed in {total_time:.2f}s (train: {training_time:.2f}s, eval: {evaluation_time:.2f}s)")
    print(f"Performance: best={best_return:.4f}, mean={mean_return:.4f}, worst={worst_return:.4f}")

    ## Find best hyperparameter configuration
    #best_config_idx = jnp.unravel_index(
    #    jnp.argmax(mean_episode_returns), mean_episode_returns.shape
    #)

    #breakpoint()
    #best_config_info = []
    #if sweep["p_lr"]["axis"] is not None:
    #    best_config_info.append(f"p_lr={sweep['p_lr']['val'][best_config_idx[0]]:.6f}")
    #if sweep["q_lr"]["axis"] is not None:
    #    best_config_info.append(f"q_lr={sweep['q_lr']['val'][best_config_idx[1]]:.6f}")
    #if sweep["alpha_lr"]["axis"] is not None:
    #    best_config_info.append(f"alpha_lr={sweep['alpha_lr']['val'][best_config_idx[2]]:.6f}")
    #if sweep["tau"]["axis"] is not None:
    #    best_config_info.append(f"tau={sweep['tau']['val'][best_config_idx[3]]:.6f}")

    #if best_config_info:
    #    print(f"Best config: {', '.join(best_config_info)}")

    #print(f"Results saved to: {config_key}/")

    print(f"✓ Sweep completed successfully")

    if config.get("PRINT_MEMORY_STATS", False):
        print_memory_stats(f"ISAC Sweep: Final Network=ISAC_FF_NPS, Env={config['ENV_NAME']}, Seeds={config['NUM_SEEDS']}, Num Envs={config['NUM_ENVS']}, Batch_size={config['BATCH_SIZE']}")


if __name__ == "__main__":
    main()
