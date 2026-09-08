"""
IPPO Training, Evaluation, and Visualization Runner

This module serves as the main orchestration script for running IPPO experiments across different
network architectures. It dynamically imports the appropriate IPPO variant based on configuration
settings, handles training execution, parameter saving, evaluation, and result visualization.

Usage:
    python ippo_run.py [hydra options] e.g. network=ff_nps
    
The script will automatically select the correct algorithm variant based on the config
values for network 
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl') # Use EGL backend for offscreen rendering in MuJoCo
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
from omegaconf import DictConfig, OmegaConf
from typing import Sequence, NamedTuple, Any, Dict
import wandb
from datetime import datetime
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns,
    _tree_shape, _stack_tree, _concat_tree, _tree_split, upload_eval_data_to_wandb, 
    log_all_metrics, upload_html_visualizations_to_wandb, upload_model_parameters_to_wandb,
    upload_mujoco_trajectories_to_wandb, upload_mujoco_videos_to_wandb, print_memory_stats
    )

os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=True ' # As recommended by MJX for better performance on NVIDIA GPUs
)


# ================================ MAIN ORCHESTRATION FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="ippo")
def main(config: DictConfig):
    """
    Main orchestration function for IPPO training and evaluation.
    
    This function:
    1. Dynamically imports the correct IPPO variant based on config
    2. Runs training with specified hyperparameters
    3. Saves model parameters and training metrics
    4. Evaluates trained agents and computes performance metrics
    5. Creates interactive HTML visualizations of episodes
    
    Args:
        config: Hydra configuration object containing all hyperparameters
    """
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
            network_type = "FF_NPS"
        case (True, False):
            from ippo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with No Parameter Sharing")
            network_type = "RNN_NPS"
        case (True, True):
            from ippo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
            print("Using: Recurrent Networks with Parameter Sharing")

    # WANDB logging
    now = datetime.now()
    param_sharing = config["network"]["agent_param_sharing"]
    if param_sharing:
        ps_tag = "ps"
    else:
        ps_tag = "nps"
    rec_config = config["network"]["recurrent"]
    if rec_config:
        rec_tag = "rnn"
    else:
        rec_tag = "ff"

    env_name = (
        config.get("ENV_NAME")
        if config.get("MAP_NAME") is None
        else config.get("MAP_NAME")
    )
    env_name = env_name.lower()
    alg_name = config.get("ALG").lower()
    name = f"{alg_name}_{ps_tag}_{rec_tag}_{env_name}_{config['EXP_ID']}_seed{config['SEED']}"
    tags = [config["EXP_ID"]] + config.get("EXP_TAGS") + [env_name] + [alg_name]
    config["EXP_TAGS"] = tags  # Update config with full tags list for easier grouping in WandB
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
    # ===== TRAINING SETUP =====
    #rng = jax.random.PRNGKey(config["SEED"])
    rng = jax.random.key(config["SEED"]) # TODO update to new jax API
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    print(f"Starting training with {config['TOTAL_TIMESTEPS']} timesteps")
    print(f"Num environments: {config['NUM_ENVS']}")
    print(f"Num seeds: {config['NUM_SEEDS']}")
    print(f"Environment: {config['ENV_NAME']}")
    
    # ===== TRAINING EXECUTION =====
    start = time.time()
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
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        EXCLUDED_METRICS = ["train_state"]  # Exclude large training states from metrics file

        # TODO here I really should use something like log_multiple_training_seeds 
        # ===== SAVE MODEL PARAMETERS =====
        print("Saving model parameters...")
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state

        # Uploading model parameters to WandB 
        
        upload_model_parameters_to_wandb(all_train_states, final_train_state, config, env, run)
        
        # ===== EVALUATION SETUP =====
        print("Setting up evaluation...")
        
        # Calculate evaluation batching for memory efficiency
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:2]
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
            env_metrics=True,
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

        
        
        # TODO Save evaluation results with wandb utility
        log_all_metrics(config, out, evals, env)
        upload_eval_data_to_wandb(evals, config, run)
        # jnp.save("returns.npy", mean_episode_returns)

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
        # TODO: limit to fewer evaluation episodes to make rendering more memory efficient
        print("Freeing memory before rendering...")
        del out
        del evals  
        del all_train_states
        del split_trainstate
        time.sleep(5)  # Wait a moment to ensure memory is freed
        
        # STEP 2: Python garbage collection
        import gc
        gc.collect()
        
        # STEP 3: Clear JAX's compilation cache and force memory release
        jax.clear_caches()

        render_eval_env, render_run_eval = make_evaluation(config)
        render_config = config
        render_config["NUM_EVAL_EPISODES"] = 1 
        render_eval_jit = jax.jit(
            render_run_eval,
            static_argnames=["log_eval_info"],
        )
        eval_final = render_eval_jit(eval_rng, _tree_take(final_train_state, 0, axis=0), render_log_config) #Take a single seed
        

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
        episodes_dict = {
            'worst': worst_episode,
            'median': median_episode,
            'best': best_episode,
        }
        
        # Upload HTML visualizations to WandB
        if config.get("SAVE_HTML_RENDER", True):
            upload_html_visualizations_to_wandb(render_eval_env, episodes_dict, run)

        if config.get("SAVE_MUJOCO_TRAJECTORIES", True):
            upload_mujoco_trajectories_to_wandb(render_eval_env, episodes_dict, run)

        # Conditionally upload rendered videos (larger files, immediate visual feedback)
        if config.get("RENDER_VIDEOS", False):
            try:
                upload_mujoco_videos_to_wandb(
                    render_eval_env,
                    episodes_dict,
                    run,
                    fps=config.get("VIDEO_FPS", 30),
                    quality=config.get("VIDEO_QUALITY", "high"),
                    width=config.get("VIDEO_WIDTH", 1280),
                    height=config.get("VIDEO_HEIGHT", 720),
                )
            except Exception as e:
                print(f"Warning: Video rendering failed: {e}")
                print("Continuing without videos...")

        # Generate interactive HTML visualizations
        # html.save("final_worst.html", eval_env.sys, worst_episode)
        # html.save("final_median.html", eval_env.sys, median_episode)
        # html.save("final_best.html", eval_env.sys, best_episode)
        
        print("Visualizations saved to WANDB artifacts:")
        # print("  - final_worst.html: Worst performing episode")
        # print("  - final_median.html: Median performing episode") 
        # print("  - final_best.html: Best performing episode")
        
        print("\nTraining and evaluation completed successfully!")
        end = time.time()
        print(f"MARL Run took {end - start:.2f} seconds") 
        if config["PRINT_MEMORY_STATS"]:
            print_memory_stats(f"IPPO Sweep: Final Network={network_type}, Env={config['ENV_NAME']}, Seeds={config['NUM_SEEDS']}, Num Envs={config['NUM_ENVS']},  Num Steps={config['NUM_STEPS']}")



if __name__ == "__main__":
    main()

