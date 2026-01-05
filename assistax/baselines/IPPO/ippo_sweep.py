"""
IPPO Hyperparameter Sweeping with WANDB Integration

This module orchestrates large-scale hyperparameter sweeps for IPPO experiments.
It uses JAX vmap for parallel execution across hyperparams and seeds, then
iteratively logs each configuration to WANDB as individual runs within a group.
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl') 
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
import safetensors.flax
import optax
import assistax
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from typing import Any, Dict
from base64 import urlsafe_b64encode
from datetime import datetime

# Import utilities from the training script logic
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns_sweep,
    _tree_shape, _stack_tree, _concat_tree, _tree_split, upload_eval_data_to_wandb, 
    log_all_metrics, upload_html_visualizations_to_wandb, upload_model_parameters_to_wandb,
    upload_mujoco_trajectories_to_wandb, upload_mujoco_videos_to_wandb
)

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True '

# ================================ SWEEP UTILITIES ================================

def _generate_sweep_axes(rng, config):
    """Generates hyperparameter configurations based on sweep settings."""
    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(rng, 3)
    sweep_config = config["SWEEP"]
    num_configs = sweep_config["num_configs"]
    
    # Sample from log-uniform distributions if enabled
    def sample_log_uni(key, bounds):
        return 10**jax.random.uniform(key, shape=(num_configs,), minval=bounds["min"], maxval=bounds["max"])

    lrs = sample_log_uni(lr_rng, sweep_config["lr"]) if sweep_config.get("lr") else config["LR"]
    ent_coefs = sample_log_uni(ent_coef_rng, sweep_config["ent_coef"]) if sweep_config.get("ent_coef") else config["ENT_COEF"]
    clip_epss = sample_log_uni(clip_eps_rng, sweep_config["clip_eps"]) if sweep_config.get("clip_eps") else config["CLIP_EPS"]

    return {
        "lr": {"val": lrs, "axis": 0 if sweep_config.get("lr") else None},
        "ent_coef": {"val": ent_coefs, "axis": 0 if sweep_config.get("ent_coef") else None},
        "clip_eps": {"val": clip_epss, "axis": 0 if sweep_config.get("clip_eps") else None},
    }

# ================================ MAIN SWEEP FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="ippo_sweep")
def main(config: DictConfig):
    # ===== EXPERIMENT ORGANIZATION =====
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Create a unique Group ID for WANDB based on this sweep's hash
    config_key = hash(OmegaConf.to_yaml(config)) % 2**62
    group_id = urlsafe_b64encode(config_key.to_bytes((config_key.bit_length() + 8) // 8, "big")).decode("utf-8").replace("=", "")
    
    os.makedirs(group_id, exist_ok=True)
    print(f"Sweep Group ID: {group_id}")

    # ===== DYNAMIC ALGORITHM SELECTION =====
    match (config_dict["network"]["recurrent"], config_dict["network"]["agent_param_sharing"]):
        case (False, False): from ippo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
        case (False, True):  from ippo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
        case (True, False):  from ippo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
        case (True, True):   from ippo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig

    # ===== TRAINING SETUP =====
    rng = jax.random.key(config_dict["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config_dict["NUM_SEEDS"])
    sweep = _generate_sweep_axes(sweep_rng, config_dict)
    
    # ===== TRAINING EXECUTION (PARALLEL) =====
    print(f"Executing parallel sweep across {config_dict['SWEEP']['num_configs']} configurations...")
    with jax.disable_jit(config_dict["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config_dict, save_train_state=True),
            device=jax.devices()[config_dict["DEVICE"]]
        )
        
        # Nested VMAP: [HyperparamConfigs, Seeds]
        sweep_train_vmap = jax.vmap(
            jax.vmap(train_jit, in_axes=(0, None, None, None)),
            in_axes=(None, sweep["lr"]["axis"], sweep["ent_coef"]["axis"], sweep["clip_eps"]["axis"])
        )
        
        out = sweep_train_vmap(
            train_rngs, sweep["lr"]["val"], sweep["ent_coef"]["val"], sweep["clip_eps"]["val"]
        )

    # ===== EVALUATION EXECUTION (PARALLEL) =====
    print("Evaluating all models...")
    all_train_states = out["metrics"]["train_state"]
    batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3] # (Configs, Seeds, Agents/Params)
    
    eval_env, run_eval = make_evaluation(config_dict)
    eval_log_config = EvalInfoLogConfig(done=True, reward=True, env_metrics=True)
    
    # Reshape states to a flat batch for eval
    def _flatten_states(ts):
        return jax.tree.map(lambda x: x.reshape((x.shape[0]*x.shape[1], *x.shape[2:])), ts)
    
    flat_ts = jax.jit(_flatten_states)(all_train_states)
    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
    evals = jax.vmap(eval_jit, in_axes=(None, 0, None))(eval_rng, flat_ts, eval_log_config)
    
    # Unflatten back to (Configs, Seeds, ...)
    evals = jax.tree.map(lambda x: x.reshape((*batch_dims[:2], *x.shape[1:])), evals)

    # ===== LOGGING TO WANDB (ITERATIVE) =====
    num_configs = config_dict['SWEEP']['num_configs']
    print(f"Syncing {num_configs} configurations to WANDB...")

    for i in range(num_configs):
        # Slice the Pytrees for this specific hyperparameter config
        h_out = jax.tree.map(lambda x: x[i], out)
        h_evals = jax.tree.map(lambda x: x[i], evals)
        
        # Identify current hyperparams for this specific run
        current_lr = sweep["lr"]["val"][i] if sweep["lr"]["axis"] is not None else config_dict["LR"]
        current_ent = sweep["ent_coef"]["val"][i] if sweep["ent_coef"]["axis"] is not None else config_dict["ENT_COEF"]
        
        run_name = f"config_{i}_lr{current_lr:.1e}_ent{current_ent:.1e}"
        
        run = wandb.init(
            project=config_dict["PROJECT"],
            group=group_id,
            name=run_name,
            config={**config_dict, "current_lr": current_lr, "current_ent": current_ent},
            reinit=True,
            mode=config_dict["WANDB_MODE"]
        )

        # 1. Log Training/Eval Metrics (Averaged across seeds automatically in your helper)
        log_all_metrics(config_dict, h_out, h_evals, eval_env)

        # 2. Optional Model Parameter Upload
        if config_dict.get("SAVE_PARAMS", False):
            upload_model_parameters_to_wandb(
                h_out["metrics"]["train_state"], 
                h_out["runner_state"].train_state, 
                config_dict, eval_env, run
            )

        # 3. Evaluation Data (NPZ Artifacts)
        upload_eval_data_to_wandb(h_evals, config_dict, run)

        run.finish()

    print(f"\nSweep completed. All runs available under Group: {group_id}")

if __name__ == "__main__":
    main()