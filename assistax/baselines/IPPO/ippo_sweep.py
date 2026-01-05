"""
IPPO Hyperparameter Sweeping with Parallel JAX and Grouped WANDB Logging

This script runs a parallelized hyperparameter sweep using JAX nested vmaps.
Results are then unstacked and logged as individual WANDB runs within a group,
allowing for seed-aggregation (mean/std) per hyperparameter configuration.
"""

import os
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.traverse_util import flatten_dict
import safetensors.flax
import assistax
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict
from base64 import urlsafe_b64encode
from datetime import datetime
import wandb

# Import your existing baseline utilities
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode,
    _tree_shape, _stack_tree, _concat_tree, _tree_split, upload_eval_data_to_wandb, 
    log_all_metrics, upload_html_visualizations_to_wandb, upload_model_parameters_to_wandb,
    upload_mujoco_trajectories_to_wandb, upload_mujoco_videos_to_wandb
)
from assistax.baselines.utils import _compute_episode_returns_sweep as _compute_episode_returns

# Set MuJoCo and XLA flags for performance
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

# ================================ SWEEP UTILITIES ================================

def _generate_sweep_axes(rng, config):
    """Generates hyperparameter configurations based on sweep settings."""
    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(rng, 3)
    sweep_config = config["SWEEP"]
    num_configs = sweep_config["num_configs"]
    
    # Helper to sample from log-uniform distributions
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
def main(config):
    # ===== EXPERIMENT ORGANIZATION =====
    # Generate a unique key for this sweep group
    config_key = hash(OmegaConf.to_yaml(config)) % 2**62
    group_id = urlsafe_b64encode(
        config_key.to_bytes((config_key.bit_length() + 8) // 8, "big", signed=False)
    ).decode("utf-8").replace("=", "")
    
    os.makedirs(group_id, exist_ok=True)
    print(f"Sweep Group Directory: {group_id}")
    
    config_dict = OmegaConf.to_container(config, resolve=True)

    # ===== DYNAMIC ALGORITHM SELECTION =====
    match (config_dict["network"]["recurrent"], config_dict["network"]["agent_param_sharing"]):
        case (False, False): from ippo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
        case (False, True):  from ippo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
        case (True, False):  from ippo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
        case (True, True):   from ippo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig

    # ===== SWEEP SETUP =====
    rng = jax.random.PRNGKey(config_dict["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config_dict["NUM_SEEDS"])
    sweep = _generate_sweep_axes(sweep_rng, config_dict)
    
    print(f"Starting parallel training for {config_dict['SWEEP']['num_configs']} configurations...")

    # ===== TRAINING EXECUTION (PARALLEL) =====
    with jax.disable_jit(config_dict["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config_dict, save_train_state=True),
            device=jax.devices()[config_dict["DEVICE"]]
        )
        
        # Nested VMAP: [Configs, Seeds]
        # Inner vmap: over seeds (axis 0)
        # Outer vmap: over sweep axes defined in _generate_sweep_axes
        out = jax.vmap(
            jax.vmap(train_jit, in_axes=(0, None, None, None)),
            in_axes=(None, sweep["lr"]["axis"], sweep["ent_coef"]["axis"], sweep["clip_eps"]["axis"])
        )(train_rngs, sweep["lr"]["val"], sweep["ent_coef"]["val"], sweep["clip_eps"]["val"])

        # Local save for backup
        EXCLUDED_METRICS = ["train_state"]
        jnp.save(f"{group_id}/metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )

    # ===== EVALUATION EXECUTION (PARALLEL) =====
    print("Evaluating all trained models...")
    all_train_states = out["metrics"]["train_state"]
    # Capture batch structure (Configs, Seeds, Updates)
    batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3]
    total_models = int(jnp.prod(jnp.array(batch_dims)))
    
    eval_env, run_eval = make_evaluation(config_dict)
    eval_log_config = EvalInfoLogConfig(done=True, reward=True, env_metrics=True)
    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])

    # Flatten the first 3 dims (Configs, Seeds, Updates) for sequential batch evaluation
    flat_ts = jax.tree.map(lambda x: x.reshape((total_models, *x.shape[3:])), all_train_states)
    
    n_sequential_evals = int(jnp.ceil(
        config_dict["NUM_EVAL_EPISODES"] * total_models / config_dict["GPU_ENV_CAPACITY"]
    ))
    split_ts = _tree_split(flat_ts, n_sequential_evals)
    
    eval_batches = []
    for ts_batch in tqdm(split_ts, desc="Evaluation Batches"):
        res = jax.vmap(eval_jit, in_axes=(None, 0, None))(eval_rng, ts_batch, eval_log_config)
        eval_batches.append(res)
    
    # Combine and reshape back to (Configs, Seeds, Updates, ...)
    evals = _concat_tree(eval_batches)
    evals = jax.tree.map(lambda x: x.reshape((*batch_dims, *x.shape[1:])), evals)

    # ===== LOGGING TO WANDB (ITERATIVE) =====
    # We iterate over the first dimension (Configs) to create unique runs
    num_configs = batch_dims[0]
    print(f"\nProcessing {num_configs} configurations for WANDB logging...")
    
    env = assistax.make(config_dict["ENV_NAME"], **config_dict["ENV_KWARGS"])

    for h_idx in range(num_configs):
        # Slice the data to get (Seeds, Updates, ...) for this config
        h_out = jax.tree.map(lambda x: x[h_idx], out)
        h_evals = jax.tree.map(lambda x: x[h_idx], evals)
        
        # Determine specific hparams for metadata
        cur_lr = float(sweep["lr"]["val"][h_idx]) if sweep["lr"]["axis"] is not None else config_dict["LR"]
        cur_ent = float(sweep["ent_coef"]["val"][h_idx]) if sweep["ent_coef"]["axis"] is not None else config_dict["ENT_COEF"]
        
        # Standard IPPO tag creation for individual run naming
        run_name = f"hp_{h_idx}_lr{cur_lr:.1e}_ent{cur_ent:.1e}"
        
        run = wandb.init(
            entity=config_dict["ENTITY"],
            project=config_dict["PROJECT"],
            group=group_id, # Shared group ID for the sweep
            name=run_name,
            config={**config_dict, "current_lr": cur_lr, "current_ent": cur_ent},
            reinit=True,
            mode=config_dict["WANDB_MODE"],
            tags=config_dict.get("EXP_TAGS", []) + ["sweep"]
        )

        # Log metrics using your existing utility
        # This will now compute mean/std across the Seeds dimension correctly
        log_all_metrics(config_dict, h_out, h_evals, env)
        
        # Optional: Save evaluation data NPZ artifacts
        if config_dict.get("UPLOAD_EVAL_DATA", True):
            upload_eval_data_to_wandb(h_evals, config_dict, run)
            
        # Optional: Save parameters (usually excessive for sweeps, but available)
        if config_dict.get("SAVE_PARAMS", False):
            upload_model_parameters_to_wandb(
                h_out["metrics"]["train_state"], 
                h_out["runner_state"].train_state, 
                config_dict, env, run
            )

        run.finish()

    print(f"\nHyperparameter sweep completed! View results under Group ID: {group_id}")

if __name__ == "__main__":
    main()