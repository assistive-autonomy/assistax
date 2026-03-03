import jax
import jax.numpy as jnp
import numpy as np
import wandb
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax 
import assistax
from assistax.wrappers.baselines import get_space_dim, LogEnvState, LogWrapper, LogCrossplayWrapper
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper, LoadEvalAgentWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Optional
import os 
import functools
import mujoco
import mediapy as media 
import pandas as pd
from datetime import datetime

# Tree Utilities 

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


def _compute_episode_returns(eval_info, time_axis=-2):
    """
    Compute undiscounted episode returns from evaluation information.
    
    Handles episode boundaries correctly by resetting cumulative rewards
    when episodes end and start new ones.
    
    Args:
        eval_info: Evaluation information containing rewards and done flags
        time_axis: Axis representing time dimension (default: -2)
        
    Returns:
        Undiscounted returns for each episode
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
    return undiscounted_returns

def _compute_episode_metrics(metric_arr, done_arr, time_axis=-2):
    """
    Compute undiscounted episode metrics from metric array and done flags.
    
    Handles episode boundaries correctly by resetting cumulative metrics
    when episodes end and start new ones.
    
    Args:
        metric_arr: Array of metrics to sum over episodes
        done_arr: Boolean array indicating episode termination
        time_axis: Axis representing time dimension (default: -2)
    Returns:
        Undiscounted metrics for each episode
    """
    # Create mask for episode boundaries
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    
    # Sum metrics within episodes only
    undiscounted_metrics = (metric_arr * (1 - episode_done)).sum(axis=time_axis)
    
    return undiscounted_metrics

def _compute_episode_returns_sweep(eval_info, common_reward=False, time_axis=-2):
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

# WANDB UPLOAD UTILITIES

def upload_eval_data_to_wandb(eval_info, config, run, suffix=""):
    """
    Upload evaluation data to wandb as artifacts (compact: NaNs removed from storage).
    
    Args:
        eval_info: EvalInfo NamedTuple from evaluation
        config: Configuration dictionary
        run: wandb run object
        suffix: Optional suffix for artifact name
    """
    if not config.get("UPLOAD_EVAL_DATA", True):
        return
    
    print("Uploading evaluation data to wandb (compact)…")
    import tempfile
    import os
    import numpy as np
    
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Compute and save episode returns
        if eval_info.reward is not None and eval_info.done is not None:
            from assistax.baselines.utils import _compute_episode_returns
            
            episode_returns = _compute_episode_returns(eval_info, time_axis=2)
            # episode_returns is a dict: {'__all__': array, 'robot1': array, ...}
            
            for agent_key, returns in episode_returns.items():
                # Shape: (num_seeds, num_updates, num_eval_episodes)
                file_path = os.path.join(temp_dir, f"episode_returns_{agent_key}.npz")
                _save_compact_npz(file_path, returns)
                print(f"Saved episode_returns_{agent_key} with shape {np.asarray(returns).shape}")
            
            # Also save mean episode returns (averaged over eval episodes)
            mean_episode_returns = {
                agent_key: jnp.mean(returns, axis=-1)  # Average over eval episodes
                for agent_key, returns in episode_returns.items()
            }
            for agent_key, mean_returns in mean_episode_returns.items():
                # Shape: (num_seeds, num_updates)
                file_path = os.path.join(temp_dir, f"mean_episode_returns_{agent_key}.npz")
                _save_compact_npz(file_path, mean_returns)
                print(f"Saved mean_episode_returns_{agent_key} with shape {np.asarray(mean_returns).shape}")
        
        # Iterate over EvalInfo fields
        for field_name in eval_info._fields:
            field_data = getattr(eval_info, field_name)
            
            # Skip None fields
            if field_data is None:
                continue
            
            # Handle dict fields (like reward, done, obs, etc.)
            if isinstance(field_data, dict):
                for subkey, subdata in field_data.items():
                    if isinstance(subdata, (np.ndarray, jnp.ndarray)):
                        file_path = os.path.join(temp_dir, f"{field_name}_{subkey}.npz")
                        _save_compact_npz(file_path, subdata)
                        print(f"Saved {field_name}_{subkey} with shape {np.asarray(subdata).shape}")
                    else:
                        # For non-arrays, save as .npy
                        file_path = os.path.join(temp_dir, f"{field_name}_{subkey}.npy")
                        np.save(file_path, np.array(subdata, dtype=object))
                        print(f"Saved non-array {field_name}_{subkey} as object npy")
            
            # Handle array fields
            elif isinstance(field_data, (np.ndarray, jnp.ndarray)):
                file_path = os.path.join(temp_dir, f"{field_name}.npz")
                _save_compact_npz(file_path, field_data)
                print(f"Saved {field_name} with shape {np.asarray(field_data).shape}")
            
            # Handle LogEnvState (for env_state field)
            elif hasattr(field_data, '_fields'):  # It's a NamedTuple
                # Save nested NamedTuple fields
                for nested_field in field_data._fields:
                    nested_data = getattr(field_data, nested_field)
                    if isinstance(nested_data, (np.ndarray, jnp.ndarray)):
                        file_path = os.path.join(temp_dir, f"{field_name}_{nested_field}.npz")
                        _save_compact_npz(file_path, nested_data)
                        print(f"Saved {field_name}_{nested_field} with shape {np.asarray(nested_data).shape}")
            
            else:
                # For other types, try to save as .npy
                try:
                    file_path = os.path.join(temp_dir, f"{field_name}.npy")
                    np.save(file_path, np.array(field_data, dtype=object))
                    print(f"Saved {field_name} as object npy")
                except:
                    print(f"Skipping {field_name} (unsupported type: {type(field_data)})")
        
        # Create and upload artifact
        artifact = wandb.Artifact(f"evaluation_data_{run.name}", type="dataset")
        artifact.add_dir(temp_dir)
        run.log_artifact(artifact)
    
    print("Evaluation data uploaded to wandb successfully!")


def _save_compact_npz(path: str, x):
    """Save array x compactly: store data (NaNs→0), bit-packed mask, shape, dtype."""
    import numpy as np
    x = np.asarray(x)
    mask = np.isnan(x).ravel()
    np.savez_compressed(
        path,
        data=np.nan_to_num(x, nan=0.0),                    # same shape as x, NaNs -> 0
        mask=np.packbits(mask),                            # 8x smaller than bools
        shape=np.array(x.shape, dtype=np.int64),
        dtype=str(x.dtype),
    )


def load_compact_npz(path: str) -> np.ndarray:
    """Load array saved with _save_compact_npz."""
    import numpy as np
    f = np.load(path)
    data  = f["data"]
    shape = tuple(f["shape"])
    dtype = np.dtype(str(f["dtype"]))
    mask  = np.unpackbits(f["mask"]).astype(bool)[:np.prod(shape)].reshape(shape)
    data  = data.astype(dtype, copy=False)
    return np.where(mask, np.nan, data)

#def log_multi_seed_metrics(
#    metrics_dict,
#    config,
#    metric_key,
#    x_axis_key="env_step",
#    prefix="train",
#    agent_names=None,
#):
#    """Log multi-seed metrics to wandb with per-agent curves."""
#    if config.get("NUM_SEEDS") is None or config["NUM_SEEDS"] <= 0:
#        return
#    
#    metric_values = metrics_dict[metric_key]
#    x_values = metrics_dict[x_axis_key]
#    
#    print(f"\n{prefix.upper()} logging - Metric: {metric_key}")
#    print(f"  Shape: {metric_values.shape}")
#    
#    has_agent_dim = len(metric_values.shape) == 3
#    x_axis = x_values[0] if len(x_values.shape) > 1 else x_values
#    
#    if has_agent_dim:
#        num_seeds, num_updates, num_agents = metric_values.shape
#        print(f"  Detected {num_agents} agents")
#        
#        # Pre-compute statistics for ALL agents
#        agent_stats = {}
#        for agent_idx in range(num_agents):
#            agent_metric = metric_values[:, :, agent_idx]
#            
#            if agent_names is not None and agent_idx < len(agent_names):
#                agent_name = agent_names[agent_idx]
#            else:
#                agent_name = f"agent_{agent_idx}"
#            
#            # Calculate statistics across seeds
#            mean_values = jnp.mean(agent_metric, axis=0)
#            std_values = jnp.std(agent_metric, axis=0)
#            max_values = jnp.max(agent_metric, axis=0)
#            min_values = jnp.min(agent_metric, axis=0)
#            
#            agent_stats[agent_name] = {
#                'mean': mean_values,
#                'std': std_values,
#                'max': max_values,
#                'min': min_values,
#            }
#            
#            # Log final statistics
#            final_window = min(10, len(mean_values))
#            final_mean = float(jnp.mean(mean_values[-final_window:]))
#            final_std = float(jnp.mean(std_values[-final_window:]))
#            
#            wandb.log({
#                f"{prefix}/final/{agent_name}_{metric_key}_mean": final_mean,
#                f"{prefix}/final/{agent_name}_{metric_key}_std": final_std,
#            })
#            
#            print(f"  Agent {agent_name}: final_mean={final_mean:.2f}")
#        
#        # NOW log time series for ALL agents at EACH timestep together
#        num_steps = len(x_axis)
#        for i in range(num_steps):
#            step_value = int(x_axis[i])
#            
#            # Build log dict with ALL agents' data for this timestep
#            log_dict = {}
#            for agent_name, stats in agent_stats.items():
#                log_dict[f"{prefix}/{agent_name}_{metric_key}_mean"] = float(stats['mean'][i])
#                log_dict[f"{prefix}/{agent_name}_{metric_key}_std"] = float(stats['std'][i])
#                log_dict[f"{prefix}/{agent_name}_{metric_key}_max"] = float(stats['max'][i])
#                log_dict[f"{prefix}/{agent_name}_{metric_key}_min"] = float(stats['min'][i])
#            
#            # Single wandb.log() call with all agents' data
#            wandb.log(log_dict, step=step_value)
#        
#        print(f"  Logged {num_steps} timesteps for {num_agents} agents\n")
#    
#    else:
#        # No agent dimension - log as single metric
#        num_seeds, num_updates = metric_values.shape
#        
#        mean_values = jnp.mean(metric_values, axis=0)
#        std_values = jnp.std(metric_values, axis=0)
#        
#        # Log final performance
#        final_window = min(10, len(mean_values))
#        final_mean = float(jnp.mean(mean_values[-final_window:]))
#        final_std = float(jnp.mean(std_values[-final_window:]))
#        
#        wandb.log({
#            f"{prefix}/final/{metric_key}_mean": final_mean,
#            f"{prefix}/final/{metric_key}_std": final_std,
#        })
#        
#        # Log time series
#        for i in range(len(mean_values)):
#            step_value = int(x_axis[i])
#            wandb.log({
#                f"{prefix}/{metric_key}_mean": float(mean_values[i]),
#                f"{prefix}/{metric_key}_std": float(std_values[i]),
#            }, step=step_value)
#        
#        print(f"  Logged {metric_key}: final_mean={final_mean:.2f}\n")
#    
#    print("Training metrics logged to wandb successfully!")

def log_all_metrics(config, out, evals, env):
    """
    Log both training and evaluation metrics together to avoid step ordering issues.
    
    Args:
        config: Configuration dictionary
        out: Training output with metrics
        evals: Evaluation info
        env: Environment
    """
    from assistax.baselines.utils import _compute_episode_returns
    
    print("\n" + "="*70)
    print("LOGGING ALL METRICS (TRAINING + EVALUATION)")
    print("="*70)
    
    # Get environment info
    if hasattr(env, 'agents'):
        agent_names = env.agents
    else:
        agent_names = [f"agent_{i}" for i in range(env.num_agents)]
    
    # Extract data
    train_metrics = out["metrics"]
    if "env_step" in train_metrics:
        # PPO: env_step is tracked directly in metrics
        env_steps = train_metrics["env_step"]  # Shape: (num_seeds, num_updates)
        x_axis = env_steps[0]  # Use first seed
        num_checkpoints = len(x_axis)
    else:
        # SAC: compute env steps per checkpoint from config
        num_checkpoints = config["NUM_CHECKPOINTS"]
        steps_per_checkpoint = config["SCAN_STEPS"] * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
        explore_steps = config.get("EXPLORE_STEPS", 0)
        x_axis = jnp.array([
            explore_steps + (i + 1) * steps_per_checkpoint
            for i in range(num_checkpoints)
        ])
    
    # ===== PRE-COMPUTE ALL TRAINING STATISTICS =====
    print("\nPre-computing training statistics...")
    train_stats = {}
    
    # Returns
    if "returned_episode_returns" in train_metrics:
        returns = train_metrics["returned_episode_returns"]
        if len(returns.shape) == 3:  # Has agent dimension
            for agent_idx, agent_name in enumerate(agent_names):
                agent_returns = returns[:, :, agent_idx]
                train_stats[f"train/returns/{agent_name}_return"] = {
                    'mean': jnp.mean(agent_returns, axis=0),
                    'std': jnp.std(agent_returns, axis=0),
                }
                
    # Losses and other metrics (dynamically detect available keys)
    all_known_metrics = [
        # PPO
        "total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl",
        "clip_frac_min", "clip_frac_max",
        # SAC
        "q1_loss", "q2_loss", "alpha_loss",
        "alpha", "log_probs", "next_log_probs",
    ]
    loss_metrics = [m for m in all_known_metrics if m in train_metrics]
    
    for metric_key in loss_metrics:
        if metric_key not in train_metrics:
            continue
            
        metric_values = train_metrics[metric_key]
        
        loss_keys = {"total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl",
                     "q1_loss", "q2_loss", "alpha_loss"}
        if len(metric_values.shape) == 3:  # Has agent dimension
            for agent_idx, agent_name in enumerate(agent_names):
                agent_metric = metric_values[:, :, agent_idx]
                prefix = "train/loss" if metric_key in loss_keys else "train/diagnostics"
                train_stats[f"{prefix}/{agent_name}_{metric_key}"] = {
                    'mean': jnp.mean(agent_metric, axis=0),
                    'std': jnp.std(agent_metric, axis=0),
                }
        else:  # No agent dimension
            prefix = "train/loss" if metric_key in loss_keys else "train/diagnostics"
            train_stats[f"{prefix}/{metric_key}"] = {
                'mean': jnp.mean(metric_values, axis=0),
                'std': jnp.std(metric_values, axis=0),
            }
    
    print(f"Pre-computed {len(train_stats)} training metrics")
    
    # ===== PRE-COMPUTE ALL EVALUATION STATISTICS =====
    print("\nPre-computing evaluation statistics...")
    eval_stats = {}
    
    if evals.reward is not None and evals.done is not None:
        # Compute episode returns
        episode_returns = _compute_episode_returns(evals, time_axis=2)
        agent_keys = [k for k in episode_returns.keys() if k != '__all__']
        
        # Individual agents
        for agent_key in agent_keys:
            agent_returns = episode_returns[agent_key]
            mean_returns_per_checkpoint = jnp.mean(agent_returns, axis=2)  # Average over eval episodes
            
            eval_stats[f"eval/{agent_key}_return"] = {
                'mean': jnp.mean(mean_returns_per_checkpoint, axis=0),
                'std': jnp.std(mean_returns_per_checkpoint, axis=0),
                'max': jnp.max(mean_returns_per_checkpoint, axis=0),
                'min': jnp.min(mean_returns_per_checkpoint, axis=0),
            }
        
        # Team aggregate
        if '__all__' in episode_returns:
            all_returns = episode_returns['__all__']
            mean_returns = jnp.mean(all_returns, axis=2)
            
            eval_stats[f"eval/team_return"] = {
                'mean': jnp.mean(mean_returns, axis=0),
                'std': jnp.std(mean_returns, axis=0),
                'max': jnp.max(mean_returns, axis=0),
                'min': jnp.min(mean_returns, axis=0),
            }
    
    # Environment metrics
    if evals.env_metrics is not None:
        for metric_name, metric_values in evals.env_metrics.items():
            if not isinstance(metric_values, (jnp.ndarray, np.ndarray)):
                continue
            
            # Check if this is a reward component that should get episode sums
            is_reward_component = 'reward' in metric_name.lower() or 'weighted' in metric_name.lower()
            
            if is_reward_component and evals.done is not None:
                # Create a temporary evals-like object with this metric as the reward
                
                metric_episode_returns = _compute_episode_metrics(metric_values, evals.done["__all__"], time_axis=2)
                
                mean_metric_per_checkpoint = jnp.mean(metric_episode_returns, axis=2)
                
                eval_stats[f"eval/env_metrics/{metric_name}"] = {
                    'mean': jnp.mean(mean_metric_per_checkpoint, axis=0),
                    'std': jnp.std(mean_metric_per_checkpoint, axis=0),
                }
                
                # for key, returns in metric_episode_returns.items():
                #     # returns shape: (num_seeds, num_checkpoints, num_episodes)
                #     mean_returns_per_checkpoint = jnp.mean(returns, axis=2)  # Average over episodes
                #     
                #     suffix = "" if key == '__all__' else f"/{key}"
                #     eval_stats[f"eval/reward_components/{metric_name}{suffix}"] = {
                #         'mean': jnp.mean(mean_returns_per_checkpoint, axis=0),
                #         'std': jnp.std(mean_returns_per_checkpoint, axis=0),
                #     }
            
            else:
                # Regular averaging for non-reward metrics
                if len(metric_values.shape) == 4:
                    mean_metric = jnp.mean(metric_values, axis=(2, 3))
                elif len(metric_values.shape) == 3:
                    mean_metric = jnp.mean(metric_values, axis=2)
                else:
                    continue
                
                eval_stats[f"eval/env_metrics/{metric_name}"] = {
                    'mean': jnp.mean(mean_metric, axis=0),
                    'std': jnp.std(mean_metric, axis=0),
                }

    print(f"Pre-computed {len(eval_stats)} evaluation metrics")
    
    # ===== LOG ALL METRICS TOGETHER AT EACH CHECKPOINT =====
    print(f"\nLogging {num_checkpoints} checkpoints...")
    
    for checkpoint_idx in range(num_checkpoints):
        step_value = int(x_axis[checkpoint_idx])
        
        log_dict = {}
        
        # Add all training metrics
        for metric_name, stats in train_stats.items():
            log_dict[f"{metric_name}_mean"] = float(stats['mean'][checkpoint_idx])
            log_dict[f"{metric_name}_std"] = float(stats['std'][checkpoint_idx])
        
        # Add all evaluation metrics
        for metric_name, stats in eval_stats.items():
            log_dict[f"{metric_name}_mean"] = float(stats['mean'][checkpoint_idx])
            log_dict[f"{metric_name}_std"] = float(stats['std'][checkpoint_idx])
            if 'max' in stats:
                log_dict[f"{metric_name}_max"] = float(stats['max'][checkpoint_idx])
            if 'min' in stats:
                log_dict[f"{metric_name}_min"] = float(stats['min'][checkpoint_idx])
        
        # Single wandb.log() call with ALL data for this checkpoint
        wandb.log(log_dict, step=step_value)
    
    print(f"Logged {num_checkpoints} checkpoints successfully!")
    
    # ===== LOG FINAL STATISTICS =====
    print("\nLogging final statistics...")
    final_stats = {}
    final_window = min(10, num_checkpoints)
    
    # Training finals
    for metric_name, stats in train_stats.items():
        final_mean = float(jnp.mean(stats['mean'][-final_window:]))
        final_std = float(jnp.mean(stats['std'][-final_window:]))
        final_stats[f"{metric_name}_final_mean"] = final_mean
        final_stats[f"{metric_name}_final_std"] = final_std
    
    # Evaluation finals
    for metric_name, stats in eval_stats.items():
        final_mean = float(jnp.mean(stats['mean'][-final_window:]))
        final_std = float(jnp.mean(stats['std'][-final_window:]))
        final_stats[f"{metric_name}_final_mean"] = final_mean
        final_stats[f"{metric_name}_final_std"] = final_std
        if 'max' in stats:
            final_stats[f"{metric_name}_final_max"] = float(jnp.max(stats['max'][-final_window:]))
        if 'min' in stats:
            final_stats[f"{metric_name}_final_min"] = float(jnp.min(stats['min'][-final_window:]))
    
    wandb.log(final_stats)
    
    print("="*70)
    print("ALL METRICS LOGGING COMPLETE")
    print("="*70 + "\n")

def log_all_metrics_zsc(config: dict, out: dict, evals_train, evals_test, env) -> None:
    """
    Log training and dual evaluation metrics (train-partner + test-partner) to wandb.

    Follows the same 3-phase pattern as ``log_all_metrics`` but logs two evaluation
    sets under ``eval_train/`` and ``eval_test/`` prefixes for ZSC generalization analysis.

    Args:
        config: Configuration dictionary.
        out: Training output containing ``out["metrics"]``.
        evals_train: Evaluation info from train-partner set.
        evals_test: Evaluation info from test-partner set.
        env: Environment (used for agent names).
    """
    from assistax.baselines.utils import _compute_episode_returns, _compute_episode_metrics

    print("\n" + "=" * 70)
    print("LOGGING ALL METRICS (TRAINING + EVAL_TRAIN + EVAL_TEST)")
    print("=" * 70)

    # Agent names
    if hasattr(env, "agents"):
        agent_names = env.agents
    else:
        agent_names = [f"agent_{i}" for i in range(env.num_agents)]

    # ===== EXTRACT TRAINING DATA =====
    train_metrics = out["metrics"]
    env_steps = train_metrics["env_step"]  # (num_seeds, num_updates)
    x_axis = env_steps[0]
    num_checkpoints = len(x_axis)

    # ===== PRE-COMPUTE TRAINING STATISTICS =====
    print("\nPre-computing training statistics...")
    train_stats = {}

    if "returned_episode_returns" in train_metrics:
        returns = train_metrics["returned_episode_returns"]
        if len(returns.shape) == 3:
            for agent_idx, agent_name in enumerate(agent_names):
                agent_returns = returns[:, :, agent_idx]
                train_stats[f"train/returns/{agent_name}_return"] = {
                    "mean": jnp.mean(agent_returns, axis=0),
                    "std": jnp.std(agent_returns, axis=0),
                }

    loss_metrics = [
        "total_loss", "actor_loss", "critic_loss", "entropy",
        "approx_kl", "clip_frac_min", "clip_frac_max",
    ]
    for metric_key in loss_metrics:
        if metric_key not in train_metrics:
            continue
        metric_values = train_metrics[metric_key]
        if len(metric_values.shape) == 3:
            for agent_idx, agent_name in enumerate(agent_names):
                agent_metric = metric_values[:, :, agent_idx]
                prefix = (
                    "train/loss"
                    if metric_key in ["total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl"]
                    else "train/diagnostics"
                )
                train_stats[f"{prefix}/{agent_name}_{metric_key}"] = {
                    "mean": jnp.mean(agent_metric, axis=0),
                    "std": jnp.std(agent_metric, axis=0),
                }
        else:
            prefix = (
                "train/loss"
                if metric_key in ["total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl"]
                else "train/diagnostics"
            )
            train_stats[f"{prefix}/{metric_key}"] = {
                "mean": jnp.mean(metric_values, axis=0),
                "std": jnp.std(metric_values, axis=0),
            }

    print(f"Pre-computed {len(train_stats)} training metrics")

    # ===== HELPER: pre-compute eval statistics for one eval set =====
    def _precompute_eval_stats(evals, prefix: str) -> dict:
        stats = {}
        if evals.reward is None or evals.done is None:
            return stats

        episode_returns = _compute_episode_returns(evals, time_axis=2)
        agent_keys = [k for k in episode_returns.keys() if k != "__all__"]

        for agent_key in agent_keys:
            agent_returns = episode_returns[agent_key]
            mean_per_cp = jnp.mean(agent_returns, axis=2)
            stats[f"{prefix}/{agent_key}_return"] = {
                "mean": jnp.mean(mean_per_cp, axis=0),
                "std": jnp.std(mean_per_cp, axis=0),
                "max": jnp.max(mean_per_cp, axis=0),
                "min": jnp.min(mean_per_cp, axis=0),
            }

        if "__all__" in episode_returns:
            all_returns = episode_returns["__all__"]
            mean_returns = jnp.mean(all_returns, axis=2)
            stats[f"{prefix}/team_return"] = {
                "mean": jnp.mean(mean_returns, axis=0),
                "std": jnp.std(mean_returns, axis=0),
                "max": jnp.max(mean_returns, axis=0),
                "min": jnp.min(mean_returns, axis=0),
            }

        # Environment metrics
        if evals.env_metrics is not None:
            for metric_name, metric_values in evals.env_metrics.items():
                if not isinstance(metric_values, (jnp.ndarray, np.ndarray)):
                    continue
                is_reward_component = "reward" in metric_name.lower() or "weighted" in metric_name.lower()
                if is_reward_component and evals.done is not None:
                    metric_ep = _compute_episode_metrics(metric_values, evals.done["__all__"], time_axis=2)
                    mean_metric_per_cp = jnp.mean(metric_ep, axis=2)
                    stats[f"{prefix}/env_metrics/{metric_name}"] = {
                        "mean": jnp.mean(mean_metric_per_cp, axis=0),
                        "std": jnp.std(mean_metric_per_cp, axis=0),
                    }
                else:
                    if len(metric_values.shape) == 4:
                        mean_metric = jnp.mean(metric_values, axis=(2, 3))
                    elif len(metric_values.shape) == 3:
                        mean_metric = jnp.mean(metric_values, axis=2)
                    else:
                        continue
                    stats[f"{prefix}/env_metrics/{metric_name}"] = {
                        "mean": jnp.mean(mean_metric, axis=0),
                        "std": jnp.std(mean_metric, axis=0),
                    }
        return stats

    # ===== PRE-COMPUTE EVAL STATISTICS FOR BOTH SETS =====
    print("\nPre-computing eval_train statistics...")
    eval_train_stats = _precompute_eval_stats(evals_train, "eval_train")
    print(f"Pre-computed {len(eval_train_stats)} eval_train metrics")

    print("Pre-computing eval_test statistics...")
    eval_test_stats = _precompute_eval_stats(evals_test, "eval_test")
    print(f"Pre-computed {len(eval_test_stats)} eval_test metrics")

    # ===== LOG ALL METRICS TOGETHER AT EACH CHECKPOINT =====
    print(f"\nLogging {num_checkpoints} checkpoints...")

    all_eval_stats = {**eval_train_stats, **eval_test_stats}

    for checkpoint_idx in range(num_checkpoints):
        step_value = int(x_axis[checkpoint_idx])
        log_dict = {}

        for metric_name, stats in train_stats.items():
            log_dict[f"{metric_name}_mean"] = float(stats["mean"][checkpoint_idx])
            log_dict[f"{metric_name}_std"] = float(stats["std"][checkpoint_idx])

        for metric_name, stats in all_eval_stats.items():
            log_dict[f"{metric_name}_mean"] = float(stats["mean"][checkpoint_idx])
            log_dict[f"{metric_name}_std"] = float(stats["std"][checkpoint_idx])
            if "max" in stats:
                log_dict[f"{metric_name}_max"] = float(stats["max"][checkpoint_idx])
            if "min" in stats:
                log_dict[f"{metric_name}_min"] = float(stats["min"][checkpoint_idx])

        wandb.log(log_dict, step=step_value)

    print(f"Logged {num_checkpoints} checkpoints successfully!")

    # ===== LOG FINAL STATISTICS =====
    print("\nLogging final statistics...")
    final_stats = {}
    final_window = min(10, num_checkpoints)

    for metric_name, stats in train_stats.items():
        final_stats[f"{metric_name}_final_mean"] = float(jnp.mean(stats["mean"][-final_window:]))
        final_stats[f"{metric_name}_final_std"] = float(jnp.mean(stats["std"][-final_window:]))

    for metric_name, stats in all_eval_stats.items():
        final_stats[f"{metric_name}_final_mean"] = float(jnp.mean(stats["mean"][-final_window:]))
        final_stats[f"{metric_name}_final_std"] = float(jnp.mean(stats["std"][-final_window:]))
        if "max" in stats:
            final_stats[f"{metric_name}_final_max"] = float(jnp.max(stats["max"][-final_window:]))
        if "min" in stats:
            final_stats[f"{metric_name}_final_min"] = float(jnp.min(stats["min"][-final_window:]))

    wandb.log(final_stats)

    print("=" * 70)
    print("ALL METRICS LOGGING COMPLETE")
    print("=" * 70 + "\n")

#def upload_html_visualizations_to_wandb(eval_env, episodes_dict, run):
#    """
#    Upload HTML visualizations to wandb as artifacts.
#    
#    Args:
#        eval_env: Evaluation environment (for sys attribute)
#        episodes_dict: Dict mapping names to episode data
#                      e.g., {'worst': worst_episode, 'median': median_episode, 'best': best_episode}
#        run: wandb run object
#    """
#    from assistax.render import html
#    import tempfile
#    import os
#    
#    print("Creating and uploading HTML visualizations...")
#    
#    with tempfile.TemporaryDirectory() as temp_dir:
#        # Generate HTML files in temporary directory
#        html_files = {}
#        for name, episode_data in episodes_dict.items():
#            file_path = os.path.join(temp_dir, f"final_{name}.html")
#            html.save(file_path, eval_env.sys, episode_data)
#            html_files[name] = file_path
#            print(f"  Generated final_{name}.html")
#        
#        # Create and upload artifact
#        artifact = wandb.Artifact(f"visualizations_{run.name}", type="visualization")
#        artifact.add_dir(temp_dir)
#        run.log_artifact(artifact)
#        for name, file_path in html_files.items():
#            with open(file_path, 'r') as f:
#                html_content = f.read()
#            # This makes it viewable in the wandb dashboard
#            wandb.log({f"visualization/{name}_episode": wandb.Html(html_content)})
#            print(f"  Logged {name} episode for interactive viewing")   
#    print("HTML visualizations uploaded to wandb successfully!")

def upload_html_visualizations_to_wandb(eval_env, episodes_dict, run):
    """
    Upload HTML visualizations to wandb as artifacts.
    
    Args:
        eval_env: Evaluation environment (Gymnax-style wrapper with env.env.sys)
        episodes_dict: Dict mapping names to episode data (qp trajectories)
                      e.g., {'worst': worst_episode, 'median': median_episode, 'best': best_episode}
        run: wandb run object
    """
    from brax.io import html  # or: from mujoco import mjx; from mujoco.mjx import io as html
    import tempfile
    import os
    
    print("Creating and uploading HTML visualizations...")
    
    # Get the MuJoCo system with proper timestep
    sys = eval_env.env.sys.tree_replace({'opt.timestep': eval_env.env.dt})
    
    with tempfile.TemporaryDirectory() as temp_dir:
        html_files = {}
        
        for name, episode_data in episodes_dict.items():
            # Render HTML string
            html_content = html.render(sys, episode_data)
            
            # Save to temporary file
            file_path = os.path.join(temp_dir, f"final_{name}.html")
            with open(file_path, 'w') as f:
                f.write(html_content)
            
            html_files[name] = (file_path, html_content)
            print(f"  Generated final_{name}.html")
        
        # Create and upload artifact for file downloads
        artifact = wandb.Artifact(f"visualizations_{run.name}", type="visualization")
        artifact.add_dir(temp_dir)
        run.log_artifact(artifact)
        print("  Artifact uploaded")
        
        # Log HTML for interactive viewing in wandb dashboard
        for name, (file_path, html_content) in html_files.items():
            wandb.log({f"visualization/{name}_episode": wandb.Html(html_content)})
            print(f"  Logged {name} episode for interactive viewing")
    
    print("HTML visualizations uploaded to wandb successfully!")

def upload_model_parameters_to_wandb(all_train_states, final_train_state, config, env, run):
    """
    Upload model parameters to wandb as artifacts using temporary files.
    
    Args:
        all_train_states: Training states from all checkpoints
        final_train_state: Final training state
        config: Configuration dictionary
        env: Environment (for agent names)
        run: wandb run object
    """
    import tempfile
    import os
    import safetensors.flax
    from flax.traverse_util import flatten_dict
    
    print("Saving and uploading model parameters...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save all training states (checkpoints)
        all_params_path = os.path.join(temp_dir, "all_params.safetensors")
        safetensors.flax.save_file(
            flatten_dict(all_train_states.params, sep='/'),
            all_params_path
        )
        print(f"  Saved all_params.safetensors")
        
        # Save final parameters
        if config["network"]["agent_param_sharing"]:
            # For parameter sharing: single set of shared parameters
            final_params_path = os.path.join(temp_dir, "final_params.safetensors")
            safetensors.flax.save_file(
                flatten_dict(final_train_state.params, sep='/'),
                final_params_path
            )
            print(f"  Saved final_params.safetensors (parameter sharing)")
        else:
            # For independent parameters: split by agent
            split_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state.params)
            )
            for agent, params in zip(env.agents, split_params):
                agent_params_path = os.path.join(temp_dir, f"{agent}.safetensors")
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    agent_params_path
                )
                print(f"  Saved {agent}.safetensors")
        
        # Create and upload artifact
        artifact = wandb.Artifact(f"model_parameters_{run.name}", type="model")
        artifact.add_dir(temp_dir)
        run.log_artifact(artifact)
    
    print("Model parameters uploaded to wandb successfully!")


def upload_mujoco_trajectories_to_wandb(eval_env, episodes_dict, run):
    """
    Save MuJoCo XML models and trajectories, then upload to wandb as artifacts.
    
    Args:
        eval_env: Evaluation environment (Gymnax-style wrapper with env.env.sys)
        episodes_dict: Dict mapping names to episode data (list of brax.mjx.base.State objects)
        run: wandb run object
    """
    import tempfile
    import os
    import numpy as np
    import shutil
    
    print("Creating and uploading MuJoCo trajectories...")
    
    # Get the XML file path from the environment
    if hasattr(eval_env.env, 'path'):
        xml_source_path = str(eval_env.env.path)
        print(f"  Using XML from: {xml_source_path}")
    else:
        print("  Warning: Environment does not have 'path' attribute. Skipping.")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Copy the XML file to temp directory
        model_xml_path = os.path.join(temp_dir, "model.xml")
        shutil.copy(xml_source_path, model_xml_path)
        print(f"  Copied model.xml")
        
        # Save trajectory data for each episode
        for name, episode_data in episodes_dict.items():
            # episode_data is a list of brax.mjx.base.State objects
            # Each State has .qpos and .qvel attributes
            
            if not isinstance(episode_data, list) or len(episode_data) == 0:
                print(f"  Warning: {name} episode is invalid")
                continue
            
            # Extract qpos and qvel from each state
            qpos_trajectory = np.array([np.array(state.qpos) for state in episode_data])
            qvel_trajectory = np.array([np.array(state.qvel) for state in episode_data])
            
            print(f"  {name}: {len(episode_data)} steps, qpos shape {qpos_trajectory.shape}, qvel shape {qvel_trajectory.shape}")
            
            # Save trajectory
            traj_path = os.path.join(temp_dir, f"{name}_trajectory.npz")
            np.savez(
                traj_path,
                qpos=qpos_trajectory,
                qvel=qvel_trajectory,
                timestep=eval_env.env.dt,
            )
            print(f"  Saved {name}_trajectory.npz")
        
        # Create README
        readme_path = os.path.join(temp_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write("""# MuJoCo Trajectory Visualization

## Files
- `model.xml`: MuJoCo model definition
- `*_trajectory.npz`: Trajectory data (qpos, qvel)

## Usage
```python
import numpy as np
import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)
traj = np.load("best_trajectory.npz")

qpos_trajectory = traj['qpos']
qvel_trajectory = traj['qvel']
timestep = float(traj['timestep'])

with mujoco.viewer.launch_passive(model, data) as viewer:
    i = 0
    while viewer.is_running():
        i = i % len(qpos_trajectory)
        data.qpos[:] = qpos_trajectory[i]
        data.qvel[:] = qvel_trajectory[i]
        mujoco.mj_forward(model, data)
        viewer.sync()
        
        import time
        time.sleep(timestep)
        i += 1
```
""")
        
        # Upload to wandb
        artifact = wandb.Artifact(
            f"mujoco_trajectories_{run.name}",
            type="trajectory",
            description="MuJoCo XML and trajectory data for rendering"
        )
        artifact.add_dir(temp_dir)
        run.log_artifact(artifact)
        
        # Log stats
        trajectory_stats = {
            f"trajectory/{name}_num_steps": len(episode_data)
            for name, episode_data in episodes_dict.items()
            if isinstance(episode_data, list)
        }
        wandb.log(trajectory_stats)
    
    print("MuJoCo trajectories uploaded successfully!")

def upload_mujoco_videos_to_wandb(eval_env, episodes_dict, run, fps=30, quality="high", width=1280, height=720):
    """
    Render MuJoCo episodes as videos and upload to wandb.
    
    Args:
        eval_env: Evaluation environment (Gymnax-style wrapper with env.env.sys)
        episodes_dict: Dict mapping names to episode data (list of brax.mjx.base.State objects)
        run: wandb run object
        fps: Frames per second for the video
        quality: Video quality - "low", "medium", or "high"
        width: Video width in pixels
        height: Video height in pixels
    """
    import tempfile
    import os
    import numpy as np
    import mujoco
    import mediapy as media
    
    print(f"Rendering MuJoCo videos at {width}x{height}, {fps} FPS, {quality} quality...")
    
    # Get the XML file path from the environment
    if hasattr(eval_env.env, 'path'):
        xml_source_path = str(eval_env.env.path)
        print(f"  Using XML from: {xml_source_path}")
    else:
        print("  Warning: Environment does not have 'path' attribute. Skipping.")
        return
    
    # Quality settings (bps = bits per second)
    #quality_settings = {
    #    "low": 2_000_000,      # 2 Mbps
    #    "medium": 5_000_000,   # 5 Mbps
    #    "high": 10_000_000,    # 10 Mbps
    #}
    #bps = quality_settings.get(quality, 5_000_000)
    quality_settings = {
    "low": 23,       # Default quality
    "medium": 18,    # High quality
    "high": 15,      # Very high quality
    }
    crf = quality_settings.get(quality, 18)
    with tempfile.TemporaryDirectory() as temp_dir:
        # Load MuJoCo model
        model = mujoco.MjModel.from_xml_path(xml_source_path)
        data = mujoco.MjData(model)
        
        # Check framebuffer size and adjust if necessary
        max_offscreen_width = model.vis.global_.offwidth
        max_offscreen_height = model.vis.global_.offheight
        
        if width > max_offscreen_width or height > max_offscreen_height:
            print(f"  Warning: Requested size {width}x{height} exceeds framebuffer size {max_offscreen_width}x{max_offscreen_height}")
            # Scale down while maintaining aspect ratio
            scale = min(max_offscreen_width / width, max_offscreen_height / height)
            width = int(width * scale)
            height = int(height * scale)
            print(f"  Adjusted to {width}x{height}")
        
        # Create renderer
        renderer = mujoco.Renderer(model, height=height, width=width)
        
        video_paths = {}
        
        for name, episode_data in episodes_dict.items():
            if not isinstance(episode_data, list) or len(episode_data) == 0:
                print(f"  Warning: {name} episode is invalid")
                continue
            
            print(f"  Rendering {name} episode ({len(episode_data)} frames)...")
            
            # Extract trajectories
            qpos_trajectory = np.array([np.array(state.qpos) for state in episode_data])
            qvel_trajectory = np.array([np.array(state.qvel) for state in episode_data])
            
            # Render frames
            frames = []
            for i in range(len(qpos_trajectory)):
                # Set state
                data.qpos[:] = qpos_trajectory[i]
                data.qvel[:] = qvel_trajectory[i]
                
                # Forward kinematics
                mujoco.mj_forward(model, data)
                
                # Render frame
                renderer.update_scene(data, camera="default")
                frame = renderer.render()
                frames.append(frame)
            
            # Save video using mediapy with correct parameters
            video_path = os.path.join(temp_dir, f"{name}_episode.mp4")
            media.write_video(
                video_path, 
                frames, 
                fps=fps,
                #bps=bps, # bits per second
                crf=crf,
                codec='h264'
            )
            video_paths[name] = video_path
            
            duration = len(episode_data) * eval_env.env.dt
            print(f"    Saved {name}_episode.mp4 ({duration:.2f}s)")
        
        # Close renderer to avoid the __del__ error
        renderer.close()
        
        # Upload to wandb
        print("  Uploading videos to wandb...")
        
        # Create artifact for downloads
        artifact = wandb.Artifact(
            f"mujoco_videos_{run.name}",
            type="video",
            description=f"Rendered episodes at {width}x{height}, {fps}fps"
        )
        for name, video_path in video_paths.items():
            artifact.add_file(video_path, name=f"{name}_episode.mp4")
        run.log_artifact(artifact)
        
        # Log for inline viewing
        for name, video_path in video_paths.items():
            wandb.log({f"video/{name}_episode": wandb.Video(video_path, fps=fps, format="mp4")})
        
        print("  Videos uploaded successfully!")
    
    print("MuJoCo videos uploaded to wandb!")


# ================================ PREFERENCE SWEEP UTILITIES ================================

def generate_preference_configs(
    rng: jax.random.PRNGKey,
    pref_sweep_config: Dict,
    base_config: Dict,
) -> Dict[str, jnp.ndarray]:
    """Sample unique preference weight + range combinations for vmapped zoo generation.

    For each parameter, if a ``{min, max}`` meta-range is specified in
    *pref_sweep_config* the value is sampled uniformly.  Otherwise the
    fixed value from *base_config* is broadcast to ``(num_configs,)``.

    Range constraints ``speed_range_min < speed_range_max`` and
    ``force_range_min < force_range_max`` are enforced by sampling min
    first, then max from ``[sampled_min, meta_max]``.

    Args:
        rng: JAX PRNG key.
        pref_sweep_config: Dict with ``num_configs`` and optional
            per-parameter ``{min, max}`` sub-dicts.
        base_config: Full training config (used for fallback values).

    Returns:
        Dict of JAX arrays each shaped ``(num_configs,)``.
    """
    n = pref_sweep_config["num_configs"]
    pref_base = base_config["ENV_KWARGS"]["preference_rewards"]

    def _sample_or_fixed(rng_key, param_name, fallback):
        spec = pref_sweep_config.get(param_name, None)
        if spec is not None and isinstance(spec, dict):
            return jax.random.uniform(rng_key, shape=(n,), minval=spec["min"], maxval=spec["max"])
        return jnp.full((n,), fallback)

    keys = jax.random.split(rng, 10)

    w_speed = _sample_or_fixed(keys[0], "w_speed", pref_base["preference_weights"]["speed_preference"])
    w_force = _sample_or_fixed(keys[1], "w_force", pref_base["preference_weights"]["force_preference"])
    w_touch = _sample_or_fixed(keys[2], "w_touch", pref_base["preference_weights"]["touch_penalty"])

    # Speed range: enforce min < max
    speed_range_min = _sample_or_fixed(keys[3], "speed_range_min", pref_base["preference_ranges"]["speed_range"][0])
    speed_max_spec = pref_sweep_config.get("speed_range_max", None)
    if speed_max_spec is not None and isinstance(speed_max_spec, dict):
        speed_range_max = jax.random.uniform(keys[4], shape=(n,), minval=speed_range_min, maxval=speed_max_spec["max"])
    else:
        speed_range_max = jnp.full((n,), pref_base["preference_ranges"]["speed_range"][1])

    # Force range: enforce min < max
    force_range_min = _sample_or_fixed(keys[5], "force_range_min", pref_base["preference_ranges"]["force_range"][0])
    force_max_spec = pref_sweep_config.get("force_range_max", None)
    if force_max_spec is not None and isinstance(force_max_spec, dict):
        force_range_max = jax.random.uniform(keys[6], shape=(n,), minval=force_range_min, maxval=force_max_spec["max"])
    else:
        force_range_max = jnp.full((n,), pref_base["preference_ranges"]["force_range"][1])

    reward_budget = _sample_or_fixed(keys[7], "reward_budget", pref_base.get("reward_budget", 1.0))
    overall_weight = _sample_or_fixed(keys[8], "overall_weight", pref_base.get("overall_weight", 1.0))
    touch_threshold = _sample_or_fixed(keys[9], "touch_threshold", pref_base.get("touch_threshold", 0.3))

    return {
        "w_speed": w_speed,
        "w_force": w_force,
        "w_touch": w_touch,
        "speed_range_min": speed_range_min,
        "speed_range_max": speed_range_max,
        "force_range_min": force_range_min,
        "force_range_max": force_range_max,
        "reward_budget": reward_budget,
        "overall_weight": overall_weight,
        "touch_threshold": touch_threshold,
    }


def extract_pref_config_at_index(pref_configs: Dict[str, jnp.ndarray], idx: int) -> Dict[str, float]:
    """Extract a single preference config from batched arrays as Python floats.

    Args:
        pref_configs: Dict of arrays each shaped ``(num_configs,)``.
        idx: Index to extract.

    Returns:
        Dict of Python floats for the given index.
    """
    return {k: float(v[idx]) for k, v in pref_configs.items()}


def print_memory_stats(label=""):
    """Prints the true peak memory used by JAX on the primary GPU."""
    try:
        # Use local_devices to ensure we target the specific GPU this process is using
        device = jax.local_devices()[0]
        stats = device.memory_stats()
        peak_gb = stats['peak_bytes_in_use'] / 1e9
        # This is the string the Bash script will 'grep' for
        print(f"--- MEMORY_STATS_PEAK: {peak_gb:.4f} GB ---")
        print("----------------------------------------------")
        print(f"All Stats: {stats}")
    except Exception:
        print("--- MEMORY_STATS_PEAK: 0.0000 GB ---")

def log_memory_to_csv(config, csv_path="", filename="memory_benchmark.csv"):
    try:
        # 1. Gather stats
        jax.block_until_ready(None) # Ensure GPU is finished
        device = jax.local_devices()[0]
        stats = device.memory_stats()
        peak_gb = stats['peak_bytes_in_use'] / 1e9
        
        # 2. Prepare the data row
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": config.ALG,
            "network": config.network,
            "env_name": config.ENV_NAME,
            "num_configs": config.SWEEP.num_configs,
            "num_seeds": config.NUM_SEEDS,
            "total_parallel": config.SWEEP.num_configs * config.NUM_SEEDS,
            "peak_vram_gb": round(peak_gb, 4)
        }
        
        # 3. Append to CSV (thread-safe enough for local machine)
        df = pd.DataFrame([data])
        full_path = os.path.join(csv_path, filename) 
        file_exists = os.path.isfile(full_path)
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)
        
        print(f"--> Memory benchmark saved to {csv_path}")
        
    except Exception as e:
        print(f"Failed to log memory: {e}")

