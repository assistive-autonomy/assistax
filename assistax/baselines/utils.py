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
    env_steps = train_metrics["env_step"]  # Shape: (num_seeds, num_updates)
    x_axis = env_steps[0]  # Use first seed
    num_checkpoints = len(x_axis)
    
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
                
    # Losses and other metrics
    loss_metrics = ["total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl", 
                   "clip_frac_min", "clip_frac_max"]
    
    for metric_key in loss_metrics:
        if metric_key not in train_metrics:
            continue
            
        metric_values = train_metrics[metric_key]
        
        if len(metric_values.shape) == 3:  # Has agent dimension
            for agent_idx, agent_name in enumerate(agent_names):
                agent_metric = metric_values[:, :, agent_idx]
                prefix = "train/loss" if metric_key in ["total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl"] else "train/diagnostics"
                train_stats[f"{prefix}/{agent_name}_{metric_key}"] = {
                    'mean': jnp.mean(agent_metric, axis=0),
                    'std': jnp.std(agent_metric, axis=0),
                }
        else:  # No agent dimension
            prefix = "train/loss" if metric_key in ["total_loss", "actor_loss", "critic_loss", "entropy", "approx_kl"] else "train/diagnostics"
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

def upload_html_visualizations_to_wandb(eval_env, episodes_dict, run):
    """
    Upload HTML visualizations to wandb as artifacts.
    
    Args:
        eval_env: Evaluation environment (for sys attribute)
        episodes_dict: Dict mapping names to episode data
                      e.g., {'worst': worst_episode, 'median': median_episode, 'best': best_episode}
        run: wandb run object
    """
    from assistax.render import html
    import tempfile
    import os
    
    print("Creating and uploading HTML visualizations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate HTML files in temporary directory
        for name, episode_data in episodes_dict.items():
            file_path = os.path.join(temp_dir, f"final_{name}.html")
            html.save(file_path, eval_env.sys, episode_data)
            print(f"  Generated final_{name}.html")
        
        # Create and upload artifact
        artifact = wandb.Artifact(f"visualizations_{run.name}", type="visualization")
        artifact.add_dir(temp_dir)
        run.log_artifact(artifact)
    
    print("HTML visualizations uploaded to wandb successfully!")