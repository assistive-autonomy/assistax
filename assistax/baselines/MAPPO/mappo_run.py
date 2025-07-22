"""
MAPPO Comprehensive Training and Evaluation Runner

This script provides a complete end-to-end pipeline for MAPPO experiments.
"""

import os
import time
from typing import Dict, List, Tuple, Any, Optional

import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from flax.traverse_util import flatten_dict
import safetensors.flax

import assistax


# ============================================================================
# TREE UTILITY FUNCTIONS
# ============================================================================

def _tree_take(pytree, indices, axis=None):
    """Take elements from pytree along specified axis."""
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)


def _tree_shape(pytree):
    """Get shapes of all leaves in pytree for debugging."""
    return jax.tree.map(lambda x: x.shape, pytree)


def _unstack_tree(pytree):
    """
    Unstack a pytree along its first axis.
    
    Converts a tree with arrays of shape (n, ...) to a list of n trees
    with arrays of shape (...).
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]


def _stack_tree(pytree_list, axis=0):
    """Stack a list of pytrees along specified axis."""
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )


def _concat_tree(pytree_list, axis=0):
    """Concatenate a list of pytrees along specified axis."""
    return jax.tree.map(
        lambda *leaf: jnp.concat(leaf, axis=axis),
        *pytree_list
    )


def _tree_split(pytree, n, axis=0):
    """Split pytree into n parts along specified axis."""
    leaves, treedef = jax.tree.flatten(pytree)
    split_leaves = zip(
        *jax.tree.map(lambda x: jnp.array_split(x, n, axis), leaves)
    )
    return [
        jax.tree.unflatten(treedef, leaves)
        for leaves in split_leaves
    ]


# ============================================================================
# EVALUATION UTILITY FUNCTIONS
# ============================================================================

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    """
    Extract episode data from pipeline states for visualization.
    
    Args:
        pipeline_states: Full pipeline state data
        dones: Done flags for episodes
        time_idx: Time index to extract from
        eval_idx: Evaluation index to extract from
        
    Returns:
        List of episode states that are not done
    """
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not done
    ]


def _compute_episode_returns(eval_info, time_axis=-2):
    """
    Compute undiscounted episode returns from evaluation info.
    
    Args:
        eval_info: Evaluation information containing rewards and done flags
        time_axis: Axis along which time is indexed
        
    Returns:
        Dictionary of undiscounted returns per agent
    """
    done_arr = eval_info.done["__all__"]
    
    # Create episode mask to separate different episodes
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    
    # Compute returns by masking out rewards from previous episodes
    undiscounted_returns = jax.tree.map(
        lambda r: (r * (1 - episode_done)).sum(axis=time_axis),
        eval_info.reward
    )
    
    return undiscounted_returns


# ============================================================================
# ARCHITECTURE SELECTION
# ============================================================================

def _import_mappo_variant(config: Dict):
    """
    Import the appropriate MAPPO variant based on configuration.
    
    Supports all four combinations:
    - FF + NPS: Feedforward with No Parameter Sharing
    - FF + PS: Feedforward with Parameter Sharing  
    - RNN + NPS: Recurrent with No Parameter Sharing
    - RNN + PS: Recurrent with Parameter Sharing
    
    Args:
        config: Configuration dictionary specifying architecture
        
    Returns:
        Tuple of (make_train, make_evaluation, EvalInfoLogConfig) functions
        
    Raises:
        ImportError: If the specified architecture files are not found
    """
    recurrent = config["network"]["recurrent"]
    param_sharing = config["network"]["agent_param_sharing"]
    
    try:
        match (recurrent, param_sharing):
            case (False, False):
                # Feedforward, No Parameter Sharing
                from mappo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
            case (False, True):
                # Feedforward, Parameter Sharing
                from mappo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
            case (True, False):
                # Recurrent, No Parameter Sharing
                from mappo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
            case (True, True):
                # Recurrent, Parameter Sharing
                from mappo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
                
        return make_train, make_evaluation, EvalInfoLogConfig
        
    except ImportError as e:
        architecture_name = f"{'RNN' if recurrent else 'FF'}_{'PS' if param_sharing else 'NPS'}"
        raise ImportError(
            f"Could not import MAPPO variant for {architecture_name}. "
            f"Please ensure the corresponding module exists. Error: {e}"
        )
# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def _run_mappo_training(config: Dict) -> Dict:
    """
    Run MAPPO training across multiple seeds.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing training results
    """
    print(f"\n{'='*60}")
    print("STARTING MAPPO TRAINING")
    print(f"{'='*60}")
    
    # Import architecture-specific functions
    make_train, _, _ = _import_mappo_variant(config)
    
    # Setup random number generation
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    print(f"Training with {config['NUM_SEEDS']} seeds for statistical robustness...")
    print(f"Total environment interactions: {config['TOTAL_TIMESTEPS'] * config['NUM_SEEDS']:,}")
    
    # Run training
    start_time = time.time()
    
    with jax.disable_jit(config["DISABLE_JIT"]):
        # Create JIT-compiled training function
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        
        print("Compiling and running training (first run includes JIT compilation)...")
        
        # Train across multiple seeds in parallel
        training_results = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], 
            config["ENT_COEF"], 
            config["CLIP_EPS"]
        )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return {
        "training_results": training_results,
        "training_time": training_time,
        "eval_rng": eval_rng
    }


# ============================================================================
# RESULT SAVING FUNCTIONS
# ============================================================================

def _save_training_metrics(training_results: Dict):
    """Save training metrics excluding large objects."""
    
    EXCLUDED_METRICS = ["train_state"]
    metrics = training_results["training_results"]["metrics"]
    filtered_metrics = {
        key: val for key, val in metrics.items() 
        if key not in EXCLUDED_METRICS
    }
    
    jnp.save("metrics.npy", filtered_metrics, allow_pickle=True)


def _save_model_parameters(config: Dict, training_results: Dict):
    """Save model parameters based on architecture type."""
    
    # Get training states
    all_train_states = training_results["training_results"]["metrics"]["train_state"]
    final_train_state = training_results["training_results"]["runner_state"].train_state
    
    # Save all training states (across all updates and seeds)
    safetensors.flax.save_file(
        flatten_dict(all_train_states.actor.params, sep='/'),
        "all_params.safetensors"
    )
    
    # Save final parameters based on architecture
    if config["network"]["agent_param_sharing"]:
        # Parameter sharing: single set of parameters
        safetensors.flax.save_file(
            flatten_dict(final_train_state.actor.params, sep='/'),
            "final_params.safetensors"
        )
    else:
        # No parameter sharing: separate parameters per agent
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        split_params = _unstack_tree(
            jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state.actor.params)
        )
        
        for agent, params in zip(env.agents, split_params):
            safetensors.flax.save_file(
                flatten_dict(params, sep='/'),
                f"{agent}.safetensors",
            )


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def _run_comprehensive_evaluation(config: Dict, training_results: Dict) -> Dict:
    """
    Run comprehensive evaluation of trained models.
    
    Args:
        config: Configuration dictionary
        training_results: Results from training pipeline
        
    Returns:
        Dictionary containing evaluation results
    """    
    # Import evaluation functions
    _, make_evaluation, EvalInfoLogConfig = _import_mappo_variant(config)
    
    # Get training states
    all_train_states = training_results["training_results"]["metrics"]["train_state"]
    eval_rng = training_results["eval_rng"]
    
    # Calculate evaluation batching for memory efficiency
    batch_dims = jax.tree.leaves(_tree_shape(all_train_states.actor.params))[:2]
    total_evaluations = config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
    n_sequential_evals = int(jnp.ceil(total_evaluations / config["GPU_ENV_CAPACITY"]))

    
    def _flatten_and_split_trainstate(trainstate):
        """Flatten and split training states for batched evaluation."""
        flat_trainstate = jax.tree.map(
            lambda x: x.reshape((x.shape[0]*x.shape[1], *x.shape[2:])),
            trainstate
        )
        return _tree_split(flat_trainstate, n_sequential_evals)
    
    # JIT compile for memory efficiency
    split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states.actor)
    
    # Setup evaluation environment
    eval_env, run_eval = make_evaluation(config)
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
    
    # JIT compile evaluation
    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
    eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
    
    # Run evaluation in batches
    print(f"Running evaluation across {len(split_trainstate)} batches...")
    start_time = time.time()
    
    evals = _concat_tree([
        eval_vmap(eval_rng, ts, eval_log_config)
        for ts in tqdm(split_trainstate, desc="Evaluation batches")
    ])
    
    # Reshape back to original batch dimensions
    evals = jax.tree.map(
        lambda x: x.reshape((*batch_dims, *x.shape[1:])),
        evals
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    return {
        "evals": evals,
        "eval_env": eval_env,
        "eval_time": eval_time
    }


def _analyze_performance(evaluation_results: Dict) -> Dict:
    """
    Analyze performance and compute episode returns.
    
    Args:
        evaluation_results: Results from evaluation pipeline
        
    Returns:
        Dictionary containing performance analysis
    """
    print(f"\nAnalyzing performance...")
    
    # Compute episode returns
    first_episode_returns = _compute_episode_returns(evaluation_results["evals"])
    first_episode_returns = first_episode_returns["__all__"]
    mean_episode_returns = first_episode_returns.mean(axis=-1)
    
    # Calculate statistics
    performance_stats = {
        "mean_return": float(jnp.mean(mean_episode_returns)),
        "std_return": float(jnp.std(mean_episode_returns)),
        "min_return": float(jnp.min(mean_episode_returns)),
        "max_return": float(jnp.max(mean_episode_returns)),
        "median_return": float(jnp.median(mean_episode_returns))
    }
    
    print(f"Performance Statistics:")
    print(f"  Mean return: {performance_stats['mean_return']:.4f}")
    print(f"  Std return: {performance_stats['std_return']:.4f}")
    print(f"  Min return: {performance_stats['min_return']:.4f}")
    print(f"  Max return: {performance_stats['max_return']:.4f}")
    print(f"  Median return: {performance_stats['median_return']:.4f}")
    
    # Save returns
    jnp.save("returns.npy", mean_episode_returns)
    print(f"  ✓ Episode returns saved to returns.npy")
    
    return {
        "mean_episode_returns": mean_episode_returns,
        "first_episode_returns": first_episode_returns,
        "performance_stats": performance_stats
    }


# ============================================================================
# VISUALIZATION GENERATION
# ============================================================================

def _generate_episode_visualizations(config: Dict, training_results: Dict, 
                                   evaluation_results: Dict) -> Dict:
    """
    Generate HTML visualizations of best, worst, and median episodes.
    
    This provides valuable qualitative analysis of the learned behaviors.
    
    Args:
        config: Configuration dictionary
        training_results: Results from training pipeline
        evaluation_results: Results from evaluation pipeline
        
    Returns:
        Dictionary containing visualization metadata
    """
    print(f"\n{'='*60}")
    print("GENERATING EPISODE VISUALIZATIONS")
    print(f"{'='*60}")
    
    # Check if environment supports visualization
    try:
        from brax.io import html
    except ImportError:
        print("Brax not available for visualization. Skipping episode rendering.")
        return {"visualizations_generated": False, "reason": "brax_not_available"}
    
    # Import evaluation functions for rendering
    _, make_evaluation, EvalInfoLogConfig = _import_mappo_variant(config)
    
    # Setup for rendering (need env_state for visualization)
    final_train_state = training_results["training_results"]["runner_state"].train_state
    eval_rng = training_results["eval_rng"]
    
    # Create evaluation environment
    eval_env, run_eval = make_evaluation(config)
    
    # Configure to save environment states for rendering
    render_log_config = EvalInfoLogConfig(
        env_state=True,  # Need this for visualization
        done=True,
        action=False,
        value=False,
        reward=True,
        log_prob=False,
        obs=False,
        info=False,
        avail_actions=False,
    )
    
    print("Running episodes for visualization...")
    
    # JIT compile evaluation for rendering
    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
    
    # Run evaluation with first seed to get episodes for rendering
    eval_final = eval_jit(
        eval_rng, 
        _tree_take(final_train_state.actor, 0, axis=0), 
        render_log_config
    )
    
    # Process episodes to find best, worst, and median
    first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
    first_episode_rewards = eval_final.reward["__all__"] * (1 - first_episode_done)
    first_episode_returns = first_episode_rewards.sum(axis=0)
    episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
    
    # Get indices for worst, best, and median episodes
    worst_idx = episode_argsort.take(0, axis=-1)
    best_idx = episode_argsort.take(-1, axis=-1)
    median_idx = episode_argsort.take(episode_argsort.shape[-1] // 2, axis=-1)
    
    # Extract episode data
    worst_episode = _take_episode(
        eval_final.env_state.env_state.pipeline_state, 
        first_episode_done,
        time_idx=-1, 
        eval_idx=worst_idx,
    )
    median_episode = _take_episode(
        eval_final.env_state.env_state.pipeline_state, 
        first_episode_done,
        time_idx=-1, 
        eval_idx=median_idx,
    )
    best_episode = _take_episode(
        eval_final.env_state.env_state.pipeline_state, 
        first_episode_done,
        time_idx=-1, 
        eval_idx=best_idx,
    )
    
    # Generate HTML visualizations
    try:
        html.save("final_worst.html", eval_env.sys, worst_episode)
        html.save("final_median.html", eval_env.sys, median_episode)
        html.save("final_best.html", eval_env.sys, best_episode)
        
        # Calculate episode returns for reference
        worst_return = float(first_episode_returns[worst_idx])
        median_return = float(first_episode_returns[median_idx])
        best_return = float(first_episode_returns[best_idx])
        
        print(f"Episode visualizations generated:")
        print(f"  ✓ Worst episode (return: {worst_return:.4f}) → final_worst.html")
        print(f"  ✓ Median episode (return: {median_return:.4f}) → final_median.html")
        print(f"  ✓ Best episode (return: {best_return:.4f}) → final_best.html")
        
        return {
            "visualizations_generated": True,
            "worst_return": worst_return,
            "median_return": median_return,
            "best_return": best_return,
            "files": ["final_worst.html", "final_median.html", "final_best.html"]
        }
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return {"visualizations_generated": False, "reason": str(e)}


# ============================================================================
# MAIN RUNNER FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax")
def main(config):
    """
    Main function for comprehensive MAPPO training and evaluation.
    
    This function orchestrates a complete MAPPO experimental pipeline:
    
    1. **Configuration Setup**: Display experiment parameters and auto-select architecture
    2. **Training**: Multi-seed MAPPO training with the specified configuration
    3. **Model Saving**: Save training metrics and model parameters
    4. **Evaluation**: Comprehensive performance evaluation across all trained models
    5. **Analysis**: Compute performance statistics and episode returns
    6. **Visualization**: Generate HTML visualizations of representative episodes
    
    The script is designed to be a one-stop solution for running MAPPO experiments,
    providing everything from training to analysis and visualization.
    
    Args:
        config: Hydra configuration object
    """
    
    # ========================================================================
    # SETUP AND CONFIGURATION
    # ========================================================================
    
    config = OmegaConf.to_container(config, resolve=True)
    
    print("="*80)
    print("MAPPO TRAINING & EVALUATION RUNNING")
    print("="*80)
       
    # ========================================================================
    # RUN TRAINING PIPELINE
    # ========================================================================
    total_start_time = time.time()
    try:
        training_results = _run_mappo_training(config)
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    # ========================================================================
    # SAVE TRAINING RESULTS
    # ========================================================================
    
    try:
        _save_training_metrics(training_results)
        _save_model_parameters(config, training_results)
        
    except Exception as e:
        print(f"Error saving training results: {e}")
        raise
    
    # ========================================================================
    # RUN EVALUATION PIPELINE
    # ========================================================================
    
    try:
        evaluation_results = _run_comprehensive_evaluation(config, training_results)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    
    # ========================================================================
    # ANALYZE PERFORMANCE
    # ========================================================================
    
    try:
        performance_analysis = _analyze_performance(evaluation_results)
        
    except Exception as e:
        print(f"Error during performance analysis: {e}")
        raise
    
    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================
    
    try:
        visualization_results = _generate_episode_visualizations(
            config, training_results, evaluation_results
        )
        
    except Exception as e:
        print(f"Warning: Visualization generation failed: {e}")
        visualization_results = {"visualizations_generated": False, "reason": str(e)}
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"MAPPO Finished running")
    print(f"{'='*80}")
    
    # Time summary
    print(f"\nTiming Summary:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Training time: {training_results['training_time']:.2f} seconds")
    print(f"  Evaluation time: {evaluation_results['eval_time']:.2f} seconds")
        
    # Performance summary
    stats = performance_analysis["performance_stats"]
    print(f"\nFinal Performance:")
    print(f"  Mean return: {stats['mean_return']:.4f} ± {stats['std_return']:.4f}")
    print(f"  Range: [{stats['min_return']:.4f}, {stats['max_return']:.4f}]")
    print(f"  Median: {stats['median_return']:.4f}")
    

if __name__ == "__main__":
    main()
