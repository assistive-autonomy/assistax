"""
ISAC (Independent Soft Actor-Critic) Comprehensive Training and Evaluation Runner

This script provides a complete end-to-end pipeline for ISAC experiments, an off-policy
multi-agent reinforcement learning algorithm based on Soft Actor-Critic (SAC).
"""

import os
import sys
import time
from typing import Dict, List, Tuple, Any, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
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
    return jax.tree_util.tree_map(lambda x: x.take(indices, axis=axis), pytree)


def _tree_shape(pytree):
    """Get shapes of all leaves in pytree for debugging."""
    return jax.tree_util.tree_map(lambda x: x.shape, pytree)


def _unstack_tree(pytree):
    """
    Unstack a pytree along its first axis.
    
    Converts a tree with arrays of shape (n, ...) to a list of n trees
    with arrays of shape (...).
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = list(zip(*leaves))
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]


def _concat_tree(pytree_list, axis=0):
    """Concatenate a list of pytrees along specified axis."""
    return jax.tree_util.tree_map(
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


def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """Convert dictionary of agent-specific arrays to batched array."""
    return jnp.stack(tuple(qty[a] for a in agents))


def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array back to dictionary of agent-specific arrays."""
    return dict(zip(agents, qty))


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
# ARCHITECTURE SELECTION AND CONFIGURATION
# ============================================================================

def _import_isac_variant(config: Dict):
    """
    Import the appropriate ISAC variant based on configuration.
    
    Currently supports feedforward networks with no parameter sharing.
    Future versions may include RNN and parameter sharing variants.
    
    Args:
        config: Configuration dictionary specifying architecture
        
    Returns:
        Tuple of (make_train, make_evaluation, EvalInfoLogConfig) functions
        
    Raises:
        ImportError: If the specified architecture files are not found
    """
    try:
        # Currently only feedforward, no parameter sharing is implemented
        from isac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
        return make_train, make_evaluation, EvalInfoLogConfig
        
    except ImportError as e:
        raise ImportError(
            f"Could not import ISAC variant. "
            f"Please ensure isac_ff_nps_mabrax.py exists. Error: {e}"
        )

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def _run_isac_training(config: Dict) -> Dict:
    """
    Run ISAC training across multiple seeds.
    
    ISAC uses three separate networks (actor, Q1, Q2) with different learning rates
    and soft target updates for stable learning.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing training results
    """
    print(f"\n{'='*60}")
    print("STARTING ISAC TRAINING")
    print(f"{'='*60}")
    
    # Import ISAC-specific functions
    make_train, _, _ = _import_isac_variant(config)
    
    # Setup random number generation
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    print(f"Training {config['NUM_SEEDS']} seeds with ISAC algorithm...")
    print(f"Total environment interactions: {config['TOTAL_TIMESTEPS']}")
    
    # Run training
    start_time = time.time()
    
    with jax.disable_jit(config["DISABLE_JIT"]):
        # Create JIT-compiled training function
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        
        print("Compiling and running ISAC training (first run includes JIT compilation)...")
        
        # Train across multiple seeds in parallel
        # ISAC has different hyperparameters than MAPPO
        training_results = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
            train_rngs,
            config["POLICY_LR"],  # Policy network learning rate
            config["Q_LR"],       # Q-network learning rate  
            config["ALPHA_LR"],   # Entropy temperature learning rate
            config["TAU"]         # Soft update coefficient
        )
    
    training_time = time.time() - start_time
    print(f"ISAC training completed in {training_time:.2f} seconds")
    
    return {
        "training_results": training_results,
        "training_time": training_time,
        "eval_rng": eval_rng
    }


# ============================================================================
# RESULT SAVING FUNCTIONS
# ============================================================================

def _save_training_metrics(training_results: Dict):
    """Save ISAC training metrics excluding large objects."""
    
    # ISAC has different training states than MAPPO
    EXCLUDED_METRICS = ["actor_train_state", "q1_train_state", "q2_train_state"]
    metrics = training_results["training_results"]["metrics"]
    filtered_metrics = {
        key: val for key, val in metrics.items() 
        if key not in EXCLUDED_METRICS
    }
    
    jnp.save("metrics.npy", filtered_metrics, allow_pickle=True)


import os
from typing import Dict, List

def _save_model_parameters(config: Dict, training_results: Dict) -> Dict[str, str]:
    """Save ISAC model parameters (actor, Q1, Q2 networks) and return file paths."""
    # Get all training states for ISAC networks
    metrics = training_results["training_results"]["metrics"]
    runner_state = training_results["training_results"]["runner_state"]
    all_train_states_actor = metrics["actor_train_state"]
    all_train_states_q1 = metrics["q1_train_state"]
    all_train_states_q2 = metrics["q2_train_state"]
    final_train_state_actor = runner_state.train_states.actor
    final_train_state_q1 = runner_state.train_states.q1
    final_train_state_q2 = runner_state.train_states.q2
    
    # Dictionary to store all saved file paths
    saved_paths = {}
    
    # Save all training states (across all updates and seeds)
    actor_all_path = "actor_all_params.safetensors"
    safetensors.flax.save_file(
        flatten_dict(all_train_states_actor.params, sep='/'),
        actor_all_path
    )
    saved_paths["actor_all"] = os.path.abspath(actor_all_path)
    
    q1_all_path = "q1_all_params.safetensors"
    safetensors.flax.save_file(
        flatten_dict(all_train_states_q1.params, sep='/'),
        q1_all_path
    )
    saved_paths["q1_all"] = os.path.abspath(q1_all_path)
    
    q2_all_path = "q2_all_params.safetensors"
    safetensors.flax.save_file(
        flatten_dict(all_train_states_q2.params, sep='/'),
        q2_all_path
    )
    saved_paths["q2_all"] = os.path.abspath(q2_all_path)
    
    # Save final parameters based on architecture
    if config["network"]["agent_param_sharing"]:
        # Parameter sharing: single set of parameters for each network
        actor_final_path = "actor_final_params.safetensors"
        safetensors.flax.save_file(
            flatten_dict(final_train_state_actor.params, sep='/'),
            actor_final_path
        )
        saved_paths["actor_final"] = os.path.abspath(actor_final_path)
        
        q1_final_path = "q1_final_params.safetensors"
        safetensors.flax.save_file(
            flatten_dict(final_train_state_q1.params, sep='/'),
            q1_final_path
        )
        saved_paths["q1_final"] = os.path.abspath(q1_final_path)
        
        q2_final_path = "q2_final_params.safetensors"
        safetensors.flax.save_file(
            flatten_dict(final_train_state_q2.params, sep='/'),
            q2_final_path
        )
        saved_paths["q2_final"] = os.path.abspath(q2_final_path)
        
    else:
        # No parameter sharing: separate parameters per agent for each network
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        
        # Split actor parameters by agent
        split_actor_params = _unstack_tree(
            jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_actor.params)
        )
        for agent, params in zip(env.agents, split_actor_params):
            actor_agent_path = f"actor_{agent}.safetensors"
            safetensors.flax.save_file(
                flatten_dict(params, sep='/'),
                actor_agent_path,
            )
            saved_paths[f"actor_{agent}"] = os.path.abspath(actor_agent_path)
        
        # Split Q1 parameters by agent
        split_q1_params = _unstack_tree(
            jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_q1.params)
        )
        for agent, params in zip(env.agents, split_q1_params):
            q1_agent_path = f"q1_{agent}.safetensors"
            safetensors.flax.save_file(
                flatten_dict(params, sep='/'),
                q1_agent_path,
            )
            saved_paths[f"q1_{agent}"] = os.path.abspath(q1_agent_path)
        
        # Split Q2 parameters by agent
        split_q2_params = _unstack_tree(
            jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_q2.params)
        )
        for agent, params in zip(env.agents, split_q2_params):
            q2_agent_path = f"q2_{agent}.safetensors"
            safetensors.flax.save_file(
                flatten_dict(params, sep='/'),
                q2_agent_path,
            )
            saved_paths[f"q2_{agent}"] = os.path.abspath(q2_agent_path)
    
    return saved_paths


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def _run_comprehensive_evaluation(config: Dict, training_results: Dict) -> Dict:
    """
    Run comprehensive evaluation of trained ISAC models.
    
    Args:
        config: Configuration dictionary
        training_results: Results from training pipeline
        
    Returns:
        Dictionary containing evaluation results
    """
    # Import evaluation functions
    _, make_evaluation, EvalInfoLogConfig = _import_isac_variant(config)
    
    # Get actor training states (only actor is needed for evaluation)
    all_train_states_actor = training_results["training_results"]["metrics"]["actor_train_state"]
    eval_rng = training_results["eval_rng"]
    
    # Calculate evaluation batching for memory efficiency
    batch_dims = jax.tree.leaves(_tree_shape(all_train_states_actor.params))[:2]
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
    split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states_actor)
    
    # Setup evaluation environment
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
    
    # JIT compile evaluation
    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
    eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
    
    # Run evaluation in batches
    start_time = time.time()
    
    evals = _concat_tree([
        eval_vmap(eval_rng, ts, eval_log_config)
        for ts in tqdm(split_trainstate, desc="ISAC evaluation batches")
    ])
    
    # Reshape back to original batch dimensions
    evals = jax.tree.map(
        lambda x: x.reshape((*batch_dims, *x.shape[1:])),
        evals
    )
    
    eval_time = time.time() - start_time
    print(f"ISAC evaluation completed in {eval_time:.2f} seconds")
    
    return {
        "evals": evals,
        "eval_env": eval_env,
        "eval_time": eval_time
    }


def _analyze_performance(evaluation_results: Dict) -> Dict:
    """
    Analyze ISAC performance and compute episode returns.
    
    Args:
        evaluation_results: Results from evaluation pipeline
        
    Returns:
        Dictionary containing performance analysis
    """
    print(f"\nAnalyzing ISAC performance...")
    
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
    
    # Save returns
    jnp.save("returns.npy", mean_episode_returns)
    
    return {
        "mean_episode_returns": mean_episode_returns,
        "first_episode_returns": first_episode_returns,
        "performance_stats": performance_stats
    }


# ============================================================================
# MAIN ISAC RUNNER FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="isac_mabrax")
def main(config):
    """
    Main function for comprehensive ISAC training and evaluation.
    
    This function orchestrates a complete ISAC experimental pipeline:
    
    1. **Configuration Setup**: Display experiment parameters and ISAC-specific settings
    2. **Training**: Multi-seed ISAC training with actor, Q1, Q2 networks
    3. **Model Saving**: Save training metrics and all network parameters
    4. **Evaluation**: Comprehensive performance evaluation using actor networks
    5. **Analysis**: Compute performance statistics and episode returns
    6. **Visualization**: Generate HTML visualizations of representative episodes
    
    ISAC (Independent Soft Actor-Critic) Key Features:
    - Off-policy learning with experience replay
    - Entropy regularization for automatic exploration
    - Dual Q-networks to reduce overestimation bias
    - Soft target updates for stable learning
    - Separate learning rates for policy and value functions
    
    Args:
        config: Hydra configuration object
    """
    
    # ========================================================================
    # SETUP AND CONFIGURATION
    # ========================================================================
    
    config = OmegaConf.to_container(config, resolve=True)
    print("="*80)
    print("ISAC TRAINING & EVALUATION STARTED")
    print("="*80)
    
    total_start_time = time.time()
    
    # ========================================================================
    # RUN ISAC TRAINING PIPELINE
    # ========================================================================
    
    try:
        training_results = _run_isac_training(config)
        
    except Exception as e:
        print(f"Error during ISAC training: {e}")
        raise
    
    # ========================================================================
    # SAVE TRAINING RESULTS
    # ========================================================================
    
    try:
        _save_training_metrics(training_results)
        model_paths = _save_model_parameters(config, training_results)
        
    except Exception as e:
        print(f"Error saving ISAC training results: {e}")
        raise
    
    # ========================================================================
    # RUN EVALUATION PIPELINE
    # ========================================================================
    
    try:
        evaluation_results = _run_comprehensive_evaluation(config, training_results)
        
    except Exception as e:
        print(f"Error during ISAC evaluation: {e}")
        raise
    
    # ========================================================================
    # ANALYZE PERFORMANCE
    # ========================================================================
    
    try:
        performance_analysis = _analyze_performance(evaluation_results)
        
    except Exception as e:
        print(f"Error during ISAC performance analysis: {e}")
        raise
    

    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"ISAC EXPERIMENT COMPLETED SUCCESSFULLY")
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

    # ========================================================================
    # GENERATE VISUALIZATIONS
    # ========================================================================

    

    if config["VIZ_POLICY"]:
        
        actor_all_path = model_paths["actor_all"]
        current_output_dir = os.getcwd()
        script_directory = os.path.dirname(os.path.abspath(__file__))
        render_script_path = os.path.join(script_directory, "render_isac.py") # Because hydra changes the dir
        
        os.execv(sys.executable, 
                 [sys.executable,
                   render_script_path,
                   f'eval.path={actor_all_path}',
                   f'hydra.run.dir={current_output_dir}',
                   f'NUM_EVAL_EPISODES={config["N_RENDER_EPISODES"]}',
                   ])

if __name__ == "__main__":
    main()
