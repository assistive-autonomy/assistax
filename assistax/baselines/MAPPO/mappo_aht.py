"""
MAPPO Ad Hoc Teamwork (AHT) and Zero-Shot Coordination Experiments

This script implements a comprehensive experimental pipeline for studying ad hoc teamwork and 
zero-shot coordination in multi-agent reinforcement learning:
"""

import os
import time
from typing import Dict, List, Tuple, Any, Optional

import jax
import jax.numpy as jnp
import pandas as pd
import hydra
from omegaconf import OmegaConf
from tqdm import tqdm
from flax.traverse_util import flatten_dict
import safetensors.flax

import assistax
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper

from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns,
    _tree_shape, _stack_tree, _concat_tree, _tree_split
    )

# ============================================================================
# ARCHITECTURE SELECTION
# ============================================================================

def _import_mappo_variant(config: Dict):
    """
    Import the appropriate MAPPO variant based on configuration.
    
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
# ZOO MANAGEMENT AND PARTNER SELECTION
# ============================================================================

def _load_and_split_zoo_partners(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, ZooManager]:
    """
    Load zoo partners and split them into training and testing sets.
    
    This is a key component of the AHT experimental design - we train against
    one set of partners and test generalization on a completely different set.
    
    Args:
        config: Configuration dictionary containing zoo path and filtering criteria
        
    Returns:
        Tuple of (train_set, test_set, zoo_manager)
        
    Raises:
        ValueError: If no partners are found matching the criteria
    """
    print("Loading partner agents from zoo...")
    
    # Initialize zoo manager
    zoo = ZooManager(config["ZOO_PATH"])
    print(f"Zoo loaded from: {config['ZOO_PATH']}")
    print(f"Total agents in zoo: {len(zoo.index)}")
    
    # Filter agents based on algorithm, scenario, and target agent ID
    algorithm = config["ALGORITHM"]
    scenario = config["ENV_NAME"]
    target_agent = config.get("TARGET_AGENT", "human")  # Agent we're training to coordinate with
    
    print(f"Filtering for:")
    print(f"  Algorithm: {algorithm}")
    print(f"  Scenario: {scenario}")
    print(f"  Target agent: {target_agent}")
    
    # Apply filters to zoo index
    filtered_index = (zoo.index
                      .query(f'algorithm == "{algorithm}"')
                      .query(f'scenario == "{scenario}"')
                      .query(f'scenario_agent_id == "{target_agent}"'))
    
    if len(filtered_index) == 0:
        raise ValueError(
            f"No agents found in zoo matching criteria: "
            f"algorithm={algorithm}, scenario={scenario}, agent={target_agent}"
        )
    
    print(f"Found {len(filtered_index)} matching partner agents")
    
    # Split into training and testing sets
    train_fraction = config.get("TRAIN_FRACTION", 0.5)
    train_set = filtered_index.sample(frac=train_fraction, random_state=config["SEED"])
    test_set = filtered_index.drop(train_set.index)
    
    print(f"Partner split:")
    print(f"  Training partners: {len(train_set)}")
    print(f"  Testing partners: {len(test_set)}")
    print(f"  Train fraction: {train_fraction}")
    
    # Display sample of partners for transparency
    if len(train_set) > 0:
        print(f"\nSubset of training partners:")
        for i, (_, partner) in enumerate(train_set.head(3).iterrows()):
            print(f"  {i+1}. {partner['agent_uuid']} (seed: {partner.get('seed', 'unknown')})")
    
    if len(test_set) > 0:
        print(f"\nSubset of testing partners:")
        for i, (_, partner) in enumerate(test_set.head(3).iterrows()):
            print(f"  {i+1}. {partner['agent_uuid']} (seed: {partner.get('seed', 'unknown')})")
    
    return train_set, test_set, zoo


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def _run_aht_training(config: Dict, train_partners: List[str]) -> Dict:
    """
    Run Ad Hoc Teamwork training against the training set of partners.
    
    Args:
        config: Configuration dictionary
        train_partners: List of partner agent UUIDs to train against
        
    Returns:
        Dictionary containing training results
    """
    print(f"\n{'='*60}")
    print("STARTING ZSC TRAINING")
    print(f"{'='*60}")
    
    # Import architecture-specific functions
    make_train, _, _ = _import_mappo_variant(config)
    
    # Setup training configuration
    target_agent = config.get("TARGET_AGENT", "human")
    architecture = f"{'RNN' if config['network']['recurrent'] else 'FF'}"
    sharing = f"{'PS' if config['network']['agent_param_sharing'] else 'NPS'}"
        
    # Setup random number generation
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, _ = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])
    
    # Run training with zoo partners
    print(f"\nStarting AHT training against {len(train_partners)} partners...")
    start_time = time.time()
    
    with jax.disable_jit(config["DISABLE_JIT"]):
        # Create training function with zoo loading
        zoo_config = {target_agent: train_partners}
        train_jit = jax.jit(
            make_train(config, save_train_state=True, load_zoo=zoo_config),
            device=jax.devices()[config["DEVICE"]]
        )
        
        # Run training across multiple seeds
        training_results = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], 
            config["ENT_COEF"], 
            config["CLIP_EPS"]
        )
    
    training_time = time.time() - start_time
    print(f"AHT training completed in {training_time:.2f} seconds")
    
    return {
        "training_results": training_results,
        "training_time": training_time,
        "train_partners": train_partners
    }


# ============================================================================
# EVALUATION PIPELINE
# ============================================================================

def _run_aht_evaluation(config: Dict, training_results: Dict, 
                       train_partners: List[str], test_partners: List[str]) -> Dict:
    """
    Run comprehensive evaluation against both seen and unseen partners.
    
    This is the core of the zero-shot coordination experiment - we test how well
    the agent performs with partners it has never encountered during training.
    
    Args:
        config: Configuration dictionary
        training_results: Results from AHT training
        train_partners: Partners used during training (seen)
        test_partners: Partners not used during training (unseen)
        
    Returns:
        Dictionary containing evaluation results for both train and test sets
    """
    print(f"\n{'='*60}")
    print("STARTING ZERO-SHOT COORDINATION EVALUATION")
    print(f"{'='*60}")
    
    # Import evaluation functions
    _, make_evaluation, EvalInfoLogConfig = _import_mappo_variant(config)
    
    # Extract training states for evaluation
    all_train_states = training_results["training_results"]["metrics"]["train_state"]
    target_agent = config.get("TARGET_AGENT", "human")
    
    # Calculate evaluation batching for memory efficiency
    batch_dims = jax.tree.leaves(_tree_shape(all_train_states.actor.params))[:2]
    n_sequential_evals = int(jnp.ceil(
        config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
        / config["GPU_ENV_CAPACITY"]
    ))
    
    def _flatten_and_split_trainstate(trainstate):
        """Flatten and split training states for batched evaluation."""
        flat_trainstate = jax.tree.map(
            lambda x: x.reshape((x.shape[0]*x.shape[1], *x.shape[2:])),
            trainstate
        )
        return _tree_split(flat_trainstate, n_sequential_evals)
    
    # JIT compile for memory efficiency
    split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)
    
    # Setup evaluation environments
    print(f"Setting up evaluation environments...")
    print(f"  Training partners (seen): {len(train_partners)}")
    print(f"  Testing partners (unseen): {len(test_partners)}")
    
    # Create evaluation environments for both sets
    eval_train_env, run_eval_train = make_evaluation(
        config, load_zoo={target_agent: train_partners}
    )
    eval_test_env, run_eval_test = make_evaluation(
        config, load_zoo={target_agent: test_partners}
    )
    
    # Configure what to log during evaluation
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
    
    # JIT compile evaluation functions
    eval_train_jit = jax.jit(run_eval_train, static_argnames=["log_eval_info"])
    eval_train_vmap = jax.vmap(eval_train_jit, in_axes=(None, 0, None))
    eval_test_jit = jax.jit(run_eval_test, static_argnames=["log_eval_info"])
    eval_test_vmap = jax.vmap(eval_test_jit, in_axes=(None, 0, None))
    
    # Run evaluation against training partners (seen during training)
    print(f"\nEvaluating against TRAINING partners (seen)...")
    eval_rng = jax.random.PRNGKey(config["SEED"] + 1000)
    
    evals_train = _concat_tree([
        eval_train_vmap(eval_rng, ts, eval_log_config)
        for ts in tqdm(split_trainstate, desc="Train eval batches")
    ])
    evals_train = jax.tree.map(
        lambda x: x.reshape((*batch_dims, *x.shape[1:])),
        evals_train
    )
    
    # Run evaluation against testing partners (unseen during training)
    print(f"\nEvaluating against TESTING partners (unseen - zero-shot)...")
    
    evals_test = _concat_tree([
        eval_test_vmap(eval_rng, ts, eval_log_config)
        for ts in tqdm(split_trainstate, desc="Test eval batches")
    ])
    evals_test = jax.tree.map(
        lambda x: x.reshape((*batch_dims, *x.shape[1:])),
        evals_test
    )
    
    # Compute episode returns for both evaluations
    train_returns = _compute_episode_returns(evals_train)["__all__"]
    test_returns = _compute_episode_returns(evals_test)["__all__"]
    
    # Calculate mean returns across episodes
    train_mean_returns = train_returns.mean(axis=-1)
    test_mean_returns = test_returns.mean(axis=-1)
    
    return {
        "train_returns": train_mean_returns,
        "test_returns": test_mean_returns,
        "train_raw_returns": train_returns,
        "test_raw_returns": test_returns,
        "evals_train": evals_train,
        "evals_test": evals_test
    }


# ============================================================================
# RESULT SAVING AND ANALYSIS
# ============================================================================

def _save_training_results(config: Dict, training_results: Dict):
    """Save training metrics and model parameters."""

    # Save training metrics (excluding large objects)
    EXCLUDED_METRICS = ["train_state"]
    metrics = training_results["training_results"]["metrics"]
    filtered_metrics = {
        key: val for key, val in metrics.items() 
        if key not in EXCLUDED_METRICS
    }
    jnp.save("metrics.npy", filtered_metrics, allow_pickle=True)
    
    # Save model parameters
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    all_train_states = metrics["train_state"]
    final_train_state = training_results["training_results"]["runner_state"].train_state
    
    # Save all training states
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
        split_params = _unstack_tree(
            jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state.actor.params)
        )
        for agent, params in zip(env.agents, split_params):
            safetensors.flax.save_file(
                flatten_dict(params, sep='/'),
                f"{agent}.safetensors",
            )
    
def _save_evaluation_results(evaluation_results: Dict):
    """Save evaluation results for both train and test sets."""

    # Save return data
    jnp.save("train_returns.npy", evaluation_results["train_returns"])
    jnp.save("test_returns.npy", evaluation_results["test_returns"])



def _analyze_aht_performance(evaluation_results: Dict, train_partners: List[str], 
                           test_partners: List[str]) -> Dict:
    """
    Analyze AHT and zero-shot coordination performance.
    
    Args:
        evaluation_results: Results from evaluation pipeline
        train_partners: List of training partner UUIDs
        test_partners: List of testing partner UUIDs
        
    Returns:
        Dictionary containing analysis results
    """
    train_returns = evaluation_results["train_returns"]
    test_returns = evaluation_results["test_returns"]
    
    # Calculate performance statistics
    train_performance = {
        "mean": float(jnp.mean(train_returns)),
        "std": float(jnp.std(train_returns)),
        "min": float(jnp.min(train_returns)),
        "max": float(jnp.max(train_returns)),
        "median": float(jnp.median(train_returns))
    }
    
    test_performance = {
        "mean": float(jnp.mean(test_returns)),
        "std": float(jnp.std(test_returns)),
        "min": float(jnp.min(test_returns)),
        "max": float(jnp.max(test_returns)),
        "median": float(jnp.median(test_returns))
    }
    
    # Calculate generalization metrics
    generalization_gap = train_performance["mean"] - test_performance["mean"]
    relative_performance = test_performance["mean"] / train_performance["mean"] if train_performance["mean"] != 0 else 0
    
    analysis = {
        "train_performance": train_performance,
        "test_performance": test_performance,
        "generalization_gap": float(generalization_gap),
        "relative_performance": float(relative_performance),
        "num_train_partners": len(train_partners),
        "num_test_partners": len(test_partners)
    }
    
    return analysis


# ============================================================================
# MAIN AHT EXPERIMENT FUNCTION
# ============================================================================

@hydra.main(version_base=None, config_path="config", config_name="mappo_aht")
def main(config):
    """
    Main function for MAPPO Ad Hoc Teamwork experiments.
    
    This function orchestrates a complete AHT experimental pipeline:
    
    1. **Partner Loading**: Load diverse partners from zoo and split into train/test
    2. **AHT Training**: Train agent against training partners only
    3. **Dual Evaluation**: Test against both seen (train) and unseen (test) partners
    4. **Analysis**: Compute generalization metrics and zero-shot coordination performance
    5. **Result Saving**: Save all results for downstream analysis
    
    Args:
        config: Hydra configuration object
    """
    
    # ========================================================================
    # SETUP AND CONFIGURATION
    # ========================================================================
    
    config = OmegaConf.to_container(config, resolve=True)
    
    print("="*80)
    print("MAPPO ZERO-SHOT COORDINATION EXPERIMENT")
    print("="*80)
   
    total_start_time = time.time()
    
    # ========================================================================
    # LOAD AND SPLIT ZOO PARTNERS
    # ========================================================================
    
    try:
        train_set, test_set, zoo = _load_and_split_zoo_partners(config)
        train_partners = list(train_set.agent_uuid)
        test_partners = list(test_set.agent_uuid)
        
        if len(train_partners) == 0:
            raise ValueError("No training partners available")
        if len(test_partners) == 0:
            raise ValueError("No testing partners available")
            
    except Exception as e:
        print(f"Error loading zoo partners: {e}")
        raise
    
    # ========================================================================
    # RUN AD HOC TEAMWORK TRAINING
    # ========================================================================
    
    try:
        training_results = _run_aht_training(config, train_partners)
        
    except Exception as e:
        print(f"Error during AHT training: {e}")
        raise
    
    # ========================================================================
    # RUN ZERO-SHOT COORDINATION EVALUATION
    # ========================================================================
    
    try:
        evaluation_results = _run_aht_evaluation(
            config, training_results, train_partners, test_partners
        )
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    try:
        _save_training_results(config, training_results)
        _save_evaluation_results(evaluation_results)
        
    except Exception as e:
        print(f"Error saving results: {e}")
        raise

    print(f"\n{'='*80}")
    print(f"ZSC Experiment Completed successfully!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

