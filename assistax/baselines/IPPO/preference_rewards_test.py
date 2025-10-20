import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from typing import Dict, Any
import assistax
from ippo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
import hydra
import time
import os


def test_ippo_preference_wrapper(
    config_dict: Dict[str, Any],
    quick_train_steps: int = 50,
    eval_episodes: int = 10,
    plot_results: bool = True,
    verbose: bool = True,
    save_plots: bool = False,
    save_dir: str = "."
):
    """
    Test IPPO training with preference rewards to verify wrapper functionality.
    
    Args:
        config_dict: Your Hydra config as a dictionary
        quick_train_steps: Number of training steps to run (keep small for testing)
        eval_episodes: Number of evaluation episodes
        plot_results: Whether to create plots
        verbose: Whether to print detailed output
        save_plots: Whether to save plots to disk
        save_dir: Directory to save plots and results
        
    Returns:
        Dict with training metrics and preference reward analysis
    """
    
    if verbose:
        print("🧪 Testing IPPO with Preference Wrapper")
        print(f"Environment: {config_dict['ENV_NAME']}")
        print(f"Training steps: {quick_train_steps}")
        print(f"Evaluation episodes: {eval_episodes}")
        print("-" * 60)
    
    # Modify config for quick testing
    test_config = config_dict.copy()
    test_config.update({
        "NUM_UPDATES": quick_train_steps,
        "NUM_EVAL_EPISODES": eval_episodes,
        "NUM_ENVS": 4,  # Small for testing
        "NUM_STEPS": 32,  # Short rollouts
        "DISABLE_JIT": False,  # Enable JIT for speed
    })
    
    # ===== SETUP =====
    rng = jax.random.PRNGKey(42)
    train_rng, eval_rng = jax.random.split(rng)
    
    # Create training function
    train_fn = make_train(test_config, save_train_state=True)
    train_jit = jax.jit(train_fn)
    
    if verbose:
        print("🚀 Starting training...")
    
    # ===== TRAINING =====
    start_time = time.time()
    training_output = train_jit(
        train_rng,
        test_config["LR"], 
        test_config["ENT_COEF"], 
        test_config["CLIP_EPS"]
    )
    train_time = time.time() - start_time
    
    if verbose:
        print(f"✅ Training completed in {train_time:.2f}s")
    
    # ===== EXTRACT TRAINING METRICS =====
    metrics = training_output["metrics"]
    final_train_state = training_output["runner_state"].train_state
    
    # Extract preference-related metrics
    preference_metrics = {}
    for key in metrics.keys():
        if "pref_" in key or "preference" in key or "touch" in key:
            preference_metrics[key] = np.array(metrics[key])
    
    if verbose:
        print("\n📊 Training Summary:")
        print(f"Total updates: {len(metrics['update_step'])}")
        
        if preference_metrics:
            print("✅ Preference metrics found during training:")
            for key, values in preference_metrics.items():
                print(f"  - {key}: {values[-1]:.4f} (final), {values.mean():.4f} (avg)")
        else:
            print("❌ No preference metrics found in training!")
    
    # ===== EVALUATION =====
    if verbose:
        print("\n🎯 Running evaluation...")
    
    eval_env, run_eval = make_evaluation(test_config)
    
    eval_log_config = EvalInfoLogConfig(
        env_state=False,
        done=True,
        action=False,
        value=False,
        reward=True,
        log_prob=False,
        obs=False,
        info=True,  # Need this for preference metrics
        avail_actions=False,
    )
    
    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
    eval_results = eval_jit(eval_rng, final_train_state, eval_log_config)
    
    # ===== ANALYZE EVALUATION RESULTS =====
    if verbose:
        print("✅ Evaluation completed")
        print("\n📈 Evaluation Analysis:")
    
    # Extract rewards and episode returns
    episode_rewards = eval_results.reward["__all__"]  # Shape: (time_steps, episodes)
    episode_returns = episode_rewards.sum(axis=0)  # Sum over time steps
    
    if verbose:
        print(f"Episode returns: {episode_returns.mean():.3f} ± {episode_returns.std():.3f}")
        print(f"Range: [{episode_returns.min():.3f}, {episode_returns.max():.3f}]")
    
    # Check for preference info in evaluation
    eval_info = eval_results.info
    has_preference_info = False
    preference_eval_data = {}
    
    if eval_info is not None:
        for key in eval_info.keys():
            if "pref_" in key or "preference" in key or "touch" in key:
                has_preference_info = True
                preference_eval_data[key] = np.array(eval_info[key])
                if verbose:
                    values = preference_eval_data[key]
                    print(f"  - {key}: {values.mean():.4f} (avg across episodes)")
    
    if not has_preference_info and verbose:
        print("⚠️ No preference metrics found in evaluation info")
    
    # ===== RESULTS COMPILATION =====
    results = {
        'training_time': train_time,
        'training_metrics': dict(metrics),
        'preference_training_metrics': preference_metrics,
        'episode_returns': np.array(episode_returns),
        'preference_eval_data': preference_eval_data,
        'config': test_config,
        'has_preference_metrics': len(preference_metrics) > 0,
    }
    
    # ===== SAVE RESULTS =====
    if save_dir != "." and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save results to file
    results_file = os.path.join(save_dir, "preference_test_results.npy")
    np.save(results_file, results, allow_pickle=True)
    if verbose:
        print(f"\n💾 Results saved to: {results_file}")
    
    # ===== PLOTTING =====
    if plot_results:
        _plot_ippo_preference_results(results, verbose, save_plots, save_dir)
    
    return results


def _plot_ippo_preference_results(results: Dict[str, Any], verbose: bool = True, 
                                 save_plots: bool = False, save_dir: str = "."):
    """Create plots for IPPO preference wrapper test results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('IPPO Preference Wrapper Test Results', fontsize=16)
    
    # Training metrics over time
    training_metrics = results['training_metrics']
    steps = np.array(training_metrics['update_step'])
    
    # Plot 1: Training rewards and losses
    axes[0, 0].plot(steps, training_metrics.get('total_loss', []), label='Total Loss', alpha=0.8)
    axes[0, 0].plot(steps, training_metrics.get('actor_loss', []), label='Actor Loss', alpha=0.8)
    axes[0, 0].plot(steps, training_metrics.get('critic_loss', []), label='Critic Loss', alpha=0.8)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Update Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Preference rewards during training (if available)
    pref_metrics = results['preference_training_metrics']
    if pref_metrics:
        for key, values in pref_metrics.items():
            if len(values) == len(steps):
                axes[0, 1].plot(steps, values, label=key.replace('pref_', ''), alpha=0.8)
        axes[0, 1].set_title('Preference Rewards (Training)')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No preference metrics\nfound during training', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Preference Rewards (Training)')
    
    # Plot 3: Episode returns distribution
    returns = results['episode_returns']
    axes[1, 0].hist(returns, bins=min(10, len(returns)//2), alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', 
                      label=f'Mean: {returns.mean():.3f}')
    axes[1, 0].set_title('Episode Returns Distribution')
    axes[1, 0].set_xlabel('Episode Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Preference metrics during evaluation (if available)
    eval_pref = results['preference_eval_data']
    if eval_pref:
        metric_names = list(eval_pref.keys())
        values = [eval_pref[key].mean() for key in metric_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
        
        bars = axes[1, 1].bar(range(len(metric_names)), values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Preference Metrics (Evaluation)')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels([name.replace('pref_', '') for name in metric_names], 
                                  rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, 'No preference metrics\nfound during evaluation', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Preference Metrics (Evaluation)')
    
    plt.tight_layout()
    
    if save_plots:
        plot_file = os.path.join(save_dir, "preference_test_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"📊 Plots saved to: {plot_file}")
    
    if not save_plots:
        plt.show()
    else:
        plt.close()
    
    if verbose and not save_plots:
        print("\n📊 Plots generated successfully!")


@hydra.main(version_base=None, config_path="config", config_name="ippo")
def main(config):
    """
    Main function for testing IPPO with preference wrapper using Hydra config.
    
    This uses your regular IPPO config and adds testing-specific parameters.
    
    Example usage:
        python test_ippo_preference.py
        python test_ippo_preference.py train_steps=100 eval_episodes=20
        python test_ippo_preference.py quick=true
        python test_ippo_preference.py save_plots=true save_dir=./test_results
    """
    
    # Convert Hydra config to container
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    # Extract test-specific parameters (with defaults)
    train_steps = config_dict.get('train_steps', 50)
    eval_episodes = config_dict.get('eval_episodes', 10)
    quick_mode = config_dict.get('quick', False)
    save_plots = config_dict.get('save_plots', False)
    save_dir = config_dict.get('save_dir', './preference_test_results')
    no_plots = config_dict.get('no_plots', False)
    quiet = config_dict.get('quiet', False)
    
    print("🧪 IPPO Preference Wrapper Test")
    print(f"Using config: {config_dict['ENV_NAME']}")
    if 'preference_rewards' in config_dict["ENV_KWARGS"]:
        print("✅ Preference rewards config found")
    else:
        print("⚠️ No preference rewards config found - testing without preferences")
    print("-" * 60)
    
    # Quick mode overrides
    if quick_mode:
        train_steps = 10
        eval_episodes = 5
        no_plots = True
        print("🚀 Quick mode: minimal training and evaluation")
    
    # Run the test
    results = test_ippo_preference_wrapper(
        config_dict=config_dict,
        quick_train_steps=train_steps,
        eval_episodes=eval_episodes,
        plot_results=not no_plots,
        verbose=not quiet,
        save_plots=save_plots,
        save_dir=save_dir
    )
    
    # Summary
    print(f"\n🎉 Test completed successfully!")
    print(f"✓ Training time: {results['training_time']:.2f}s")
    print(f"✓ Preference metrics found: {results['has_preference_metrics']}")
    print(f"✓ Average episode return: {results['episode_returns'].mean():.3f}")
    
    if save_plots or save_dir != './preference_test_results':
        print(f"📁 Results saved in: {save_dir}")
    
    return results


if __name__ == "__main__":
    main()


def _plot_ippo_preference_results(results: Dict[str, Any], verbose: bool = True, 
                                 save_plots: bool = False, save_dir: str = "."):
    """Create plots for IPPO preference wrapper test results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('IPPO Preference Wrapper Test Results', fontsize=16)
    
    # Training metrics over time
    training_metrics = results['training_metrics']
    steps = np.array(training_metrics['update_step'])
    
    # Plot 1: Training rewards and losses
    axes[0, 0].plot(steps, training_metrics.get('total_loss', []), label='Total Loss', alpha=0.8)
    axes[0, 0].plot(steps, training_metrics.get('actor_loss', []), label='Actor Loss', alpha=0.8)
    axes[0, 0].plot(steps, training_metrics.get('critic_loss', []), label='Critic Loss', alpha=0.8)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Update Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Preference rewards during training (if available)
    pref_metrics = results['preference_training_metrics']
    if pref_metrics:
        for key, values in pref_metrics.items():
            if len(values) == len(steps):
                axes[0, 1].plot(steps, values, label=key.replace('pref_', ''), alpha=0.8)
        axes[0, 1].set_title('Preference Rewards (Training)')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No preference metrics\nfound during training', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Preference Rewards (Training)')
    
    # Plot 3: Episode returns distribution
    returns = results['episode_returns']
    axes[1, 0].hist(returns, bins=min(10, len(returns)//2), alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', 
                      label=f'Mean: {returns.mean():.3f}')
    axes[1, 0].set_title('Episode Returns Distribution')
    axes[1, 0].set_xlabel('Episode Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Preference metrics during evaluation (if available)
    eval_pref = results['preference_eval_data']
    if eval_pref:
        metric_names = list(eval_pref.keys())
        values = [eval_pref[key].mean() for key in metric_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
        
        bars = axes[1, 1].bar(range(len(metric_names)), values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Preference Metrics (Evaluation)')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels([name.replace('pref_', '') for name in metric_names], 
                                  rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, 'No preference metrics\nfound during evaluation', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Preference Metrics (Evaluation)')
    
    plt.tight_layout()
    
    if save_plots:
        plot_file = os.path.join(save_dir, "preference_test_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        if verbose:
            print(f"📊 Plots saved to: {plot_file}")
    
    if not save_plots:
        plt.show()
    else:
        plt.close()
    
    if verbose and not save_plots:
        print("\n📊 Plots generated successfully!")


def quick_ippo_preference_test(config_dict: Dict[str, Any]):
    """
    Super quick test function - just verify IPPO + preference wrapper works.
    
    Args:
        config_dict: Config dictionary from Hydra
    """
    print("🚀 Quick IPPO Preference Test...")
    
    # Run quick test
    results = test_ippo_preference_wrapper(
        config_dict, 
        quick_train_steps=10,  # Very short for quick test
        eval_episodes=5,
        plot_results=False,
        verbose=True
    )
    
    # Summary
    print("\n✅ Quick test completed!")
    print(f"✓ Training worked: {results['training_time']:.2f}s")
    print(f"✓ Preference metrics found: {results['has_preference_metrics']}")
    print(f"✓ Average episode return: {results['episode_returns'].mean():.3f}")
    
    return results


# Usage examples for Jupyter:
"""
# Example 1: Using this in Jupyter with Hydra config
from omegaconf import OmegaConf

config = OmegaConf.load('path/to/your/config.yaml')
config_dict = OmegaConf.to_container(config, resolve=True)

# Quick test
results = quick_ippo_preference_test(config_dict)

# Full test with plots
results = test_ippo_preference_wrapper(
    config_dict, 
    quick_train_steps=100, 
    eval_episodes=20,
    plot_results=True
)
"""


def _plot_ippo_preference_results(results: Dict[str, Any], verbose: bool = True):
    """Create plots for IPPO preference wrapper test results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('IPPO Preference Wrapper Test Results', fontsize=16)
    
    # Training metrics over time
    training_metrics = results['training_metrics']
    steps = np.array(training_metrics['update_step'])
    
    # Plot 1: Training rewards and losses
    axes[0, 0].plot(steps, training_metrics.get('total_loss', []), label='Total Loss', alpha=0.8)
    axes[0, 0].plot(steps, training_metrics.get('actor_loss', []), label='Actor Loss', alpha=0.8)
    axes[0, 0].plot(steps, training_metrics.get('critic_loss', []), label='Critic Loss', alpha=0.8)
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].set_xlabel('Update Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Preference rewards during training (if available)
    pref_metrics = results['preference_training_metrics']
    if pref_metrics:
        for key, values in pref_metrics.items():
            if len(values) == len(steps):
                axes[0, 1].plot(steps, values, label=key.replace('pref_', ''), alpha=0.8)
        axes[0, 1].set_title('Preference Rewards (Training)')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No preference metrics\nfound during training', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Preference Rewards (Training)')
    
    # Plot 3: Episode returns distribution
    returns = results['episode_returns']
    axes[1, 0].hist(returns, bins=min(10, len(returns)//2), alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', 
                      label=f'Mean: {returns.mean():.3f}')
    axes[1, 0].set_title('Episode Returns Distribution')
    axes[1, 0].set_xlabel('Episode Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Preference metrics during evaluation (if available)
    eval_pref = results['preference_eval_data']
    if eval_pref:
        metric_names = list(eval_pref.keys())
        values = [eval_pref[key].mean() for key in metric_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(metric_names)))
        
        bars = axes[1, 1].bar(range(len(metric_names)), values, color=colors, alpha=0.7)
        axes[1, 1].set_title('Preference Metrics (Evaluation)')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].set_xticks(range(len(metric_names)))
        axes[1, 1].set_xticklabels([name.replace('pref_', '') for name in metric_names], 
                                  rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    else:
        axes[1, 1].text(0.5, 0.5, 'No preference metrics\nfound during evaluation', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Preference Metrics (Evaluation)')
    
    plt.tight_layout()
    plt.show()
    
    if verbose:
        print("\n📊 Plots generated successfully!")


def quick_ippo_preference_test(config_path: str = None, config_dict: Dict[str, Any] = None):
    """
    Super quick test function - just verify IPPO + preference wrapper works.
    
    Args:
        config_path: Path to Hydra config file, OR
        config_dict: Config dictionary directly
    """
    print("🚀 Quick IPPO Preference Test...")
    
    if config_dict is None:
        if config_path is None:
            # Create minimal test config
            config_dict = {
                "ENV_NAME": "scratchitch",
                "ENV_KWARGS": {"ctrl_cost_weight": 0},
                "preference_rewards": {
                    "preference_weights": {
                        "speed_preference": 0.25,
                        "force_preference": 0.35,
                        "action_efficiency": 0.15,
                        "touch_penalty": -0.03,
                    },
                    "preference_ranges": {
                        "speed_range": [0.06, 0.14],
                        "force_range": [1.5, 3.5],
                        "max_action_magnitude": 0.8,
                    },
                    "touch_threshold": 0.3,
                },
                "LR": 2.5e-4,
                "ENT_COEF": 0.01,
                "CLIP_EPS": 0.2,
                "GAMMA": 0.99,
                "GAE_LAMBDA": 0.95,
                "VF_COEF": 0.5,
                "MAX_GRAD_NORM": 0.5,
                "ANNEAL_LR": True,
                "NUM_MINIBATCHES": 4,
                "UPDATE_EPOCHS": 4,
                "network": {"activation": "tanh", "actor_hidden_dim": 64, "critic_hidden_dim": 64},
            }
        else:
            config_dict = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    
    # Run quick test
    results = test_ippo_preference_wrapper(
        config_dict, 
        quick_train_steps=10,  # Very short for quick test
        eval_episodes=5,
        plot_results=False,
        verbose=True
    )
    
    # Summary
    print("\n✅ Quick test completed!")
    print(f"✓ Training worked: {results['training_time']:.2f}s")
    print(f"✓ Preference metrics found: {results['has_preference_metrics']}")
    print(f"✓ Average episode return: {results['episode_returns'].mean():.3f}")
    
    return results

