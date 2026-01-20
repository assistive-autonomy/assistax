"""
Hyperparameter Sweep Visualization Module

This module provides flexible tools for visualizing hyperparameter sweep results
with support for:
- Mean with bootstrapped 95% confidence intervals
- Grouping by one or more hyperparameters (color-coded)
- Flexible algorithm families (IPPO, SAC, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from itertools import product
import os


# =============================================================================
# Data Loading
# =============================================================================

def load_npy_files(directory: str) -> Dict[str, np.ndarray]:
    """
    Load all .npy files from a directory into a dictionary.
    
    Args:
        directory: Path to directory containing .npy files
        
    Returns:
        Dictionary mapping filename (without extension) to numpy array
    """
    npy_dict = {}
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            key = file.replace('.npy', '')
            file_path = os.path.join(directory, file)
            npy_dict[key] = np.load(file_path)
    return npy_dict


# =============================================================================
# Statistical Functions
# =============================================================================

def compute_mean(data: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute mean along specified axis.
    
    Args:
        data: Input array
        axis: Axis along which to compute mean
        
    Returns:
        Mean values
    """
    return np.mean(data, axis=axis)


def bootstrap_ci_mean(
    data: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate bootstrapped confidence intervals for the mean.
    
    Args:
        data: numpy array of shape [n_seeds, n_points]
        n_bootstrap: number of bootstrap samples
        confidence_level: confidence level for the interval (default 0.95 for 95% CI)
        random_seed: optional random seed for reproducibility
        
    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_seeds, n_points = data.shape
    bootstrap_means = np.zeros((n_bootstrap, n_points))
    
    for i in range(n_bootstrap):
        # Resample with replacement along seed axis
        seed_indices = np.random.randint(0, n_seeds, size=n_seeds)
        bootstrap_sample = data[seed_indices]
        bootstrap_means[i] = np.mean(bootstrap_sample, axis=0)
    
    # Compute point estimate
    mean = np.mean(data, axis=0)
    
    # Compute confidence intervals
    alpha = (1 - confidence_level) / 2
    lower_ci = np.percentile(bootstrap_means, 100 * alpha, axis=0)
    upper_ci = np.percentile(bootstrap_means, 100 * (1 - alpha), axis=0)
    
    return mean, lower_ci, upper_ci


def compute_auc(returns: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute Area Under Curve (mean over timesteps).
    
    Args:
        returns: Array of shape [..., n_steps]
        axis: Axis representing timesteps (default -1, the last axis)
        
    Returns:
        AUC values with timestep axis removed
    """
    return np.mean(returns, axis=axis)


# =============================================================================
# Grouping Utilities
# =============================================================================

@dataclass
class GroupInfo:
    """Information about a group of configurations."""
    name: str
    indices: np.ndarray
    color: str
    values: Dict[str, any]  # The hyperparameter values that define this group


def create_groups(
    hp_dict: Dict[str, np.ndarray],
    group_by: Union[str, List[str]]
) -> Tuple[List[GroupInfo], Dict[str, List]]:
    """
    Create groups based on one or more hyperparameters.
    
    Args:
        hp_dict: Dictionary of hyperparameters, each with shape [n_configs]
        group_by: Single hyperparameter name or list of names to group by
        
    Returns:
        Tuple of (list of GroupInfo, dict mapping group_param to unique values)
        
    Example:
        If group_by=['update_epochs', 'num_minibatches'] and we have:
        - update_epochs: [2, 4]
        - num_minibatches: [2, 4]
        
        Groups will be: (2, 2), (2, 4), (4, 2), (4, 4)
    """
    if isinstance(group_by, str):
        group_by = [group_by]
    
    # Get unique values for each grouping parameter
    unique_values = {}
    for param in group_by:
        if param not in hp_dict:
            raise ValueError(f"Grouping parameter '{param}' not found in hp_dict. "
                           f"Available: {list(hp_dict.keys())}")
        unique_values[param] = sorted(list(set(hp_dict[param])))
    
    # Generate all combinations of group values
    all_combinations = list(product(*[unique_values[p] for p in group_by]))
    
    # Create color palette
    n_groups = len(all_combinations)
    colors = sns.color_palette("husl", n_groups)
    
    groups = []
    for combo_idx, combo in enumerate(all_combinations):
        # Find indices matching this combination
        mask = np.ones(len(hp_dict[group_by[0]]), dtype=bool)
        values_dict = {}
        
        for param, value in zip(group_by, combo):
            mask &= (np.array(hp_dict[param]) == value)
            values_dict[param] = value
        
        indices = np.where(mask)[0]
        
        if len(indices) > 0:
            # Create readable name
            if len(group_by) == 1:
                name = f"{group_by[0]}={combo[0]}"
            else:
                name = ", ".join([f"{p}={v}" for p, v in zip(group_by, combo)])
            
            groups.append(GroupInfo(
                name=name,
                indices=indices,
                color=colors[combo_idx],
                values=values_dict
            ))
    
    return groups, unique_values


def format_hparam_value(value: float, param_name: str = "") -> str:
    """
    Format hyperparameter value for display.
    
    Args:
        value: The value to format
        param_name: Name of parameter (used to determine formatting)
        
    Returns:
        Formatted string
    """
    # Learning rate type parameters - use scientific notation
    if 'lr' in param_name.lower() or 'learning' in param_name.lower():
        return f"{value:.2e}"
    # Small floats (like entropy coef, tau)
    elif isinstance(value, float) and abs(value) < 0.1:
        return f"{value:.4f}"
    # Moderate floats
    elif isinstance(value, float):
        return f"{value:.3f}"
    # Integers
    elif isinstance(value, (int, np.integer)):
        return str(int(value))
    else:
        return str(value)


# =============================================================================
# Algorithm-Specific Configurations
# =============================================================================

# Predefined parameter sets for common algorithms
ALGORITHM_CONFIGS = {
    'ippo': {
        'params': ['lr', 'clip_eps', 'ent_coef'],
        'display_names': ['Learning Rate', 'Clip Epsilon', 'Entropy Coef'],
        'log_scale': [True, False, False],
    },
    'ppo': {
        'params': ['lr', 'clip_eps', 'ent_coef'],
        'display_names': ['Learning Rate', 'Clip Epsilon', 'Entropy Coef'],
        'log_scale': [True, False, False],
    },
    'sac': {
        'params': ['p_lr', 'q_lr', 'alpha_lr', 'tau'],
        'display_names': ['Policy LR', 'Q LR', 'Alpha LR', 'Tau'],
        'log_scale': [True, True, True, False],
    },
    'isac': {
        'params': ['p_lr', 'q_lr', 'alpha_lr', 'tau'],
        'display_names': ['Policy LR', 'Q LR', 'Alpha LR', 'Tau'],
        'log_scale': [True, True, True, False],
    },
    'td3': {
        'params': ['actor_lr', 'critic_lr', 'tau'],
        'display_names': ['Actor LR', 'Critic LR', 'Tau'],
        'log_scale': [True, True, False],
    },
    'dqn': {
        'params': ['lr', 'epsilon_decay', 'target_update_freq'],
        'display_names': ['Learning Rate', 'Epsilon Decay', 'Target Update Freq'],
        'log_scale': [True, False, False],
    },
}


def get_algorithm_config(
    algorithm: str,
    custom_params: Optional[List[str]] = None,
    custom_display_names: Optional[List[str]] = None,
    custom_log_scale: Optional[List[bool]] = None
) -> Dict:
    """
    Get or create algorithm configuration.
    
    Args:
        algorithm: Algorithm name (e.g., 'ippo', 'sac')
        custom_params: Override default parameters
        custom_display_names: Override display names
        custom_log_scale: Override log scale settings
        
    Returns:
        Configuration dictionary
    """
    # Start with predefined config if available
    if algorithm.lower() in ALGORITHM_CONFIGS:
        config = ALGORITHM_CONFIGS[algorithm.lower()].copy()
    else:
        config = {'params': [], 'display_names': [], 'log_scale': []}
    
    # Apply overrides
    if custom_params is not None:
        config['params'] = custom_params
        # Auto-generate display names if not provided
        if custom_display_names is None:
            config['display_names'] = [p.replace('_', ' ').title() for p in custom_params]
        else:
            config['display_names'] = custom_display_names
        # Auto-determine log scale if not provided
        if custom_log_scale is None:
            config['log_scale'] = ['lr' in p.lower() for p in custom_params]
        else:
            config['log_scale'] = custom_log_scale
    elif custom_display_names is not None:
        config['display_names'] = custom_display_names
    
    if custom_log_scale is not None:
        config['log_scale'] = custom_log_scale
    
    return config


# =============================================================================
# Main Plotting Functions
# =============================================================================

def plot_training_curves(
    returns: np.ndarray,
    checkpoint_steps: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    title: str = "Training Returns",
    xlabel: str = "Steps",
    ylabel: str = "Mean Episode Return",
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    figsize: Tuple[int, int] = (12, 8),
    random_seed: Optional[int] = None
) -> plt.Figure:
    """
    Plot training returns for all hyperparameter settings with mean and 95% CI.
    
    Args:
        returns: numpy array of shape [n_hyperparams, n_seeds, n_steps]
        checkpoint_steps: array of x-axis values (optional)
        labels: list of labels for each hyperparameter setting
        title, xlabel, ylabel: plot labels
        n_bootstrap: number of bootstrap samples for CI
        confidence_level: confidence level for CI
        figsize: figure size
        random_seed: random seed for reproducibility
        
    Returns:
        matplotlib Figure
    """
    if checkpoint_steps is None:
        checkpoint_steps = np.arange(returns.shape[2])
    
    n_hyperparams = returns.shape[0]
    
    if labels is None:
        labels = [f"Config {i+1}" for i in range(n_hyperparams)]
    
    # Set up the plot style
    fig, ax = plt.subplots(figsize=figsize)
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", n_hyperparams)
    
    # Plot each hyperparameter setting
    for param_idx in range(n_hyperparams):
        data = returns[param_idx]  # Shape: [n_seeds, n_steps]
        
        # Calculate mean and confidence intervals
        mean, lower_ci, upper_ci = bootstrap_ci_mean(
            data, 
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed
        )
        
        # Plot mean and confidence interval
        ax.plot(checkpoint_steps, mean, label=labels[param_idx], color=colors[param_idx])
        ax.fill_between(checkpoint_steps, lower_ci, upper_ci, alpha=0.2, color=colors[param_idx])
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    return fig


def plot_auc_scatter(
    returns: np.ndarray,
    hp_dict: Dict[str, np.ndarray],
    params_to_plot: List[str],
    group_by: Optional[Union[str, List[str]]] = None,
    display_names: Optional[List[str]] = None,
    log_scale: Optional[List[bool]] = None,
    title: str = "AUC vs Hyperparameters",
    figsize_per_subplot: Tuple[float, float] = (5, 4),
    alpha: float = 0.6,
    marker_size: int = 50
) -> plt.Figure:
    """
    Create side-by-side scatter plots of AUC vs hyperparameters with optional grouping.
    
    Args:
        returns: numpy array of shape [n_configs, n_seeds, n_steps]
        hp_dict: dictionary containing hyperparameters
        params_to_plot: list of hyperparameter names to create subplots for
        group_by: hyperparameter(s) to group by for coloring (optional)
        display_names: human-readable names for params_to_plot
        log_scale: whether to use log scale for x-axis (list matching params_to_plot)
        title: overall figure title
        figsize_per_subplot: size of each subplot
        alpha: transparency of scatter points
        marker_size: size of scatter points
        
    Returns:
        matplotlib Figure
        
    Example:
        # IPPO style
        plot_auc_scatter(returns, hp_dict, 
                        params_to_plot=['lr', 'clip_eps', 'ent_coef'],
                        group_by=['update_epochs', 'num_minibatches'])
        
        # SAC style
        plot_auc_scatter(returns, hp_dict,
                        params_to_plot=['p_lr', 'q_lr', 'tau'],
                        group_by='buffer_size')
    """
    # Validate parameters exist
    for param in params_to_plot:
        if param not in hp_dict:
            raise ValueError(f"Parameter '{param}' not found in hp_dict. "
                           f"Available: {list(hp_dict.keys())}")
    
    # Set defaults
    n_params = len(params_to_plot)
    if display_names is None:
        display_names = [p.replace('_', ' ').title() for p in params_to_plot]
    if log_scale is None:
        log_scale = ['lr' in p.lower() for p in params_to_plot]
    
    # Compute AUC
    auc_returns = compute_auc(returns, axis=2)  # Shape: [n_configs, n_seeds]
    
    # Create groups if grouping is specified
    if group_by is not None:
        groups, unique_vals = create_groups(hp_dict, group_by)
    else:
        # Single group containing all configs
        groups = [GroupInfo(
            name='All',
            indices=np.arange(len(hp_dict[params_to_plot[0]])),
            color='blue',
            values={}
        )]
    
    # Create figure
    fig, axes = plt.subplots(1, n_params, 
                             figsize=(figsize_per_subplot[0] * n_params, figsize_per_subplot[1]))
    
    if n_params == 1:
        axes = [axes]
    
    # Plot each parameter
    for ax, param, display_name, use_log in zip(axes, params_to_plot, display_names, log_scale):
        param_values = np.array(hp_dict[param])
        
        for group in groups:
            group_indices = group.indices
            
            for idx in group_indices:
                x_val = param_values[idx]
                if use_log:
                    x_val = np.log10(x_val)
                
                # Plot all seeds for this config
                y_vals = auc_returns[idx]  # Shape: [n_seeds]
                ax.scatter(
                    [x_val] * len(y_vals),
                    y_vals,
                    alpha=alpha,
                    color=group.color,
                    s=marker_size,
                    label=group.name if idx == group_indices[0] else None
                )
        
        ax.set_xlabel(f"{'log10(' + display_name + ')' if use_log else display_name}")
        ax.set_ylabel('AUC')
        ax.set_title(display_name)
        ax.grid(True, alpha=0.3)
    
    # Add legend to the last subplot
    if group_by is not None:
        handles, labels_legend = axes[-1].get_legend_handles_labels()
        # Remove duplicates
        by_label = dict(zip(labels_legend, handles))
        fig.legend(by_label.values(), by_label.keys(), 
                  loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_auc_scatter_algorithm(
    returns: np.ndarray,
    hp_dict: Dict[str, np.ndarray],
    algorithm: str,
    group_by: Optional[Union[str, List[str]]] = None,
    custom_params: Optional[List[str]] = None,
    custom_display_names: Optional[List[str]] = None,
    custom_log_scale: Optional[List[bool]] = None,
    title: Optional[str] = None,
    **kwargs
) -> plt.Figure:
    """
    Convenience function for algorithm-specific AUC scatter plots.
    
    Args:
        returns: numpy array of shape [n_configs, n_seeds, n_steps]
        hp_dict: dictionary containing hyperparameters
        algorithm: algorithm name ('ippo', 'sac', 'td3', etc.)
        group_by: hyperparameter(s) to group by for coloring
        custom_params: override default parameters
        custom_display_names: override display names
        custom_log_scale: override log scale settings
        title: figure title (auto-generated if None)
        **kwargs: additional arguments passed to plot_auc_scatter
        
    Returns:
        matplotlib Figure
    """
    config = get_algorithm_config(
        algorithm,
        custom_params=custom_params,
        custom_display_names=custom_display_names,
        custom_log_scale=custom_log_scale
    )
    
    # Filter to only params that exist in hp_dict
    valid_params = []
    valid_display_names = []
    valid_log_scale = []
    
    for param, display, log in zip(config['params'], config['display_names'], config['log_scale']):
        if param in hp_dict:
            valid_params.append(param)
            valid_display_names.append(display)
            valid_log_scale.append(log)
        else:
            print(f"Warning: Parameter '{param}' not found in hp_dict, skipping.")
    
    if not valid_params:
        raise ValueError(f"No valid parameters found for algorithm '{algorithm}'. "
                        f"Available in hp_dict: {list(hp_dict.keys())}")
    
    if title is None:
        title = f"{algorithm.upper()} - AUC vs Hyperparameters"
    
    return plot_auc_scatter(
        returns,
        hp_dict,
        params_to_plot=valid_params,
        group_by=group_by,
        display_names=valid_display_names,
        log_scale=valid_log_scale,
        title=title,
        **kwargs
    )


def plot_final_return_scatter(
    returns: np.ndarray,
    hp_dict: Dict[str, np.ndarray],
    params_to_plot: List[str],
    group_by: Optional[Union[str, List[str]]] = None,
    window_size: int = 10,
    display_names: Optional[List[str]] = None,
    log_scale: Optional[List[bool]] = None,
    title: str = "Final Returns vs Hyperparameters",
    **kwargs
) -> plt.Figure:
    """
    Create scatter plots of final returns (averaged over last window_size steps).
    
    Args:
        returns: numpy array of shape [n_configs, n_seeds, n_steps]
        hp_dict: dictionary containing hyperparameters
        params_to_plot: list of hyperparameter names to create subplots for
        group_by: hyperparameter(s) to group by for coloring
        window_size: number of final steps to average over
        display_names: human-readable names for params_to_plot
        log_scale: whether to use log scale for x-axis
        title: overall figure title
        **kwargs: additional arguments passed to scatter plot
        
    Returns:
        matplotlib Figure
    """
    # Compute final returns as mean over last window_size steps
    final_returns = returns[:, :, -window_size:].mean(axis=2)  # Shape: [n_configs, n_seeds]
    
    # Create fake "returns" with shape [n_configs, n_seeds, 1] for compatibility
    pseudo_returns = final_returns[:, :, np.newaxis]
    
    return plot_auc_scatter(
        pseudo_returns,
        hp_dict,
        params_to_plot=params_to_plot,
        group_by=group_by,
        display_names=display_names,
        log_scale=log_scale,
        title=title,
        **kwargs
    )


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_performance(
    returns: np.ndarray,
    hp_dict: Dict[str, np.ndarray],
    window_size: int = 10
) -> Tuple[Dict, np.ndarray]:
    """
    Analyze performance metrics for each hyperparameter configuration.
    
    Args:
        returns: numpy array of shape [n_configs, n_seeds, n_steps]
        hp_dict: dictionary containing hyperparameters
        window_size: number of last steps to consider for final performance
        
    Returns:
        performance_metrics: dictionary containing performance metrics for each config
        final_performances: array of final performances (mean over seeds and last steps)
    """
    n_configs, n_seeds, n_steps = returns.shape
    
    final_performances = np.zeros(n_configs)
    performance_metrics = {}
    
    for config_idx in range(n_configs):
        data = returns[config_idx]  # Shape: [n_seeds, n_steps]
        
        # Calculate mean performance
        mean_curve = np.mean(data, axis=0)
        
        # Final performance: mean over last window_size steps
        final_performance = np.mean(mean_curve[-window_size:])
        final_performances[config_idx] = final_performance
        
        # AUC
        auc = np.mean(mean_curve)
        
        # Store metrics
        performance_metrics[config_idx] = {
            'config_idx': config_idx,
            'hyperparams': {k: v[config_idx] for k, v in hp_dict.items()},
            'final_performance': final_performance,
            'auc': auc,
            'max_performance': np.max(mean_curve),
            'mean_performance': auc,
            'std_across_seeds': np.std(np.mean(data, axis=1))
        }
    
    return performance_metrics, final_performances


def print_best_configs(
    performance_metrics: Dict,
    final_performances: np.ndarray,
    n_best: int = 5,
    metric: str = 'final_performance'
) -> None:
    """
    Print the best performing hyperparameter configurations.
    
    Args:
        performance_metrics: dictionary containing performance metrics
        final_performances: array of final performances
        n_best: number of best configurations to show
        metric: which metric to rank by ('final_performance', 'auc', 'max_performance')
    """
    if metric == 'final_performance':
        values = final_performances
    else:
        values = np.array([performance_metrics[i][metric] for i in range(len(performance_metrics))])
    
    best_indices = np.argsort(values)[-n_best:][::-1]
    
    print(f"\nTop {n_best} Configurations (ranked by {metric}):")
    print("=" * 80)
    
    for rank, idx in enumerate(best_indices, 1):
        metrics = performance_metrics[idx]
        print(f"\n{'='*40}")
        print(f"Rank {rank} (Config {metrics['config_idx'] + 1})")
        print(f"{'='*40}")
        print(f"Final Performance: {metrics['final_performance']:.2f}")
        print(f"AUC: {metrics['auc']:.2f}")
        print(f"Max Performance: {metrics['max_performance']:.2f}")
        print(f"Std Across Seeds: {metrics['std_across_seeds']:.2f}")
        print("\nHyperparameters:")
        for param, value in metrics['hyperparams'].items():
            print(f"  {param}: {format_hparam_value(value, param)}")


def create_labels_from_hp_dict(
    hp_dict: Dict[str, np.ndarray],
    params_to_include: Optional[List[str]] = None
) -> List[str]:
    """
    Create readable labels from a hyperparameter dictionary.
    
    Args:
        hp_dict: dictionary containing hyperparameters
        params_to_include: which parameters to include in labels (None = all)
        
    Returns:
        list of labels for each configuration
    """
    if params_to_include is None:
        params_to_include = list(hp_dict.keys())
    
    n_configs = len(hp_dict[params_to_include[0]])
    labels = []
    
    for config_idx in range(n_configs):
        parts = []
        for param in params_to_include:
            if param in hp_dict:
                value = hp_dict[param][config_idx]
                parts.append(f"{param}={format_hparam_value(value, param)}")
        labels.append(", ".join(parts))
    
    return labels


# =============================================================================
# Example Usage (when run as script)
# =============================================================================

if __name__ == "__main__":
    # Create synthetic data for demonstration
    np.random.seed(42)
    
    n_configs = 20
    n_seeds = 6
    n_steps = 100
    
    # Generate synthetic returns
    returns = np.random.randn(n_configs, n_seeds, n_steps).cumsum(axis=2)
    returns = returns + np.linspace(0, 100, n_steps)  # Add trend
    
    # Generate synthetic hyperparameters
    hp_dict = {
        'lr': np.random.choice([1e-4, 2.5e-4, 5e-4, 1e-3], n_configs),
        'clip_eps': np.random.choice([0.1, 0.2, 0.3], n_configs),
        'ent_coef': np.random.choice([0.001, 0.01, 0.1], n_configs),
        'update_epochs': np.random.choice([2, 4, 8], n_configs),
        'num_minibatches': np.random.choice([2, 4], n_configs),
    }
    
    print("=" * 60)
    print("Hyperparameter Visualization Demo")
    print("=" * 60)
    
    # Example 1: IPPO-style plot with grouping
    print("\nExample 1: IPPO AUC plot grouped by update_epochs")
    fig1 = plot_auc_scatter_algorithm(
        returns, hp_dict, 
        algorithm='ippo',
        group_by='update_epochs'
    )
    plt.savefig('/tmp/demo_ippo_single_group.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/demo_ippo_single_group.png")
    
    # Example 2: Grouping by multiple parameters
    print("\nExample 2: IPPO AUC plot grouped by (update_epochs, num_minibatches)")
    fig2 = plot_auc_scatter_algorithm(
        returns, hp_dict,
        algorithm='ippo',
        group_by=['update_epochs', 'num_minibatches']
    )
    plt.savefig('/tmp/demo_ippo_multi_group.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/demo_ippo_multi_group.png")
    
    # Example 3: Custom parameters
    print("\nExample 3: Custom parameters plot")
    fig3 = plot_auc_scatter(
        returns, hp_dict,
        params_to_plot=['lr', 'ent_coef'],
        group_by='clip_eps',
        display_names=['Learning Rate', 'Entropy Coefficient'],
        title="Custom Parameter Plot"
    )
    plt.savefig('/tmp/demo_custom.png', dpi=150, bbox_inches='tight')
    print("Saved to /tmp/demo_custom.png")
    
    # Example 4: Print best configurations
    print("\nExample 4: Best configurations analysis")
    metrics, final_perfs = analyze_performance(returns, hp_dict)
    print_best_configs(metrics, final_perfs, n_best=3)
    
    print("\n" + "=" * 60)
    print("Demo complete!")