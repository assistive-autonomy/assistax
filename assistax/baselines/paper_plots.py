"""
Paper Plots Pipeline for Assistax.

Pulls evaluation data from wandb artifacts, processes it, and produces
publication-quality figures for the Assistax research paper.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import wandb

from assistax.baselines.sweep_plots import bootstrap_ci_mean
from assistax.baselines.utils import load_compact_npz


# =============================================================================
# Constants
# =============================================================================

DEFAULT_ENTITY = "lh-from-kb"
DEFAULT_PROJECT = "assistax-dev"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/assistax_paper_plots")
DEFAULT_METRIC = "mean_episode_returns___all__"

ENV_DISPLAY_NAMES: Dict[str, str] = {
    "scratchitch": "Scratch Itch",
    "bedbathing": "Bed Bathing",
    "armmanipulation": "Arm Manipulation",
    "pushcoop": "Push Coop",
    "feeding": "Feeding",
    "handover": "Handover",
    "teethbrushing": "Teeth Brushing",
}

ALGO_COLORS: Dict[str, str] = {
    "IPPO (FF)": "#1f77b4",
    "IPPO (RNN)": "#ff7f0e",
    "IPPO (FF, PS)": "#2ca02c",
    "IPPO (RNN, PS)": "#d62728",
    "MAPPO (FF)": "#9467bd",
    "MAPPO (RNN)": "#8c564b",
    "ISAC": "#e377c2",
    "MASAC": "#7f7f7f",
}

ALGO_MARKERS: Dict[str, str] = {
    "IPPO (FF)": "o",
    "IPPO (RNN)": "s",
    "IPPO (FF, PS)": "^",
    "IPPO (RNN, PS)": "D",
    "MAPPO (FF)": "v",
    "MAPPO (RNN)": "<",
    "ISAC": "P",
    "MASAC": "X",
}

PAPER_COLUMN_WIDTH = 3.33  # inches (single column in two-column paper)
PAPER_FULL_WIDTH = 6.75    # inches (full width in two-column paper)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AlgorithmData:
    """Container for one algorithm's evaluation data in one environment."""

    name: str               # e.g., "IPPO (FF)"
    tags: List[str]         # tags used to fetch from wandb
    returns: np.ndarray     # shape: (num_seeds, num_updates)
    total_timesteps: int    # total environment steps for x-axis scaling


@dataclass
class ExperimentCollection:
    """Collection of evaluation data across environments and algorithms."""

    data: Dict[str, Dict[str, AlgorithmData]] = field(default_factory=dict)

    @property
    def env_names(self) -> List[str]:
        """Return sorted list of environment names."""
        return sorted(self.data.keys())

    @property
    def algo_names(self) -> List[str]:
        """Return sorted list of unique algorithm names across all envs."""
        names = set()
        for env_data in self.data.values():
            names.update(env_data.keys())
        return sorted(names)

    def add(self, env_name: str, algo_data: AlgorithmData) -> None:
        """Add algorithm data for an environment."""
        if env_name not in self.data:
            self.data[env_name] = {}
        self.data[env_name][algo_data.name] = algo_data


# =============================================================================
# Layer 1: Data Extraction
# =============================================================================

def fetch_runs_by_tags(
    tags: List[str],
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    exclude_tags: Optional[List[str]] = None,
) -> list:
    """Fetch wandb runs matching all specified tags.

    Args:
        tags: List of tags that must ALL be present on each run.
        entity: wandb entity (team/user).
        project: wandb project name.
        exclude_tags: Tags that, if present on a run, cause it to be excluded.

    Returns:
        List of matching wandb Run objects.
    """
    import wandb

    api = wandb.Api()
    mongo_filter = {"tags": {"$all": tags}}
    runs = list(api.runs(f"{entity}/{project}", filters=mongo_filter))
    print(f"  Found {len(runs)} runs for tags {tags}")

    if exclude_tags:
        runs = [r for r in runs if not any(t in r.tags for t in exclude_tags)]

    return runs


def download_artifact_data(
    run,
    metric_name: str = DEFAULT_METRIC,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[np.ndarray]:
    """Download evaluation artifact for a run and load the requested metric.

    Caches downloaded artifacts under ``cache_dir/{run.id}/``.

    Args:
        run: A wandb Run object.
        metric_name: Name of the metric file inside the artifact (without extension).
        cache_dir: Local directory for caching downloaded artifacts.

    Returns:
        Numpy array loaded via ``load_compact_npz``, or None on failure.
    """
    run_cache = os.path.join(cache_dir, run.id)
    npz_path = os.path.join(run_cache, f"{metric_name}.npz")

    # Return cached data if available
    if os.path.exists(npz_path):
        return load_compact_npz(npz_path)

    # Download artifact
    artifact_name = f"evaluation_data_{run.name}"
    try:
        api = wandb.Api()
        artifact_path = f"{run.entity}/{run.project}/{artifact_name}:latest"
        artifact = api.artifact(artifact_path)
        artifact_dir = artifact.download(root=run_cache)
    except Exception as e:
        print(f"[WARN] Could not download artifact for run {run.name}: {e}")
        return None

    # Load metric
    npz_path = os.path.join(artifact_dir, f"{metric_name}.npz")
    if not os.path.exists(npz_path):
        available = os.listdir(artifact_dir) if os.path.isdir(artifact_dir) else []
        print(f"[WARN] Metric '{metric_name}' not found in artifact for run {run.name}. Available files: {available}")
        return None

    return load_compact_npz(npz_path)


def extract_experiment_data(
    tags: List[str],
    metric_name: str = DEFAULT_METRIC,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    exclude_tags: Optional[List[str]] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[Tuple[np.ndarray, int]]:
    """Fetch runs by tags, download artifacts, and concatenate along seed axis.

    Args:
        tags: Tags for filtering runs.
        metric_name: Metric to extract from each artifact.
        entity: wandb entity.
        project: wandb project.
        exclude_tags: Tags to exclude.
        cache_dir: Cache directory.

    Returns:
        Tuple of (concatenated array of shape ``(total_seeds, min_num_updates)``,
        total_timesteps from wandb config) or None.
    """
    runs = fetch_runs_by_tags(tags, entity, project, exclude_tags)
    if not runs:
        print(f"[WARN] No runs found for tags {tags}")
        return None

    # Read TOTAL_TIMESTEPS from the first run's wandb config
    total_timesteps = int(runs[0].config.get("TOTAL_TIMESTEPS", 0))

    arrays = []
    for run in runs:
        arr = download_artifact_data(run, metric_name, cache_dir)
        if arr is not None:
            arrays.append(arr)

    if not arrays:
        print(f"[WARN] No artifact data loaded for tags {tags}")
        return None

    # Truncate to minimum num_updates across runs
    min_updates = min(a.shape[1] for a in arrays)
    arrays = [a[:, :min_updates] for a in arrays]

    return np.concatenate(arrays, axis=0), total_timesteps


def build_experiment_collection(
    spec: Dict,
    metric_name: str = DEFAULT_METRIC,
    exclude_pref: bool = True,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> ExperimentCollection:
    """Build an ExperimentCollection from a specification dict.

    ``spec`` has the form::

        {
            "algorithms": {
                "IPPO (FF)": {"base_tags": ["MARL_FINAL", "IPPO", "FF_NPS"]},
                "IPPO (RNN)": {"base_tags": ["MARL_FINAL", "IPPO", "RNN_NPS"]},
                ...
            },
            "environments": ["scratchitch", "bedbathing", ...],
        }

    For each (algorithm, environment) pair the function queries wandb for runs
    matching ``base_tags + [env_name]`` (with ``PREF`` excluded when
    ``exclude_pref`` is True).

    Args:
        spec: Specification dictionary.
        metric_name: Metric to extract.
        exclude_pref: Whether to exclude runs tagged ``PREF``.
        entity: wandb entity.
        project: wandb project.
        cache_dir: Cache directory.

    Returns:
        Populated ``ExperimentCollection``.
    """
    collection = ExperimentCollection()
    algos = spec["algorithms"]
    envs = spec["environments"]
    exclude_tags = ["PREF"] if exclude_pref else None

    for env_name in envs:
        for algo_name, algo_cfg in algos.items():
            tags = algo_cfg["base_tags"] + [env_name]
            data = extract_experiment_data(
                tags=tags,
                metric_name=metric_name,
                entity=entity,
                project=project,
                exclude_tags=exclude_tags,
                cache_dir=cache_dir,
            )
            if data is not None:
                returns_arr, total_ts = data
                collection.add(env_name, AlgorithmData(
                    name=algo_name,
                    tags=tags,
                    returns=returns_arr,
                    total_timesteps=total_ts,
                ))
            else:
                print(f"[WARN] Skipping {algo_name} / {env_name} — no data found")

    return collection


# =============================================================================
# Layer 3: Processing
# =============================================================================

def subsample_curve(
    data: np.ndarray,
    n_points: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample a curve to ``n_points`` evenly-spaced actual data points.

    Args:
        data: Array of shape ``(num_seeds, num_updates)``.
        n_points: Number of points to keep.

    Returns:
        Tuple of ``(subsampled_data, indices)`` where subsampled_data has shape
        ``(num_seeds, n_points)`` and indices are the selected column positions.
    """
    num_updates = data.shape[1]
    indices = np.linspace(0, num_updates - 1, n_points, dtype=int)
    return data[:, indices], indices


def zscore_normalize_across_envs(
    collection: ExperimentCollection,
    final_n: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Z-score normalize final returns across environments.

    For each environment, pools all algorithms' final ``final_n`` returns
    to compute env-level mean and std, then normalizes.

    Args:
        collection: Experiment data.
        final_n: Number of trailing updates to average for final performance.

    Returns:
        Nested dict ``{env: {algo: normalized_mean}}`` plus a special
        ``"__stats__"`` key per env with ``(env_mean, env_std)``.
    """
    result: Dict[str, Dict[str, float]] = {}

    for env_name in collection.env_names:
        env_data = collection.data[env_name]
        # Pool all final returns for this env
        all_finals = []
        algo_finals: Dict[str, np.ndarray] = {}
        for algo_name, ad in env_data.items():
            finals = ad.returns[:, -final_n:].mean(axis=1)  # (num_seeds,)
            algo_finals[algo_name] = finals
            all_finals.append(finals)

        pooled = np.concatenate(all_finals)
        env_mean = pooled.mean()
        env_std = pooled.std()
        if env_std == 0:
            env_std = 1.0

        result[env_name] = {"__stats__": (env_mean, env_std)}
        for algo_name, finals in algo_finals.items():
            result[env_name][algo_name] = ((finals.mean() - env_mean) / env_std)

    return result


def zscore_normalize_curves(
    collection: ExperimentCollection,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Z-score normalize full learning curves within each environment.

    Pools ALL data points across all algorithms within each environment
    to compute the env-level mean and std.

    Args:
        collection: Experiment data.

    Returns:
        Nested dict ``{env: {algo: normalized_array}}`` with same shapes.
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}

    for env_name in collection.env_names:
        env_data = collection.data[env_name]
        # Pool every data point for normalization stats
        all_points = np.concatenate([ad.returns.ravel() for ad in env_data.values()])
        env_mean = all_points.mean()
        env_std = all_points.std()
        if env_std == 0:
            env_std = 1.0

        result[env_name] = {}
        for algo_name, ad in env_data.items():
            result[env_name][algo_name] = (ad.returns - env_mean) / env_std

    return result


def minmax_normalize_across_envs(
    collection: ExperimentCollection,
    final_n: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Min-max normalize final returns across environments.

    For each environment, pools all algorithms' final ``final_n`` returns
    to compute env-level min and max, then normalizes to [0, 1].

    Args:
        collection: Experiment data.
        final_n: Number of trailing updates to average for final performance.

    Returns:
        Nested dict ``{env: {algo: normalized_mean}}`` plus a special
        ``"__stats__"`` key per env with ``(env_min, env_max)``.
    """
    result: Dict[str, Dict[str, float]] = {}

    for env_name in collection.env_names:
        env_data = collection.data[env_name]
        all_finals = []
        algo_finals: Dict[str, np.ndarray] = {}
        for algo_name, ad in env_data.items():
            finals = ad.returns[:, -final_n:].mean(axis=1)
            algo_finals[algo_name] = finals
            all_finals.append(finals)

        pooled = np.concatenate(all_finals)
        env_min = pooled.min()
        env_max = pooled.max()
        denom = env_max - env_min
        if denom == 0:
            denom = 1.0

        result[env_name] = {"__stats__": (env_min, env_max)}
        for algo_name, finals in algo_finals.items():
            result[env_name][algo_name] = (finals.mean() - env_min) / denom

    return result


def minmax_normalize_curves(
    collection: ExperimentCollection,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Min-max normalize full learning curves within each environment.

    Pools ALL data points across all algorithms within each environment
    to compute the env-level min and max, then normalizes to [0, 1].

    Args:
        collection: Experiment data.

    Returns:
        Nested dict ``{env: {algo: normalized_array}}`` with same shapes.
    """
    result: Dict[str, Dict[str, np.ndarray]] = {}

    for env_name in collection.env_names:
        env_data = collection.data[env_name]
        all_points = np.concatenate([ad.returns.ravel() for ad in env_data.values()])
        env_min = all_points.min()
        env_max = all_points.max()
        denom = env_max - env_min
        if denom == 0:
            denom = 1.0

        result[env_name] = {}
        for algo_name, ad in env_data.items():
            result[env_name][algo_name] = (ad.returns - env_min) / denom

    return result


# =============================================================================
# Layer 4: Plotting
# =============================================================================

def set_paper_style(usetex_fallback: bool = False) -> None:
    """Configure matplotlib for publication-quality figures.

    Args:
        usetex_fallback: If True, skip LaTeX rendering and use sans-serif fonts.
    """
    plt.rcdefaults()

    base = {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }

    if usetex_fallback:
        base["font.family"] = "serif"
        base["font.serif"] = ["Times New Roman"]
    else:
        base["text.usetex"] = True
        base["font.family"] = "serif"
        base["font.serif"] = ["Times New Roman"]

    plt.rcParams.update(base)
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["ytick.major.size"] = 3


def _get_color(algo_name: str) -> str:
    """Get color for an algorithm, falling back to tab10 cycle."""
    if algo_name in ALGO_COLORS:
        return ALGO_COLORS[algo_name]
    idx = hash(algo_name) % 10
    return plt.cm.tab10(idx)


def _get_marker(algo_name: str) -> str:
    """Get marker for an algorithm, falling back to circle."""
    return ALGO_MARKERS.get(algo_name, "o")


def _env_steps_axis(ad: AlgorithmData, indices: np.ndarray) -> np.ndarray:
    """Convert update indices to environment step values."""
    num_updates = ad.returns.shape[1]
    steps_per_update = ad.total_timesteps / num_updates
    return indices * steps_per_update


def _env_display(name: str) -> str:
    """Convert internal env name to display name."""
    return ENV_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    """Save figure to path (creating dirs) or show it."""
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()


def plot_learning_curves(
    collection: ExperimentCollection,
    n_subsample: int = 10,
    save_dir: Optional[str] = None,
    usetex_fallback: bool = False,
) -> None:
    """Plot per-environment learning curves with subsampled points and CI bands.

    Produces one standalone figure per environment for LaTeX composition.

    Args:
        collection: Experiment data.
        n_subsample: Number of subsampled points per curve.
        save_dir: If set, save individual figures to this directory.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
    """
    set_paper_style(usetex_fallback)

    for env_name in collection.env_names:
        fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.2))
        env_data = collection.data[env_name]

        for algo_name in sorted(env_data.keys()):
            ad = env_data[algo_name]
            sub_data, sub_idx = subsample_curve(ad.returns, n_subsample)
            x_vals = _env_steps_axis(ad, sub_idx)
            mean, lower, upper = bootstrap_ci_mean(sub_data)
            color = _get_color(algo_name)
            marker = _get_marker(algo_name)

            ax.plot(x_vals, mean, color=color, marker=marker, label=algo_name)
            ax.fill_between(x_vals, lower, upper, color=color, alpha=0.15)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Test Return")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        fig.tight_layout()

        save_path = (
            os.path.join(save_dir, f"{env_name}_learning_curves.pdf")
            if save_dir else None
        )
        _save_or_show(fig, save_path)


def plot_normalized_learning_curves(
    collection: ExperimentCollection,
    n_subsample: int = 10,
    normalization: str = "zscore",
    save_path: Optional[str] = None,
    usetex_fallback: bool = False,
) -> None:
    """Plot normalized learning curves aggregated across environments.

    One line per algorithm on a single axis.

    Args:
        collection: Experiment data.
        n_subsample: Number of subsampled points per curve.
        normalization: Normalization method, ``"zscore"`` or ``"minmax"``.
        save_path: If set, save figure to this path.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
    """
    set_paper_style(usetex_fallback)

    if normalization == "minmax":
        norm_curves = minmax_normalize_curves(collection)
    else:
        norm_curves = zscore_normalize_curves(collection)

    # Aggregate across envs: for each algo, stack all envs and average
    algo_agg: Dict[str, List[np.ndarray]] = {}
    algo_ref_ad: Dict[str, AlgorithmData] = {}
    for env_name in collection.env_names:
        for algo_name, norm_arr in norm_curves[env_name].items():
            algo_agg.setdefault(algo_name, []).append(norm_arr)
            if algo_name not in algo_ref_ad:
                algo_ref_ad[algo_name] = collection.data[env_name][algo_name]

    fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.2))

    for algo_name in sorted(algo_agg.keys()):
        arrs = algo_agg[algo_name]
        # Truncate to min length across envs
        min_len = min(a.shape[1] for a in arrs)
        stacked = np.concatenate([a[:, :min_len] for a in arrs], axis=0)
        sub_data, sub_idx = subsample_curve(stacked, n_subsample)
        x_vals = _env_steps_axis(algo_ref_ad[algo_name], sub_idx)
        mean, lower, upper = bootstrap_ci_mean(sub_data)
        color = _get_color(algo_name)
        marker = _get_marker(algo_name)

        ax.plot(x_vals, mean, color=color, marker=marker, label=algo_name)
        ax.fill_between(x_vals, lower, upper, color=color, alpha=0.15)

    ax.set_xlabel("Environment Steps")
    ylabel = "Mean Test Return" 
   # (
   #     "" if normalization == "minmax"
   #     else "Normalized Return (z-score)"
   # )
    ax.set_ylabel(ylabel)
    if normalization == "minmax":
        ref_total_timesteps = max(ad.total_timesteps for ad in algo_ref_ad.values())
        ax.xaxis.set_major_locator(ticker.MultipleLocator(ref_total_timesteps / 4))
        ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    else:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_final_returns(
    collection: ExperimentCollection,
    final_n: int = 10,
    save_dir: Optional[str] = None,
    usetex_fallback: bool = False,
) -> None:
    """Plot horizontal interval plots of final returns per environment.

    Produces one standalone figure per environment for LaTeX composition.

    Args:
        collection: Experiment data.
        final_n: Number of trailing updates to average for final performance.
        save_dir: If set, save individual figures to this directory.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
    """
    set_paper_style(usetex_fallback)

    for env_name in collection.env_names:
        env_data = collection.data[env_name]
        algo_names_present = sorted(env_data.keys())
        n_algos = len(algo_names_present)

        fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 0.5 * n_algos + 0.6))
        y_positions = np.arange(n_algos)

        bar_height = 0.6
        for i, algo_name in enumerate(algo_names_present):
            ad = env_data[algo_name]
            finals = ad.returns[:, -final_n:].mean(axis=1, keepdims=True)
            m, lo, hi = bootstrap_ci_mean(finals)
            color = _get_color(algo_name)

            ax.barh(i, hi[0] - lo[0], left=lo[0], height=bar_height,
                    color=color, alpha=0.4, edgecolor=color, linewidth=0.5)
            ax.vlines(m[0], i - bar_height / 2, i + bar_height / 2,
                      color=color, linewidth=1.5)

        ax.set_yticks(y_positions)
        ax.set_yticklabels(algo_names_present)
        ax.set_xlabel("Mean Test Return")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
        ax.invert_yaxis()
        fig.tight_layout()

        save_path = (
            os.path.join(save_dir, f"{env_name}_final_returns.pdf")
            if save_dir else None
        )
        _save_or_show(fig, save_path)


def plot_aggregate_normalized_returns(
    collection: ExperimentCollection,
    final_n: int = 10,
    normalization: str = "minmax",
    save_path: Optional[str] = None,
    usetex_fallback: bool = False,
) -> None:
    """Plot normalized final returns aggregated across environments.

    One horizontal CI interval per algorithm.

    Args:
        collection: Experiment data.
        final_n: Number of trailing updates for final performance.
        normalization: Normalization method, ``"zscore"`` or ``"minmax"``.
        save_path: If set, save figure to this path.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
    """
    set_paper_style(usetex_fallback)
    algos = collection.algo_names

    # Collect per-seed normalized finals for each algo across envs
    algo_seeds: Dict[str, List[np.ndarray]] = {}
    for env_name in collection.env_names:
        env_data = collection.data[env_name]
        # Compute env-level normalization stats from all algos
        all_finals = []
        algo_finals: Dict[str, np.ndarray] = {}
        for algo_name, ad in env_data.items():
            finals = ad.returns[:, -final_n:].mean(axis=1)  # (num_seeds,)
            algo_finals[algo_name] = finals
            all_finals.append(finals)

        pooled = np.concatenate(all_finals)
        if normalization == "minmax":
            env_min, env_max = pooled.min(), pooled.max()
            denom = env_max - env_min if env_max != env_min else 1.0
            for algo_name, finals in algo_finals.items():
                normed = (finals - env_min) / denom
                algo_seeds.setdefault(algo_name, []).append(normed)
        else:
            env_mean, env_std = pooled.mean(), pooled.std()
            env_std = env_std if env_std > 0 else 1.0
            for algo_name, finals in algo_finals.items():
                normed = (finals - env_mean) / env_std
                algo_seeds.setdefault(algo_name, []).append(normed)

    # Build plot
    names = [a for a in algos if a in algo_seeds]
    n_algos = len(names)
    fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 0.32 * n_algos + 0.5))

    bar_height = 0.6
    for i, algo_name in enumerate(names):
        seeds = np.concatenate(algo_seeds[algo_name])  # (total_seeds,)
        data_2d = seeds.reshape(-1, 1)  # (total_seeds, 1)
        m, lo, hi = bootstrap_ci_mean(data_2d)
        color = _get_color(algo_name)

        ax.barh(i, hi[0] - lo[0], left=lo[0], height=bar_height,
                color=color, alpha=0.4, edgecolor=color, linewidth=0.5)
        ax.vlines(m[0], i - bar_height / 2, i + bar_height / 2,
                  color=color, linewidth=1.5)

    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel("Mean Test Return")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_shared_legend(
    algo_names: List[str],
    save_path: Optional[str] = None,
    usetex_fallback: bool = False,
) -> None:
    """Export a standalone horizontal legend as a tight-cropped PDF.

    Creates dummy Line2D handles for each algorithm and renders them
    in a single horizontal row with no axes.

    Args:
        algo_names: Algorithm names to include in the legend.
        save_path: If set, save legend figure to this path.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
    """
    from matplotlib.lines import Line2D

    set_paper_style(usetex_fallback)

    handles = [
        Line2D(
            [0], [0],
            color=_get_color(name),
            marker=_get_marker(name),
            label=name,
            linewidth=1.2,
            markersize=4,
        )
        for name in algo_names
    ]

    fig = plt.figure(figsize=(PAPER_FULL_WIDTH, 0.4))
    fig.legend(
        handles=handles,
        labels=algo_names,
        loc="center",
        ncol=len(algo_names),
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)


def plot_sps_scaling(
    csv_path: str,
    save_path: Optional[str] = None,
    usetex_fallback: bool = False,
    show_error_bars: bool = False,
) -> None:
    """Plot steps-per-second scaling with number of parallel environments.

    One curve per environment with optional 95% CI error bars.

    Args:
        csv_path: Path to benchmark_sps.csv with columns
            env_name, num_envs, mean_sps, std_sps.
        save_path: If set, save figure to this path.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
        show_error_bars: If True, show 95% CI error bars at data points.
    """
    set_paper_style(usetex_fallback)
    df = pd.read_csv(csv_path)
    cmap = plt.cm.tab10

    n_trials = 12
    z = 1.96  # 95% CI

    fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH * 1.5, PAPER_COLUMN_WIDTH))

    env_real_names = {
        'scratchitch': 'Scratching',
        'bedbathing': 'Bed Bathing',
        'armmanipulation': 'Arm Assist',
        'teethbrushing': 'Tooth Brushing',
        'feeding': 'Feeding'
    }
    
    for i, env_name in enumerate(df["env_name"].unique()):
        env_df = df[df["env_name"] == env_name].sort_values("num_envs")
        x = env_df["num_envs"].values
        mean = env_df["mean_sps"].values
        std = env_df["std_sps"].values
        color = cmap(i)
        # label = _env_display(env_name)
        label = env_real_names[env_name]

        if show_error_bars:
            ci = z * std / np.sqrt(n_trials)
            ax.errorbar(x, mean, yerr=ci, color=color, marker="o",
                        capsize=3, label=label)
        else:
            ax.plot(x, mean, color=color, marker="o", label=label)

    num_envs_vals = sorted(df["num_envs"].unique())
    ax.set_xticks(num_envs_vals)
    ax.set_xticklabels([str(v) for v in num_envs_vals], rotation=90)

    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.get_major_formatter().set_powerlimits((0, 0))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    ax.set_xlabel("Number of Parallel Environments")
    ax.set_ylabel("Steps per Second")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_or_show(fig, save_path)
