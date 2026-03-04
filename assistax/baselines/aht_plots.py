"""
AHT (Ad Hoc Teamwork) Plotting Pipeline for Assistax.

Pulls dual evaluation data (train partners + test partners) from wandb
artifacts and produces publication-quality learning curve figures showing
zero-shot coordination generalization.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import wandb

from assistax.baselines.paper_plots import (
    ALGO_COLORS,
    ALGO_MARKERS,
    DEFAULT_CACHE_DIR,
    DEFAULT_ENTITY,
    DEFAULT_METRIC,
    DEFAULT_PROJECT,
    PAPER_COLUMN_WIDTH,
    PAPER_FULL_WIDTH,
    _env_display,
    _get_color,
    _get_marker,
    _save_or_show,
    fetch_runs_by_tags,
    set_paper_style,
    subsample_curve,
)
from assistax.baselines.sweep_plots import bootstrap_ci_mean
from assistax.baselines.utils import load_compact_npz

# Distinct colours for train vs test curves in AHT plots.
# Each entry maps algorithm name -> (test_color, train_color).
AHT_TRAIN_TEST_COLORS: Dict[str, Tuple[str, str]] = {
    "PPO AHT (FF)": ("#ff7f0e", "#1f77b4"),   # orange (test), blue (train)
    "SAC AHT (FF)": ("#d62728", "#9467bd"),    # red (test), purple (train)
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AHTCurveData:
    """Container for one AHT algorithm's dual evaluation data in one environment."""

    name: str                  # e.g., "PPO AHT (FF)"
    tags: List[str]
    train_returns: np.ndarray  # (num_seeds, num_updates) — eval with train partners
    test_returns: np.ndarray   # (num_seeds, num_updates) — eval with test partners
    total_timesteps: int


@dataclass
class AHTCollection:
    """Collection of AHT evaluation data across environments and algorithms."""

    data: Dict[str, Dict[str, AHTCurveData]] = field(default_factory=dict)

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

    def add(self, env_name: str, curve_data: AHTCurveData) -> None:
        """Add AHT curve data for an environment."""
        if env_name not in self.data:
            self.data[env_name] = {}
        self.data[env_name][curve_data.name] = curve_data


# =============================================================================
# Layer 1: Data Extraction (artifact-based)
# =============================================================================

def _download_artifact_by_name(
    run,
    artifact_name: str,
    metric_name: str,
    cache_dir: str,
    cache_suffix: str,
) -> Optional[np.ndarray]:
    """Download a single artifact by name and load the metric.

    Args:
        run: wandb Run object.
        artifact_name: Full artifact name (without entity/project prefix).
        metric_name: Metric file to load from the artifact.
        cache_dir: Local cache root.
        cache_suffix: Subdirectory suffix for caching (e.g., "_train").

    Returns:
        Loaded numpy array or None on failure.
    """
    run_cache = os.path.join(cache_dir, f"{run.id}{cache_suffix}")
    npz_path = os.path.join(run_cache, f"{metric_name}.npz")

    if os.path.exists(npz_path):
        return load_compact_npz(npz_path)

    try:
        api = wandb.Api()
        artifact_path = f"{run.entity}/{run.project}/{artifact_name}"
        artifact = api.artifact(artifact_path)
        artifact.download(root=run_cache)
    except Exception as e:
        print(f"[WARN] Could not download artifact '{artifact_name}' for run {run.name}: {e}")
        return None

    npz_path = os.path.join(run_cache, f"{metric_name}.npz")
    if not os.path.exists(npz_path):
        available = os.listdir(run_cache) if os.path.isdir(run_cache) else []
        print(f"[WARN] Metric '{metric_name}' not in artifact '{artifact_name}'. Available: {available}")
        return None

    return load_compact_npz(npz_path)


def _assign_train_test(
    run,
    arr_a: np.ndarray,
    arr_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Auto-detect which array is train vs test by comparing to run summary scalars.

    Compares the mean of the last few checkpoints of each array against the
    run's logged ``eval_train/team_return_mean_final_mean`` and
    ``eval_test/team_return_mean_final_mean`` summary scalars.

    Args:
        run: wandb Run object.
        arr_a: First array candidate.
        arr_b: Second array candidate.
        label_a: Label for arr_a (for logging).
        label_b: Label for arr_b (for logging).

    Returns:
        (train_array, test_array) tuple.
    """
    train_key = "eval_train/team_return_mean_final_mean"
    test_key = "eval_test/team_return_mean_final_mean"

    summary = run.summary
    train_scalar = summary.get(train_key)
    test_scalar = summary.get(test_key)

    if train_scalar is None or test_scalar is None:
        print(f"  [INFO] Summary scalars not found for run {run.name}, using default order ({label_a}=train, {label_b}=test)")
        return arr_a, arr_b

    # Compare last few checkpoints' mean to the summary scalars
    window = min(10, arr_a.shape[1], arr_b.shape[1])
    mean_a = float(np.mean(arr_a[:, -window:]))
    mean_b = float(np.mean(arr_b[:, -window:]))

    # Try both assignments and pick the one with smaller total error
    err_ab = abs(mean_a - train_scalar) + abs(mean_b - test_scalar)  # a=train, b=test
    err_ba = abs(mean_b - train_scalar) + abs(mean_a - test_scalar)  # b=train, a=test

    if err_ba < err_ab:
        print(f"  [INFO] Swapped artifact assignment for run {run.name} ({label_b}=train, {label_a}=test)")
        return arr_b, arr_a

    return arr_a, arr_b


def download_aht_artifact_data(
    run,
    metric_name: str = DEFAULT_METRIC,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Download train and test evaluation artifacts for an AHT run.

    Tries suffix-based naming first (``_train_partners``, ``_test_partners``),
    then falls back to version-based (``v0``, ``v1``) with auto-detection.

    Args:
        run: wandb Run object.
        metric_name: Metric file to load from each artifact.
        cache_dir: Local cache directory.

    Returns:
        (train_array, test_array) tuple, or None if both fail.
    """
    base_name = f"evaluation_data_{run.name}"

    # --- Strategy 1: suffix-based naming (from fixed upload code) ---
    train_arr = _download_artifact_by_name(
        run, f"{base_name}_train_partners:latest", metric_name, cache_dir, "_train",
    )
    test_arr = _download_artifact_by_name(
        run, f"{base_name}_test_partners:latest", metric_name, cache_dir, "_test",
    )
    if train_arr is not None and test_arr is not None:
        print(f"  Loaded suffix-named artifacts for run {run.name}")
        return train_arr, test_arr

    # --- Strategy 2: version-based (legacy, both under same artifact name) ---
    arr_v0 = _download_artifact_by_name(
        run, f"{base_name}:v0", metric_name, cache_dir, "_v0",
    )
    arr_v1 = _download_artifact_by_name(
        run, f"{base_name}:v1", metric_name, cache_dir, "_v1",
    )
    if arr_v0 is not None and arr_v1 is not None:
        print(f"  Loaded version-based artifacts for run {run.name}, auto-detecting train/test...")
        return _assign_train_test(run, arr_v0, arr_v1, "v0", "v1")

    # --- Partial fallback ---
    if train_arr is not None or test_arr is not None:
        print(f"[WARN] Only one suffix-named artifact found for run {run.name}")
    if arr_v0 is not None or arr_v1 is not None:
        print(f"[WARN] Only one versioned artifact found for run {run.name}")

    return None


def fetch_aht_runs(
    tags: List[str],
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    exclude_tags: Optional[List[str]] = None,
    network_filter: Optional[Dict[str, bool]] = None,
) -> list:
    """Fetch wandb runs for AHT experiments, with optional network config filtering.

    Args:
        tags: Tags that must all be present.
        entity: wandb entity.
        project: wandb project.
        exclude_tags: Tags to exclude.
        network_filter: Optional dict with keys like ``"recurrent"`` and
            ``"agent_param_sharing"`` to filter by run config.

    Returns:
        List of matching wandb Run objects.
    """
    runs = fetch_runs_by_tags(tags, entity, project, exclude_tags)

    if network_filter:
        filtered = []
        for r in runs:
            net_cfg = r.config.get("network", {})
            match = all(net_cfg.get(k) == v for k, v in network_filter.items())
            if match:
                filtered.append(r)
        print(f"  After network filter {network_filter}: {len(filtered)}/{len(runs)} runs")
        runs = filtered

    return runs


def extract_aht_experiment_data(
    tags: List[str],
    metric_name: str = DEFAULT_METRIC,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    exclude_tags: Optional[List[str]] = None,
    network_filter: Optional[Dict[str, bool]] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Fetch AHT runs, download dual artifacts, concatenate across seeds.

    Args:
        tags: Tags for filtering runs.
        metric_name: Metric to extract.
        entity: wandb entity.
        project: wandb project.
        exclude_tags: Tags to exclude.
        network_filter: Network config filter.
        cache_dir: Cache directory.

    Returns:
        (train_returns, test_returns, total_timesteps) or None.
    """
    runs = fetch_aht_runs(tags, entity, project, exclude_tags, network_filter)
    if not runs:
        print(f"[WARN] No AHT runs found for tags {tags}")
        return None

    total_timesteps = int(runs[0].config.get("TOTAL_TIMESTEPS", 0))

    train_arrays, test_arrays = [], []
    for run in runs:
        result = download_aht_artifact_data(run, metric_name, cache_dir)
        if result is not None:
            train_arr, test_arr = result
            train_arrays.append(train_arr)
            test_arrays.append(test_arr)

    if not train_arrays:
        print(f"[WARN] No AHT artifact data loaded for tags {tags}")
        return None

    # Truncate to minimum num_updates across all runs and both splits
    min_updates = min(
        min(a.shape[1] for a in train_arrays),
        min(a.shape[1] for a in test_arrays),
    )
    train_arrays = [a[:, :min_updates] for a in train_arrays]
    test_arrays = [a[:, :min_updates] for a in test_arrays]

    return (
        np.concatenate(train_arrays, axis=0),
        np.concatenate(test_arrays, axis=0),
        total_timesteps,
    )


def build_aht_collection(
    spec: Dict,
    metric_name: str = DEFAULT_METRIC,
    entity: str = DEFAULT_ENTITY,
    project: str = DEFAULT_PROJECT,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> AHTCollection:
    """Build an AHTCollection from a specification dict.

    ``spec`` has the form::

        {
            "algorithms": {
                "PPO AHT (FF)": {
                    "base_tags": ["ZSC", "PPO", "AHT"],
                    "network_filter": {"recurrent": False, "agent_param_sharing": False},
                },
                ...
            },
            "environments": ["scratchitch", "bedbathing"],
        }

    Args:
        spec: Specification dictionary.
        metric_name: Metric to extract.
        entity: wandb entity.
        project: wandb project.
        cache_dir: Cache directory.

    Returns:
        Populated AHTCollection.
    """
    collection = AHTCollection()
    algos = spec["algorithms"]
    envs = spec["environments"]

    for env_name in envs:
        for algo_name, algo_cfg in algos.items():
            tags = algo_cfg["base_tags"] + [env_name]
            network_filter = algo_cfg.get("network_filter")
            exclude_tags = algo_cfg.get("exclude_tags")

            data = extract_aht_experiment_data(
                tags=tags,
                metric_name=metric_name,
                entity=entity,
                project=project,
                exclude_tags=exclude_tags,
                network_filter=network_filter,
                cache_dir=cache_dir,
            )
            if data is not None:
                train_ret, test_ret, total_ts = data
                collection.add(env_name, AHTCurveData(
                    name=algo_name,
                    tags=tags,
                    train_returns=train_ret,
                    test_returns=test_ret,
                    total_timesteps=total_ts,
                ))
            else:
                print(f"[WARN] Skipping {algo_name} / {env_name} — no AHT data found")

    return collection


# =============================================================================
# Layer 2: Plotting
# =============================================================================

def _aht_steps_axis(curve: AHTCurveData, indices: np.ndarray) -> np.ndarray:
    """Convert update indices to environment step values."""
    num_updates = curve.train_returns.shape[1]
    steps_per_update = curve.total_timesteps / num_updates
    return indices * steps_per_update


def plot_aht_learning_curves(
    collection: AHTCollection,
    n_subsample: int = 10,
    save_dir: Optional[str] = None,
    usetex_fallback: bool = False,
) -> None:
    """Plot per-environment AHT learning curves with train/test split.

    For each algorithm: **solid line** = eval_test (zero-shot coordination),
    **dashed line** = eval_train (seen partners), same color.
    Bootstrap CI bands on both curves.

    Args:
        collection: AHT experiment data.
        n_subsample: Number of subsampled points per curve.
        save_dir: If set, save individual figures to this directory.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
    """
    set_paper_style(usetex_fallback)

    for env_name in collection.env_names:
        fig, ax = plt.subplots(figsize=(PAPER_COLUMN_WIDTH, 2.2))
        env_data = collection.data[env_name]

        for algo_name in sorted(env_data.keys()):
            cd = env_data[algo_name]
            fallback = _get_color(algo_name)
            test_color, train_color = AHT_TRAIN_TEST_COLORS.get(
                algo_name, (fallback, fallback)
            )
            marker = _get_marker(algo_name)

            # --- Test curve (solid — zero-shot coordination) ---
            test_sub, test_idx = subsample_curve(cd.test_returns, n_subsample)
            x_vals = _aht_steps_axis(cd, test_idx)
            mean_t, lo_t, hi_t = bootstrap_ci_mean(test_sub)
            ax.plot(x_vals, mean_t, color=test_color, marker=marker,
                    linestyle="-", label=f"{algo_name} (test)")
            ax.fill_between(x_vals, lo_t, hi_t, color=test_color, alpha=0.15)

            # --- Train curve (dashed — seen partners) ---
            train_sub, train_idx = subsample_curve(cd.train_returns, n_subsample)
            x_train = _aht_steps_axis(cd, train_idx)
            mean_tr, lo_tr, hi_tr = bootstrap_ci_mean(train_sub)
            ax.plot(x_train, mean_tr, color=train_color, marker=marker,
                    linestyle="-", label=f"{algo_name} (train)")
            ax.fill_between(x_train, lo_tr, hi_tr, color=train_color, alpha=0.08)

            # --- Generalization gap arrow at last datapoint ---
            target_x = cd.total_timesteps
            gap_idx_test = int(np.argmin(np.abs(x_vals - target_x)))
            gap_idx_train = int(np.argmin(np.abs(x_train - target_x)))
            gap_x = x_vals[gap_idx_test]
            train_y_at_gap = mean_tr[gap_idx_train]
            test_y_at_gap = mean_t[gap_idx_test]
            ax.annotate(
                "",
                xy=(gap_x, train_y_at_gap),
                xytext=(gap_x, test_y_at_gap),
                arrowprops=dict(arrowstyle="<->", color="red", lw=2.5),
            )

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Return")
        ax.set_title(_env_display(env_name))
        fig.tight_layout()

        save_path = (
            os.path.join(save_dir, f"{env_name}_aht_learning_curves.pdf")
            if save_dir else None
        )
        _save_or_show(fig, save_path)


def plot_aht_shared_legend(
    algo_names: List[str],
    save_path: Optional[str] = None,
    usetex_fallback: bool = False,
) -> None:
    """Export a standalone legend showing train / test colour convention.

    Args:
        algo_names: Algorithm names to include.
        save_path: If set, save legend figure.
        usetex_fallback: Use sans-serif fonts if LaTeX is unavailable.
    """
    from matplotlib.lines import Line2D

    set_paper_style(usetex_fallback)

    handles = []
    labels = []
    for name in algo_names:
        fallback = _get_color(name)
        test_color, train_color = AHT_TRAIN_TEST_COLORS.get(
            name, (fallback, fallback)
        )
        marker = _get_marker(name)
        # Train (solid)
        handles.append(Line2D(
            [0], [0], color=train_color, marker=marker, linestyle="-",
            linewidth=1.2, markersize=4,
        ))
        labels.append(f"{name} (train)")
        # Test (solid)
        handles.append(Line2D(
            [0], [0], color=test_color, marker=marker, linestyle="-",
            linewidth=1.2, markersize=4,
        ))
        labels.append(f"{name} (test)")

    fig = plt.figure(figsize=(PAPER_FULL_WIDTH, 0.6))
    fig.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=min(len(handles), 6),
        frameon=False,
        fontsize=7,
    )
    fig.tight_layout()
    _save_or_show(fig, save_path)


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # Example spec — adjust tags/envs to match your wandb runs
    spec = {
        "algorithms": {
            "PPO AHT (FF)": {
                "base_tags": ["ZSC", "PPO", "AHT"],
                "network_filter": {"recurrent": False, "agent_param_sharing": False},
            },
        },
        "environments": ["scratchitch"],
    }

    print("Building AHT collection from wandb...")
    collection = build_aht_collection(spec)

    if collection.env_names:
        print(f"\nPlotting AHT learning curves for: {collection.env_names}")
        plot_aht_learning_curves(
            collection,
            n_subsample=10,
            save_dir="outputs/aht_plots",
            usetex_fallback=True,
        )
        plot_aht_shared_legend(
            collection.algo_names,
            save_path="outputs/aht_plots/aht_legend.pdf",
            usetex_fallback=True,
        )
        print("Done!")
    else:
        print("No data found. Check your wandb tags and project settings.")
