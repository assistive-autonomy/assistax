"""Crossplay heatmap plotting for Assistax.

Loads NxN crossplay return matrices from .npy files and produces
publication-quality square heatmap visualizations.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from assistax.baselines.paper_plots import (
    ALGO_COLORS,
    ENV_DISPLAY_NAMES,
    PAPER_COLUMN_WIDTH,
    PAPER_FULL_WIDTH,
    set_paper_style,
)


def load_crossplay_results(path: str) -> dict:
    """Load crossplay results from a .npy file.

    Args:
        path: Path to the .npy file saved by crossplay evaluation.

    Returns:
        Dictionary keyed by algorithm name, each containing returns,
        uuid lists, and team uuid lists.
    """
    return np.load(path, allow_pickle=True).item()


def build_crossplay_matrix(
    alg_data: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build an NxN return matrix with diagonal-aligned team pairs.

    Reorders rows and columns so that robot_team_uuids[i] ==
    human_team_uuids[i], placing trained partners on the diagonal.

    Args:
        alg_data: Single algorithm's data dict with keys 'returns',
            'robot_uuids', 'human_uuids', 'robot_team_uuids',
            'human_team_uuids'.

    Returns:
        Tuple of (matrix, diagonal_mask) where matrix[i, j] is the
        return when robot i plays with human j, and diagonal_mask[i]
        is True where robot i's team matches human i's team.
    """
    robot_uuids = alg_data["robot_uuids"]
    human_uuids = alg_data["human_uuids"]
    robot_team_uuids = alg_data["robot_team_uuids"]
    human_team_uuids = alg_data["human_team_uuids"]
    returns = alg_data["returns"]

    n_robots = len(robot_uuids)
    n_humans = len(human_uuids)

    # Build raw matrix: rows=robots (ordered by robot_uuids), cols=humans
    raw_matrix = np.zeros((n_robots, n_humans))
    for i, r_uuid in enumerate(robot_uuids):
        raw_matrix[i, :] = returns[r_uuid]

    # Reorder columns so human[i] has same team_uuid as robot[i]
    human_team_to_idx = {team: i for i, team in enumerate(human_team_uuids)}
    col_order = [human_team_to_idx[t] for t in robot_team_uuids]
    matrix = raw_matrix[:, col_order]

    # Reordered human team uuids
    reordered_human_teams = [human_team_uuids[j] for j in col_order]

    # Diagonal mask: True where robot's team matches human's team
    diagonal_mask = np.array([
        robot_team_uuids[i] == reordered_human_teams[i]
        for i in range(n_robots)
    ])

    return matrix, diagonal_mask


def plot_crossplay_heatmap(
    matrix: np.ndarray,
    diagonal_mask: np.ndarray,
    algo_name: str = "",
    env_name: str = "",
    usetex_fallback: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a square crossplay heatmap.

    Args:
        matrix: NxN return matrix (rows=robots, cols=humans).
        diagonal_mask: Boolean array; True for diagonal (trained partner) entries.
        algo_name: Algorithm name for title and accent color.
        env_name: Environment name for title.
        usetex_fallback: Use sans-serif fonts if LaTeX unavailable.
        figsize: Optional figure size override.

    Returns:
        Tuple of (Figure, Axes).
    """
    set_paper_style(usetex_fallback)

    n = matrix.shape[0]
    if figsize is None:
        side = max(PAPER_COLUMN_WIDTH, min(PAPER_FULL_WIDTH, 0.6 * n + 1.5))
        figsize = (side, side)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap="viridis", aspect="equal", interpolation="nearest")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Return")

    # Diagonal highlight: thin box outlines on diagonal cells
    for i in range(n):
        if diagonal_mask[i]:
            rect = patches.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                linewidth=1.5, edgecolor="black", facecolor="none",
            )
            ax.add_patch(rect)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n))
    ax.set_xlabel("Human policy index")
    ax.set_ylabel("Robot policy index")

    # Title
    env_display = ENV_DISPLAY_NAMES.get(env_name, env_name.replace("_", " ").title())
    title_parts = [p for p in [algo_name, env_display] if p]
    if title_parts:
        title_color = ALGO_COLORS.get(algo_name, "black")
        ax.set_title(" — ".join(title_parts), color=title_color)

    fig.tight_layout()
    return fig, ax


def plot_all_crossplay(
    results_path: str,
    save_dir: Optional[str] = None,
    env_name: Optional[str] = None,
    usetex_fallback: bool = False,
) -> Dict[str, Tuple[plt.Figure, plt.Axes]]:
    """Load crossplay results and plot one heatmap per algorithm.

    Args:
        results_path: Path to the crossplay .npy results file.
        save_dir: If set, save figures as PNGs in this directory.
        env_name: Environment name for titles.
        usetex_fallback: Use sans-serif fonts if LaTeX unavailable.

    Returns:
        Dictionary mapping algorithm name to (Figure, Axes) tuples.
    """
    results = load_crossplay_results(results_path)
    figures: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}

    for algo_name, alg_data in results.items():
        matrix, diag_mask = build_crossplay_matrix(alg_data)
        fig, ax = plot_crossplay_heatmap(
            matrix,
            diag_mask,
            algo_name=algo_name,
            env_name=env_name or "",
            usetex_fallback=usetex_fallback,
        )
        figures[algo_name] = (fig, ax)

        if save_dir:
            save_path = Path(save_dir) / f"crossplay_{algo_name.replace(' ', '_').lower()}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            print(f"Saved: {save_path}")

    if not save_dir:
        plt.show()

    return figures


def build_combined_crossplay_matrix(
    results: Dict[str, dict],
) -> Tuple[np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """Build a combined return matrix across all algorithms.

    Rows are grouped by algorithm (each algo's robots form a row-block).
    Columns contain ALL humans, grouped by their source algorithm so that
    trained partner cells lie on the block diagonal.

    Args:
        results: Dictionary keyed by algorithm name, each containing
            'returns', 'robot_uuids', 'human_uuids', 'robot_team_uuids',
            'human_team_uuids'.

    Returns:
        Tuple of (combined_matrix, block_diagonal_mask, algo_names_ordered,
        row_boundaries, col_boundaries) where boundaries are cumulative
        counts marking where each algorithm block starts.
    """
    algo_names = list(results.keys())

    # All algorithms share the same human_uuids list; grab it from first entry
    first_data = next(iter(results.values()))
    all_human_uuids = list(first_data["human_uuids"])
    all_human_team_uuids = list(first_data["human_team_uuids"])
    n_humans = len(all_human_uuids)

    # Map each human team_uuid to its source algorithm
    # A human's source algo is the algo whose robot_team_uuids contain that team
    human_team_to_source_algo: Dict[str, str] = {}
    for algo_name, alg_data in results.items():
        robot_teams = set(alg_data["robot_team_uuids"])
        for h_team in all_human_team_uuids:
            if h_team in robot_teams and h_team not in human_team_to_source_algo:
                human_team_to_source_algo[h_team] = algo_name

    # Build column ordering: group humans by source algo, within each group
    # order so trained partners align on the block diagonal
    col_order: List[int] = []
    col_boundaries = [0]
    human_idx_by_uuid = {uuid: i for i, uuid in enumerate(all_human_uuids)}

    for algo_name in algo_names:
        alg_data = results[algo_name]
        robot_team_uuids = alg_data["robot_team_uuids"]

        # Humans belonging to this algo's group, ordered to match robot order
        algo_human_indices = []
        seen_teams = set()
        for r_team in robot_team_uuids:
            if r_team in seen_teams:
                continue
            seen_teams.add(r_team)
            # Find the human with this team_uuid
            for h_idx, h_team in enumerate(all_human_team_uuids):
                if h_team == r_team and h_idx not in algo_human_indices:
                    algo_human_indices.append(h_idx)
                    break

        # Add any remaining humans from this algo group not yet included
        for h_idx, h_team in enumerate(all_human_team_uuids):
            if (human_team_to_source_algo.get(h_team) == algo_name
                    and h_idx not in algo_human_indices):
                algo_human_indices.append(h_idx)

        col_order.extend(algo_human_indices)
        col_boundaries.append(len(col_order))

    # Build row blocks and stack
    row_blocks: List[np.ndarray] = []
    row_boundaries = [0]

    for algo_name in algo_names:
        alg_data = results[algo_name]
        robot_uuids = alg_data["robot_uuids"]
        returns = alg_data["returns"]
        n_robots = len(robot_uuids)

        # Raw sub-matrix: rows = this algo's robots, cols = all humans
        raw_block = np.zeros((n_robots, n_humans))
        for i, r_uuid in enumerate(robot_uuids):
            raw_block[i, :] = returns[r_uuid]

        # Reorder columns to combined ordering
        row_blocks.append(raw_block[:, col_order])
        row_boundaries.append(row_boundaries[-1] + n_robots)

    combined_matrix = np.vstack(row_blocks)

    # Reordered human team uuids in combined column order
    reordered_human_teams = [all_human_team_uuids[j] for j in col_order]

    # Build block-diagonal mask (2D boolean)
    total_rows = combined_matrix.shape[0]
    total_cols = combined_matrix.shape[1]
    block_mask = np.zeros((total_rows, total_cols), dtype=bool)

    for a_idx, algo_name in enumerate(algo_names):
        alg_data = results[algo_name]
        robot_team_uuids = alg_data["robot_team_uuids"]
        r_start = row_boundaries[a_idx]
        r_end = row_boundaries[a_idx + 1]

        for ri, r_team in enumerate(robot_team_uuids):
            for cj in range(total_cols):
                if reordered_human_teams[cj] == r_team:
                    block_mask[r_start + ri, cj] = True

    return (combined_matrix, block_mask, algo_names,
            row_boundaries, col_boundaries)


def plot_combined_crossplay(
    results_path: str,
    save_path: Optional[str] = None,
    env_name: Optional[str] = None,
    usetex_fallback: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot a single combined crossplay heatmap across all algorithms.

    Args:
        results_path: Path to the crossplay .npy results file.
        save_path: If set, save figure to this path.
        env_name: Environment name (unused, kept for API consistency).
        usetex_fallback: Use sans-serif fonts if LaTeX unavailable.

    Returns:
        Tuple of (Figure, Axes).
    """
    set_paper_style(usetex_fallback)

    results = load_crossplay_results(results_path)
    matrix, block_mask, algo_names, row_bounds, col_bounds = (
        build_combined_crossplay_matrix(results)
    )

    n_rows, n_cols = matrix.shape
    side = max(PAPER_COLUMN_WIDTH, min(PAPER_FULL_WIDTH, 0.4 * max(n_rows, n_cols) + 1.5))
    fig, ax = plt.subplots(figsize=(side, side))

    im = ax.imshow(
        matrix, cmap="viridis", aspect="equal",
        interpolation="nearest", origin="lower",
    )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Return")

    # Block-diagonal highlight: black rectangle outlines on trained-partner cells
    for r in range(n_rows):
        for c in range(n_cols):
            if block_mask[r, c]:
                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    linewidth=1.5, edgecolor="black", facecolor="none",
                )
                ax.add_patch(rect)

    # No title, no gridlines
    ax.grid(False)

    # Simple integer tick labels
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(range(n_cols))
    ax.set_yticklabels(range(n_rows))
    ax.set_xlabel("Human policy index")
    ax.set_ylabel("Robot policy index")

    fig.tight_layout()

    if save_path:
        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_p)
        print(f"Saved: {save_p}")

    return fig, ax
