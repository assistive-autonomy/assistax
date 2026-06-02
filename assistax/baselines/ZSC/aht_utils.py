"""Utility functions for Ad Hoc Teamwork (AHT) partner selection and splitting."""

import numpy as np
import pandas as pd
from typing import Any


def split_partners_extreme(
    all_partners: pd.DataFrame,
    extreme_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split partners into extreme and non-extreme sets based on preference dimensions.

    Args:
        all_partners: DataFrame with columns w_speed, w_force, w_touch, etc.
        extreme_config: The EXTREME_SPLIT config dict with keys:
            - num_extreme: Number of extreme agents to select.
            - mode: "single", "multi", or "composite".
            - column/end (single mode), columns (multi mode), dimensions (composite mode).

    Returns:
        (extreme_set, remainder_set) DataFrames. Caller assigns these to
        train/test based on extreme_config["role"].
    """
    num_extreme = extreme_config["num_extreme"]
    mode = extreme_config.get("mode", "single")

    if mode == "single":
        extreme_idx = _select_single(all_partners, extreme_config)
    elif mode == "multi":
        extreme_idx = _select_multi(all_partners, extreme_config)
    elif mode == "composite":
        extreme_idx = _select_composite(all_partners, extreme_config)
    else:
        raise ValueError(f"Unknown EXTREME_SPLIT mode: {mode!r}. Expected 'single', 'multi', or 'composite'.")

    extreme_set = all_partners.loc[extreme_idx].reset_index(drop=True)
    remainder_set = all_partners.drop(index=extreme_idx).reset_index(drop=True)

    print(f"  Extreme split ({mode}): selected {len(extreme_set)} extreme, {len(remainder_set)} remainder")
    return extreme_set, remainder_set


def _select_by_end(sorted_df: pd.DataFrame, num_extreme: int, end: str) -> list[int]:
    """Select indices from a sorted DataFrame based on end specification.

    Args:
        sorted_df: DataFrame sorted ascending by the column of interest.
        num_extreme: Total number of agents to select.
        end: "min", "max", or "both".

    Returns:
        List of original DataFrame indices for the selected agents.
    """
    if end == "min":
        return list(sorted_df.index[:num_extreme])
    elif end == "max":
        return list(sorted_df.index[-num_extreme:])
    elif end == "both":
        n_bottom = num_extreme // 2
        n_top = num_extreme - n_bottom
        bottom = list(sorted_df.index[:n_bottom])
        top = list(sorted_df.index[-n_top:])
        return bottom + top
    else:
        raise ValueError(f"Unknown end: {end!r}. Expected 'min', 'max', or 'both'.")


def _select_single(all_partners: pd.DataFrame, cfg: dict[str, Any]) -> list[int]:
    """Select extreme agents along a single preference dimension."""
    column = cfg["column"]
    end = cfg.get("end", "both")
    num_extreme = cfg["num_extreme"]
    sorted_df = all_partners.sort_values(by=column)
    indices = _select_by_end(sorted_df, num_extreme, end)
    print(f"  Single mode: column={column}, end={end}, selected {len(indices)} agents")
    return indices


def _select_multi(all_partners: pd.DataFrame, cfg: dict[str, Any]) -> list[int]:
    """Select extreme agents along multiple preference dimensions independently."""
    columns_cfg = cfg["columns"]
    num_extreme = cfg["num_extreme"]
    all_indices: set[int] = set()

    for entry in columns_cfg:
        column = entry["column"]
        end = entry.get("end", "both")
        sorted_df = all_partners.sort_values(by=column)
        indices = _select_by_end(sorted_df, num_extreme, end)
        print(f"  Multi mode: column={column}, end={end}, selected {len(indices)} agents")
        all_indices.update(indices)

    print(f"  Multi mode total (deduplicated): {len(all_indices)} agents")
    return list(all_indices)


def _select_composite(all_partners: pd.DataFrame, cfg: dict[str, Any]) -> list[int]:
    """Select agents farthest from the centroid across all preference dimensions."""
    default_dims = ["w_speed", "w_force", "w_touch"]
    dimensions = cfg.get("dimensions", default_dims)
    num_extreme = cfg["num_extreme"]

    values = all_partners[dimensions].values.astype(float)

    # Min-max normalize each dimension to [0, 1]
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0  # avoid division by zero for constant columns
    normalized = (values - mins) / ranges

    # Compute centroid and Euclidean distance
    centroid = normalized.mean(axis=0)
    distances = np.sqrt(((normalized - centroid) ** 2).sum(axis=1))

    # Select the num_extreme most distant agents
    top_indices = np.argsort(distances)[-num_extreme:]
    indices = list(all_partners.index[top_indices])
    print(f"  Composite mode: dimensions={dimensions}, selected {len(indices)} most distant agents")
    return indices
