import os
import os.path as osp
import numpy as np
import jax.numpy as jnp
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--load-dirs",
    help="A list of directories containing to be concatenated.",
    nargs="+",
    default=["."],
)
parser.add_argument(
    "--save-metrics",
    help="A list of metrics to save.",
    nargs="+",
    default=["returned_episode_returns"],
)
parser.add_argument(
    "-o",
    "--output-dir",
    help="The directory to save the resulting npy files to. Metrics will be placed in the metrics subdirectory, and hyperparameters placed in the hparams subdirectory.",
)
parser.add_argument(
    "--num-seeds",
    help="Only include runs with this many seeds. If not specified, includes all runs.",
    type=int,
    default=None,
)
args = parser.parse_args()

def ignore_dir(d):
    if "." in d:
        return True
    return False

def has_required_files(dirpath):
    """Check if directory contains all required .npy files"""
    required_files = ["hparams.npy", "metrics.npy", "returns.npy"]
    for f in required_files:
        if not os.path.exists(os.path.join(dirpath, f)):
            return False
    return True

all_hparams = defaultdict(list)
all_metrics = defaultdict(list)
all_returns = []

for dirpath in args.load_dirs:
    key_dirs = [d for d in os.listdir(dirpath) if not ignore_dir(d)]
    files = os.listdir(dirpath)
    for key_dir in key_dirs:
        full_path = os.path.join(dirpath, key_dir)
        
        # Skip directories that don't have the required files
        if not has_required_files(full_path):
            print(f"Skipping {full_path} - missing required .npy files")
            continue
            
        hparams = jnp.load(
            os.path.join(full_path, "hparams.npy"), allow_pickle=True
        ).item()
        returns = jnp.load(
            os.path.join(full_path, "returns.npy"), allow_pickle=True
        )
        metrics = jnp.load(
            os.path.join(full_path, "metrics.npy"), allow_pickle=True
        ).item()
        
        # Check seed dimension if filter is specified
        num_seeds_in_run = metrics["returned_episode_returns"].shape[1]
        if args.num_seeds is not None and num_seeds_in_run != args.num_seeds:
            print(f"Skipping {full_path} - has {num_seeds_in_run} seeds, expected {args.num_seeds}")
            continue
        
        n_configs = metrics["returned_episode_returns"].shape[0]
        for hparam_key in hparams:
            all_hparams[hparam_key].append(
                hparams[hparam_key]
                if isinstance(hparams[hparam_key], jnp.ndarray)
                else jnp.full(n_configs, hparams[hparam_key])
            )
        for metric_key in args.save_metrics:
            all_metrics[metric_key].append(metrics[metric_key])
        all_returns.append(returns)

all_hparams = {key: np.concatenate(val) for key, val in all_hparams.items()}
all_metrics = {key: np.concatenate(val) for key, val in all_metrics.items()}
all_returns = jnp.concat(all_returns)

output_dir = args.output_dir
existing_hparam_dir = osp.join(output_dir, "hparam")
existing_metric_dir = osp.join(output_dir, "metric")

# Merge with existing data if output directory already has results
if osp.isdir(existing_hparam_dir) or osp.isdir(existing_metric_dir):
    print(f"Existing data found in {output_dir}, merging into {output_dir}_merge")
    # Merge hparams
    if osp.isdir(existing_hparam_dir):
        for f in os.listdir(existing_hparam_dir):
            if f.endswith(".npy"):
                key = f.removesuffix(".npy")
                existing = np.load(osp.join(existing_hparam_dir, f), allow_pickle=True)
                if key in all_hparams:
                    all_hparams[key] = np.concatenate([existing, all_hparams[key]])
                else:
                    all_hparams[key] = existing
    # Merge metrics
    if osp.isdir(existing_metric_dir):
        for f in os.listdir(existing_metric_dir):
            if f.endswith(".npy"):
                key = f.removesuffix(".npy")
                existing = np.load(osp.join(existing_metric_dir, f), allow_pickle=True)
                if key == "eval_returns":
                    all_returns = np.concatenate([existing, all_returns])
                elif key in all_metrics:
                    all_metrics[key] = np.concatenate([existing, all_metrics[key]])
                else:
                    all_metrics[key] = existing
    # Write to a separate merge directory
    output_dir = output_dir + "_merge"

os.makedirs(osp.join(output_dir, "hparam"), exist_ok=True)
os.makedirs(osp.join(output_dir, "metric"), exist_ok=True)

for key, hparam in all_hparams.items():
    np.save(osp.join(output_dir, "hparam", f"{key}.npy"), hparam)

for key, metric in all_metrics.items():
    np.save(osp.join(output_dir, "metric", f"{key}.npy"), metric)

np.save(osp.join(output_dir, "metric", "eval_returns.npy"), all_returns)