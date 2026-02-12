import os
import argparse
from collections import defaultdict
import numpy as np
import yaml

REQUIRED = ("hparams.npy", "metrics.npy", "returns.npy")

def is_hidden_or_dot(path_part: str) -> bool:
    return path_part.startswith(".") or path_part == "__pycache__"

def find_run_dirs(root):
    """
    A 'run dir' = directory that contains hparams.npy.
    """
    run_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune hidden dirs
        dirnames[:] = [d for d in dirnames if not is_hidden_or_dot(d)]
        if "hparams.npy" in filenames:
            run_dirs.append(dirpath)
    return run_dirs

def check_returns_shape(returns_path):
    """
    Check if returns.npy has shape (x, 6, 610).
    Returns: (is_valid, actual_shape, error_message)
    """
    try:
        data = np.load(returns_path, allow_pickle=True)
        shape = data.shape
        
        if len(shape) != 3:
            return False, shape, f"Expected 3 dimensions, got {len(shape)}"
        
        if shape[1] != 6:
            return False, shape, f"Dimension 1 should be 6, got {shape[1]}"
        
        if shape[2] != 610:
            return False, shape, f"Dimension 2 should be 610, got {shape[2]}"
        
        return True, shape, None
    except Exception as e:
        return False, None, f"Error loading: {str(e)}"

def find_multirun_yaml(run_dir):
    """
    Search upward from run_dir to find multirun.yaml in parent date/time directories.
    Returns: (yaml_path, parent_dir) or (None, None)
    """
    current = run_dir
    # Go up a few levels to find date/time directories
    for _ in range(5):  # limit search depth
        parent = os.path.dirname(current)
        if parent == current:  # reached root
            break
        
        yaml_path = os.path.join(current, "multirun.yaml")
        if os.path.isfile(yaml_path):
            return yaml_path, current
        
        current = parent
    
    return None, None

def check_recurrent_setting(yaml_path):
    """
    Parse multirun.yaml and extract network.recurrent setting.
    Returns: (recurrent_value, error_message)
    """
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            return None, "Empty YAML file"
        
        # Check for network.recurrent
        if 'network' in data and isinstance(data['network'], dict):
            if 'recurrent' in data['network']:
                return data['network']['recurrent'], None
        
        return None, "network.recurrent key not found"
    except Exception as e:
        return None, f"Error parsing YAML: {str(e)}"

def main():
    p = argparse.ArgumentParser(description="Check returns.npy shapes and multirun.yaml recurrent settings")
    p.add_argument("root", nargs="?", default=".", help="Root folder containing run directories.")
    p.add_argument("--verbose", action="store_true", help="Print detailed information for each run.")
    args = p.parse_args()

    run_dirs = find_run_dirs(args.root)
    
    if not run_dirs:
        print("No run directories found (looking for directories with hparams.npy)")
        return

    print(f"Found {len(run_dirs)} run directories\n")

    # Track results
    returns_valid = []
    returns_invalid = []
    yaml_missing = set()  # parent directories without multirun.yaml
    recurrent_true = []
    recurrent_false = []
    recurrent_unclear = []

    for run_dir in sorted(run_dirs):
        returns_path = os.path.join(run_dir, "returns.npy")
        
        # Check returns.npy shape
        if os.path.exists(returns_path):
            is_valid, shape, error = check_returns_shape(returns_path)
            if is_valid:
                returns_valid.append(run_dir)
                if args.verbose:
                    print(f"✓ VALID SHAPE: {run_dir} | shape={shape}")
            else:
                returns_invalid.append((run_dir, shape, error))
                if args.verbose:
                    print(f"✗ INVALID SHAPE: {run_dir} | shape={shape} | {error}")
        else:
            returns_invalid.append((run_dir, None, "returns.npy not found"))
            if args.verbose:
                print(f"✗ MISSING: {run_dir} | returns.npy not found")
        
        # Check multirun.yaml
        yaml_path, parent_dir = find_multirun_yaml(run_dir)
        
        if yaml_path is None:
            yaml_missing.add(run_dir)
            if args.verbose:
                print(f"  ⚠ No multirun.yaml found in parent directories")
        else:
            recurrent_val, error = check_recurrent_setting(yaml_path)
            
            if error:
                recurrent_unclear.append((run_dir, parent_dir, error))
                if args.verbose:
                    print(f"  ? YAML ISSUE: {yaml_path} | {error}")
            elif recurrent_val is True:
                recurrent_true.append((run_dir, parent_dir, yaml_path))
                if args.verbose:
                    print(f"  RNN: {parent_dir} | recurrent=True")
            elif recurrent_val is False:
                recurrent_false.append((run_dir, parent_dir, yaml_path))
                if args.verbose:
                    print(f"  FF: {parent_dir} | recurrent=False")
            else:
                recurrent_unclear.append((run_dir, parent_dir, f"Unexpected value: {recurrent_val}"))
                if args.verbose:
                    print(f"  ? UNEXPECTED: recurrent={recurrent_val}")
        
        if args.verbose:
            print()

    # Print summary
    print("\n" + "="*80)
    print("=== SUMMARY ===")
    print("="*80)
    
    print(f"\n📊 RETURNS.NPY SHAPE CHECK:")
    print(f"  Valid (x, 6, 610): {len(returns_valid)}")
    print(f"  Invalid/Missing: {len(returns_invalid)}")
    
    if returns_invalid:
        print(f"\n  Invalid/Missing runs (first 20):")
        for run_dir, shape, error in returns_invalid[:20]:
            print(f"    {run_dir}")
            print(f"      → {error} | shape={shape}")
        if len(returns_invalid) > 20:
            print(f"    ... and {len(returns_invalid) - 20} more")
    
    print(f"\n🔄 MULTIRUN.YAML RECURRENT SETTINGS:")
    print(f"  Recurrent=True (RNN): {len(recurrent_true)}")
    print(f"  Recurrent=False (FF): {len(recurrent_false)}")
    print(f"  Unclear/Error: {len(recurrent_unclear)}")
    print(f"  No multirun.yaml found: {len(yaml_missing)}")
    
    if recurrent_true:
        print(f"\n  ✓ RNN Directories (recurrent=True):")
        # Group by parent directory
        by_parent = defaultdict(list)
        for run_dir, parent_dir, yaml_path in recurrent_true:
            by_parent[parent_dir].append(run_dir)
        
        for parent_dir in sorted(by_parent.keys()):
            print(f"    {parent_dir}/")
            for run_dir in by_parent[parent_dir][:5]:
                rel = os.path.relpath(run_dir, parent_dir)
                print(f"      ├─ {rel}")
            if len(by_parent[parent_dir]) > 5:
                print(f"      └─ ... and {len(by_parent[parent_dir]) - 5} more runs")
    
    if recurrent_false:
        print(f"\n  ✓ FF Directories (recurrent=False):")
        by_parent = defaultdict(list)
        for run_dir, parent_dir, yaml_path in recurrent_false:
            by_parent[parent_dir].append(run_dir)
        
        for parent_dir in sorted(by_parent.keys()):
            print(f"    {parent_dir}/")
            for run_dir in by_parent[parent_dir][:5]:
                rel = os.path.relpath(run_dir, parent_dir)
                print(f"      ├─ {rel}")
            if len(by_parent[parent_dir]) > 5:
                print(f"      └─ ... and {len(by_parent[parent_dir]) - 5} more runs")
    
    if yaml_missing:
        print(f"\n  ⚠ Directories without multirun.yaml (first 20):")
        for run_dir in sorted(yaml_missing)[:20]:
            print(f"    {run_dir}")
        if len(yaml_missing) > 20:
            print(f"    ... and {len(yaml_missing) - 20} more")
    
    if recurrent_unclear:
        print(f"\n  ? Unclear/Error reading YAML (first 20):")
        for run_dir, parent_dir, error in recurrent_unclear[:20]:
            print(f"    {run_dir}")
            print(f"      → {error}")
        if len(recurrent_unclear) > 20:
            print(f"    ... and {len(recurrent_unclear) - 20} more")
    
    # Final verdict
    print("\n" + "="*80)
    if len(recurrent_true) == 0:
        print("⚠️  NO DIRECTORIES WITH recurrent=True FOUND")
    else:
        print(f"✓ Found {len(recurrent_true)} directories with recurrent=True")
    
    if len(recurrent_false) > 0:
        print(f"✓ Found {len(recurrent_false)} directories with recurrent=False")
    
    print("="*80)

if __name__ == "__main__":
    main()