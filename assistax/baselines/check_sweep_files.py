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

def restructure_path(run_dir, root, subfolder):
    """
    Transform path like: root/env/algo/variant/date/time/...
    Into: root/env/algo/subfolder/variant/date/time/...
    
    Returns new path or None if transformation fails.
    """
    try:
        # Normalize paths
        run_dir = os.path.normpath(os.path.abspath(run_dir))
        root = os.path.normpath(os.path.abspath(root))
        
        # Get relative path from root
        rel_path = os.path.relpath(run_dir, root)
        parts = rel_path.split(os.sep)
        
        # Expected structure: algo/date/time/... (when root is env)
        # We want: algo/subfolder/date/time/...
        if len(parts) >= 1:
            # Insert subfolder after algo (position 1)
            new_parts = parts[:1] + [subfolder] + parts[1:]
            new_path = os.path.join(root, *new_parts)
            return new_path
        else:
            return None
    except Exception as e:
        print(f"Error restructuring path {run_dir}: {e}")
        return None

def main():
    p = argparse.ArgumentParser(description="Check returns.npy shapes and multirun.yaml recurrent settings")
    p.add_argument("root", nargs="?", default=".", help="Root folder containing run directories.")
    p.add_argument("--verbose", action="store_true", help="Print detailed information for each run.")
    p.add_argument("--restructure", action="store_true", 
                   help="Restructure directories: move valid FF runs to ff_nps/ and valid RNN runs to rnn_nps/")
    p.add_argument("--dry-run", action="store_true", 
                   help="Show what would be moved without actually moving files (use with --restructure)")
    p.add_argument("--copy", action="store_true",
                   help="Copy instead of move (preserves originals, use with --restructure)")
    args = p.parse_args()

    run_dirs = find_run_dirs(args.root)
    
    if not run_dirs:
        print("No run directories found (looking for directories with hparams.npy)")
        return

    print(f"Found {len(run_dirs)} run directories\n")

    # Track results - separate by shape validity AND recurrent setting
    rnn_valid_shape = []      # recurrent=True AND valid shape
    rnn_invalid_shape = []    # recurrent=True BUT invalid shape
    ff_valid_shape = []       # recurrent=False AND valid shape
    ff_invalid_shape = []     # recurrent=False BUT invalid shape
    yaml_missing = []         # no multirun.yaml found
    recurrent_unclear = []    # YAML parse error or unclear setting
    
    all_returns_valid = []
    all_returns_invalid = []

    for run_dir in sorted(run_dirs):
        returns_path = os.path.join(run_dir, "returns.npy")
        
        # Check returns.npy shape
        shape_valid = False
        shape_info = None
        if os.path.exists(returns_path):
            is_valid, shape, error = check_returns_shape(returns_path)
            shape_valid = is_valid
            shape_info = (shape, error)
            if is_valid:
                all_returns_valid.append(run_dir)
                if args.verbose:
                    print(f"✓ VALID SHAPE: {run_dir} | shape={shape}")
            else:
                all_returns_invalid.append((run_dir, shape, error))
                if args.verbose:
                    print(f"✗ INVALID SHAPE: {run_dir} | shape={shape} | {error}")
        else:
            shape_info = (None, "returns.npy not found")
            all_returns_invalid.append((run_dir, None, "returns.npy not found"))
            if args.verbose:
                print(f"✗ MISSING: {run_dir} | returns.npy not found")
        
        # Check multirun.yaml
        yaml_path, parent_dir = find_multirun_yaml(run_dir)
        
        if yaml_path is None:
            yaml_missing.append((run_dir, shape_valid, shape_info))
            if args.verbose:
                print(f"  ⚠ No multirun.yaml found in parent directories")
        else:
            recurrent_val, error = check_recurrent_setting(yaml_path)
            
            if error:
                recurrent_unclear.append((run_dir, parent_dir, error, shape_valid, shape_info))
                if args.verbose:
                    print(f"  ? YAML ISSUE: {yaml_path} | {error}")
            elif recurrent_val is True:
                if shape_valid:
                    rnn_valid_shape.append((run_dir, parent_dir, yaml_path))
                    if args.verbose:
                        print(f"  ✓ RNN + VALID: {parent_dir} | recurrent=True")
                else:
                    rnn_invalid_shape.append((run_dir, parent_dir, yaml_path, shape_info))
                    if args.verbose:
                        print(f"  ✗ RNN + INVALID: {parent_dir} | recurrent=True | {shape_info[1]}")
            elif recurrent_val is False:
                if shape_valid:
                    ff_valid_shape.append((run_dir, parent_dir, yaml_path))
                    if args.verbose:
                        print(f"  ✓ FF + VALID: {parent_dir} | recurrent=False")
                else:
                    ff_invalid_shape.append((run_dir, parent_dir, yaml_path, shape_info))
                    if args.verbose:
                        print(f"  ✗ FF + INVALID: {parent_dir} | recurrent=False | {shape_info[1]}")
            else:
                recurrent_unclear.append((run_dir, parent_dir, f"Unexpected value: {recurrent_val}", shape_valid, shape_info))
                if args.verbose:
                    print(f"  ? UNEXPECTED: recurrent={recurrent_val}")
        
        if args.verbose:
            print()

    # Restructure directories if requested
    if args.restructure:
        print("\n" + "="*80)
        print("=== RESTRUCTURING DIRECTORIES ===")
        print("="*80)
        
        if args.dry_run:
            print("DRY RUN MODE - No files will be moved/copied\n")
        
        operation = "COPY" if args.copy else "MOVE"
        moved_count = 0
        
        # Process FF runs (including those without multirun.yaml - treat as FF)
        ff_from_missing = [(run_dir, None, None) for run_dir, shape_valid, _ in yaml_missing if shape_valid]
        ff_runs_to_process = ff_valid_shape + ff_from_missing
        
        print(f"\nProcessing {len(ff_runs_to_process)} FF runs with valid shapes...")
        for run_dir, parent_dir, yaml_path in ff_runs_to_process:
            new_path = restructure_path(run_dir, args.root, "ff_nps")
            
            if new_path and new_path != run_dir:
                print(f"{operation}: {run_dir}")
                print(f"  → {new_path}")
                
                if not args.dry_run:
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    if args.copy:
                        shutil.copytree(run_dir, new_path)
                    else:
                        shutil.move(run_dir, new_path)
                moved_count += 1
        
        # Process RNN runs
        print(f"\nProcessing {len(rnn_valid_shape)} RNN runs with valid shapes...")
        for run_dir, parent_dir, yaml_path in rnn_valid_shape:
            new_path = restructure_path(run_dir, args.root, "rnn_nps")
            
            if new_path and new_path != run_dir:
                print(f"{operation}: {run_dir}")
                print(f"  → {new_path}")
                
                if not args.dry_run:
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    if args.copy:
                        shutil.copytree(run_dir, new_path)
                    else:
                        shutil.move(run_dir, new_path)
                moved_count += 1
        
        print(f"\n{operation} complete: {moved_count} directories processed")
        if args.dry_run:
            print("(No actual changes made - remove --dry-run to execute)")
        print("="*80 + "\n")

    # Print summary
    print("\n" + "="*80)
    print("=== SUMMARY ===")
    print("="*80)
    
    print(f"\n📊 OVERALL SHAPE CHECK:")
    print(f"  Valid (x, 6, 610): {len(all_returns_valid)}")
    print(f"  Invalid/Missing: {len(all_returns_invalid)}")
    
    print(f"\n🔄 RNN vs FF BREAKDOWN (by shape validity):")
    print(f"\n  ✅ RNN with VALID shape (recurrent=True): {len(rnn_valid_shape)}")
    print(f"  ❌ RNN with INVALID shape (recurrent=True): {len(rnn_invalid_shape)}")
    print(f"  ✅ FF with VALID shape (recurrent=False): {len(ff_valid_shape)}")
    print(f"  ❌ FF with INVALID shape (recurrent=False): {len(ff_invalid_shape)}")
    print(f"  ⚠️  No multirun.yaml found: {len(yaml_missing)}")
    print(f"  ❓ YAML unclear/error: {len(recurrent_unclear)}")
    
    # Show RNN with valid shape
    if rnn_valid_shape:
        print(f"\n✅ RNN Directories with VALID shape (recurrent=True + correct shape):")
        by_parent = defaultdict(list)
        for run_dir, parent_dir, yaml_path in rnn_valid_shape:
            by_parent[parent_dir].append(run_dir)
        
        for parent_dir in sorted(by_parent.keys()):
            print(f"    {parent_dir}/")
            for run_dir in by_parent[parent_dir][:5]:
                rel = os.path.relpath(run_dir, parent_dir)
                print(f"      ├─ {rel}")
            if len(by_parent[parent_dir]) > 5:
                print(f"      └─ ... and {len(by_parent[parent_dir]) - 5} more runs")
    
    # Show RNN with invalid shape
    if rnn_invalid_shape:
        print(f"\n❌ RNN Directories with INVALID shape (recurrent=True but wrong shape):")
        for run_dir, parent_dir, yaml_path, shape_info in rnn_invalid_shape[:20]:
            print(f"    {run_dir}")
            print(f"      → {shape_info[1]} | shape={shape_info[0]}")
        if len(rnn_invalid_shape) > 20:
            print(f"    ... and {len(rnn_invalid_shape) - 20} more")
    
    # Show FF with valid shape
    if ff_valid_shape:
        print(f"\n✅ FF Directories with VALID shape (recurrent=False + correct shape):")
        by_parent = defaultdict(list)
        for run_dir, parent_dir, yaml_path in ff_valid_shape:
            by_parent[parent_dir].append(run_dir)
        
        for parent_dir in sorted(by_parent.keys()):
            print(f"    {parent_dir}/")
            for run_dir in by_parent[parent_dir][:5]:
                rel = os.path.relpath(run_dir, parent_dir)
                print(f"      ├─ {rel}")
            if len(by_parent[parent_dir]) > 5:
                print(f"      └─ ... and {len(by_parent[parent_dir]) - 5} more runs")
    
    # Show FF with invalid shape
    if ff_invalid_shape:
        print(f"\n❌ FF Directories with INVALID shape (recurrent=False but wrong shape):")
        for run_dir, parent_dir, yaml_path, shape_info in ff_invalid_shape[:20]:
            print(f"    {run_dir}")
            print(f"      → {shape_info[1]} | shape={shape_info[0]}")
        if len(ff_invalid_shape) > 20:
            print(f"    ... and {len(ff_invalid_shape) - 20} more")
    
    # Show missing YAML
    if yaml_missing:
        print(f"\n⚠️  Directories without multirun.yaml (first 20):")
        for run_dir, shape_valid, shape_info in yaml_missing[:20]:
            status = "✓" if shape_valid else "✗"
            print(f"    {status} {run_dir}")
        if len(yaml_missing) > 20:
            print(f"    ... and {len(yaml_missing) - 20} more")
    
    # Show unclear YAML
    if recurrent_unclear:
        print(f"\n❓ Unclear/Error reading YAML (first 20):")
        for run_dir, parent_dir, error, shape_valid, shape_info in recurrent_unclear[:20]:
            status = "✓" if shape_valid else "✗"
            print(f"    {status} {run_dir}")
            print(f"      → {error}")
        if len(recurrent_unclear) > 20:
            print(f"    ... and {len(recurrent_unclear) - 20} more")
    
    # Final verdict
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    if len(rnn_valid_shape) == 0:
        print("⚠️  NO RNN directories with valid shape found")
    else:
        print(f"✓ Found {len(rnn_valid_shape)} RNN directories with valid shape (recurrent=True + correct shape)")
    
    if len(ff_valid_shape) == 0:
        print("⚠️  NO FF directories with valid shape found")
    else:
        print(f"✓ Found {len(ff_valid_shape)} FF directories with valid shape (recurrent=False + correct shape)")
    
    if len(rnn_invalid_shape) > 0:
        print(f"⚠️  {len(rnn_invalid_shape)} RNN directories have INVALID shapes")
    
    if len(ff_invalid_shape) > 0:
        print(f"⚠️  {len(ff_invalid_shape)} FF directories have INVALID shapes")
    
    print("="*80)

if __name__ == "__main__":
    main()