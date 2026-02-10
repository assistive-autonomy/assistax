import os
import shutil
import argparse
from collections import defaultdict
import numpy as np

REQUIRED = ("hparams.npy", "metrics.npy", "returns.npy")

def is_hidden_or_dot(path_part: str) -> bool:
    return path_part.startswith(".") or path_part == "__pycache__"

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_hparams(hparams_path):
    """
    Loads a dict from hparams.npy.
    Works for np.save(dict) and for pickled objects.
    """
    try:
        obj = np.load(hparams_path, allow_pickle=True)
        # common patterns:
        # - saved dict -> array( dict, dtype=object )
        # - already a python object
        if hasattr(obj, "item"):
            return obj.item()
        return obj
    except Exception as e:
        return {"__LOAD_ERROR__": str(e)}

def has_required_files(run_dir):
    return all(os.path.isfile(os.path.join(run_dir, f)) for f in REQUIRED)

def find_run_dirs(root):
    """
    A 'run dir' = directory that contains hparams.npy (optionally also metrics/returns).
    """
    run_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # prune hidden dirs
        dirnames[:] = [d for d in dirnames if not is_hidden_or_dot(d)]
        if "hparams.npy" in filenames:
            run_dirs.append(dirpath)
    return run_dirs

def detect_total_timesteps(hparams: dict):
    # common key names; extend if needed
    for k in ("TOTAL_TIMESTEPS", "total_timesteps", "num_timesteps", "timesteps"):
        if k in hparams:
            v = hparams[k]
            # handle numpy scalars / arrays
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            f = safe_float(v)
            if f is not None:
                return f, k
    return None, None

def detect_rnn(hparams: dict):
    """
    Heuristics for 'RNN run' vs 'FF run'.
    You should tweak based on what keys your code writes.

    Returns: (is_rnn: bool or None, reason: str)
    """
    # Direct flags
    for k in ("use_rnn", "USE_RNN", "recurrent", "is_recurrent", "use_lstm", "use_gru"):
        if k in hparams:
            v = hparams[k]
            if isinstance(v, np.ndarray) and v.size == 1:
                v = v.item()
            if isinstance(v, (bool, np.bool_)):
                return bool(v), f"{k}={v}"
            # sometimes 0/1
            if safe_float(v) is not None:
                return (safe_float(v) != 0.0), f"{k}={v}"

    # Architecture hints
    for k in ("network", "model", "policy_network", "agent_network", "encoder_type"):
        if k in hparams:
            v = str(hparams[k]).lower()
            if any(s in v for s in ("rnn", "lstm", "gru", "recurrent")):
                return True, f"{k} contains {v}"
            if any(s in v for s in ("mlp", "ff", "feedforward")):
                return False, f"{k} contains {v}"

    # Common RNN hyperparams presence
    rnn_keys = ("rnn_hidden_size", "lstm_hidden_size", "gru_hidden_size", "recurrent_layers", "rnn_layers")
    if any(k in hparams for k in rnn_keys):
        present = [k for k in rnn_keys if k in hparams]
        return True, f"found {present}"

    # Common FF hints
    ff_keys = ("mlp_hidden_sizes", "hidden_sizes", "fc_layers", "num_layers")
    if any(k in hparams for k in ff_keys):
        present = [k for k in ff_keys if k in hparams]
        return False, f"found {present}"

    return None, "no clear rnn/ff signal"

def approx_equal(a, b, rel=1e-6, abs_=1.0):
    # good for comparing 4e7 vs 40000000 etc.
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))

def delete_empty_dirs(root, dry_run=True):
    removed = []
    # bottom-up so children removed first
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        # ignore hidden
        if any(is_hidden_or_dot(p) for p in dirpath.split(os.sep) if p):
            continue
        if not dirnames and not filenames:
            removed.append(dirpath)
            if not dry_run:
                os.rmdir(dirpath)
    return removed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=".", help="Root folder containing env/algo/... directories.")
    p.add_argument("--dry-run", action="store_true", help="Print actions but don't delete/move anything.")
    p.add_argument("--delete-empty-dirs", action="store_true", help="Delete empty directories.")
    p.add_argument("--require-files", action="store_true",
                   help="Only consider run dirs that contain hparams.npy AND metrics.npy AND returns.npy.")
    p.add_argument("--target-timesteps", type=float, default=4e7, help="Expected TOTAL_TIMESTEPS.")
    p.add_argument("--delete-nonmatching", action="store_true",
                   help="Delete runs that don't match target timesteps (and optionally rnn/ff filters).")
    p.add_argument("--move-nonmatching-to", default=None,
                   help="Instead of deleting nonmatching runs, move them to this folder (keeps structure).")
    p.add_argument("--require-rnn", action="store_true", help="Keep only runs detected as RNN.")
    p.add_argument("--require-ff", action="store_true", help="Keep only runs detected as FF.")
    args = p.parse_args()

    if args.require_rnn and args.require_ff:
        raise SystemExit("Choose at most one of --require-rnn or --require-ff")

    run_dirs = find_run_dirs(args.root)

    summary = defaultdict(int)
    problems = []

    def env_algo_variant_from_path(path):
        # expects ./env/algo/variant/...
        parts = os.path.normpath(path).split(os.sep)
        # find first three non-dot segments after root "."
        # if root is not ".", we still just take last 3 meaningful segments before date/time
        # simplest: assume structure env/algo/variant appears early; take parts[1:4] when root="."
        if parts and parts[0] == ".":
            parts = parts[1:]
        env = parts[0] if len(parts) > 0 else "?"
        algo = parts[1] if len(parts) > 1 else "?"
        variant = parts[2] if len(parts) > 2 else "?"
        return env, algo, variant

    # Audit
    for d in sorted(run_dirs):
        if args.require_files and not has_required_files(d):
            summary["skipped_missing_required_files"] += 1
            continue

        hp = load_hparams(os.path.join(d, "hparams.npy"))
        if "__LOAD_ERROR__" in hp:
            problems.append((d, "hparams_load_error", hp["__LOAD_ERROR__"]))
            summary["hparams_load_error"] += 1
            continue

        ts, ts_key = detect_total_timesteps(hp)
        is_rnn, rnn_reason = detect_rnn(hp)

        env, algo, variant = env_algo_variant_from_path(d)
        base_key = f"{env}/{algo}/{variant}"

        matches_ts = (ts is not None) and approx_equal(ts, args.target_timesteps)
        matches_arch = True
        if args.require_rnn:
            matches_arch = (is_rnn is True)
        if args.require_ff:
            matches_arch = (is_rnn is False)

        if matches_ts and matches_arch:
            summary[f"KEEP {base_key}"] += 1
        else:
            why = []
            if ts is None:
                why.append("no_TOTAL_TIMESTEPS_key")
            elif not matches_ts:
                why.append(f"{ts_key}={ts}")
            if args.require_rnn and is_rnn is not True:
                why.append(f"not_rnn ({rnn_reason})")
            if args.require_ff and is_rnn is not False:
                why.append(f"not_ff ({rnn_reason})")
            problems.append((d, "NONMATCH", "; ".join(why)))
            summary[f"NONMATCH {base_key}"] += 1

            # Apply action
            if args.delete_nonmatching or args.move_nonmatching_to:
                if args.move_nonmatching_to:
                    dest_root = args.move_nonmatching_to
                    os.makedirs(dest_root, exist_ok=True)
                    # preserve relative path
                    rel = os.path.relpath(d, args.root)
                    dest = os.path.join(dest_root, rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    print(f"MOVE: {d} -> {dest}")
                    if not args.dry_run:
                        shutil.move(d, dest)
                else:
                    print(f"DELETE RUN DIR: {d}")
                    if not args.dry_run:
                        shutil.rmtree(d)

    # Print summary
    print("\n=== SUMMARY ===")
    for k in sorted(summary.keys()):
        print(f"{k}: {summary[k]}")

    # Print top problems
    if problems:
        print("\n=== NONMATCH / PROBLEMS (first 50) ===")
        for row in problems[:50]:
            print(f"{row[1]} | {row[0]} | {row[2]}")

    # Delete empty directories (optional)
    if args.delete_empty_dirs:
        removed = delete_empty_dirs(args.root, dry_run=args.dry_run)
        print(f"\nEmpty directories {'(dry-run) ' if args.dry_run else ''}to remove: {len(removed)}")
        for d in removed[:50]:
            print(f"RMDIR: {d}")
        if len(removed) > 50:
            print("... (more omitted)")

if __name__ == "__main__":
    main()