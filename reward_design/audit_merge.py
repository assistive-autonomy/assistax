"""Merge per-seed audit recordings (<root>/<env>_s<k>/audit_<mode>.npz) into <out>/<env>/ with the seed axis stacked.

Usage: python -m reward_design.audit_merge <root> <out>
"""
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
for env in ("teethbrushing", "feeding", "scratchitch", "bedbathing", "armmanipulation"):
    seeds = sorted(p for p in root.glob(f"{env}_s*") if (p / "audit_human_zero.npz").exists())
    if not seeds:
        continue
    (out / env).mkdir(parents=True, exist_ok=True)
    for mode in ("sampled", "mean", "human_zero"):
        parts = [dict(np.load(s / f"audit_{mode}.npz")) for s in seeds]
        merged = {k: (parts[0][k] if k == "agents" else np.concatenate([p[k] for p in parts])) for k in parts[0]}
        np.savez(out / env / f"audit_{mode}.npz", **merged)
    print(env, "seeds:", [s.name for s in seeds])
