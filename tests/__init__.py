"""Regression tests for assistax's MuJoCo indexing (joint addresses, agent splits, frames, contacts).

Run from the repo root: ``python -m unittest discover -s tests -t . -v``.
"""
import os

# Must happen before jax is imported: keeps tests off a (possibly busy) GPU and avoids TF32 tolerance noise.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
