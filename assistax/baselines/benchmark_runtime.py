"""
Wall-clock runtime benchmark for IPPO (ff_nps) across all Assistax environments.

Runs 5 PPO updates per trial, repeats 12 timed trials (plus 1 warmup for JIT),
and extrapolates to the full 610-update training run.

Usage:
    uv run python assistax/baselines/benchmark_runtime.py
"""

import time
import csv
import jax
import jax.numpy as jnp
import numpy as np

from assistax.baselines.IPPO.ippo_ff_nps import make_train

ENV_NAMES = [
    "scratchitch",
    "bedbathing",
    "armmanipulation",
    "feeding",
    "teethbrushing",
]

NUM_STEPS = 64
NUM_ENVS = 1024
BENCH_UPDATES = 5
FULL_UPDATES = 610  # 4e7 // 64 // 1024
NUM_TRIALS = 12
OUTPUT_CSV = "benchmark_runtime.csv"


def build_config(env_name: str) -> dict:
    """Build a minimal config dict for IPPO ff_nps benchmarking."""
    het_reward = env_name == "pushcoop"
    return {
        "ENV_NAME": env_name,
        "ALG": "IPPO",
        "ENV_KWARGS": {
            "ctrl_cost_weight": 0,
            "homogenisation_method": "max",
            "backend": "mjx",
            "het_reward": het_reward,
            "episode_length": 1000,
        },
        "TOTAL_TIMESTEPS": BENCH_UPDATES * NUM_STEPS * NUM_ENVS,
        "NUM_STEPS": NUM_STEPS,
        "NUM_ENVS": NUM_ENVS,
        "NUM_SEEDS": 1,
        "SEED": 0,
        "UPDATE_EPOCHS": 16,
        "NUM_MINIBATCHES": 16,
        "ANNEAL_LR": False,
        "LR": 1.37e-4,
        "ENT_COEF": 0.0017260804,
        "CLIP_EPS": 0.225,
        "SCALE_CLIP_EPS": False,
        "RATIO_CLIP_EPS": False,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "ADAM_EPS": 1e-8,
        "GPU_ENV_CAPACITY": 8192,
        "ADVANTAGE_UNROLL_DEPTH": 8,
        "DISABLE_JIT": False,
        "DEVICE": 0,
        "network": {
            "recurrent": False,
            "agent_param_sharing": False,
            "actor_hidden_dim": 128,
            "critic_hidden_dim": 128,
            "activation": "relu",
        },
    }


def benchmark_env(env_name: str) -> dict:
    """Run timed trials for a single environment and return stats."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {env_name}")
    print(f"{'='*60}")

    config = build_config(env_name)
    train_fn = make_train(config)
    train_jit = jax.jit(train_fn)

    lr = jnp.float32(config["LR"])
    ent_coef = jnp.float32(config["ENT_COEF"])
    clip_eps = jnp.float32(config["CLIP_EPS"])

    # Warmup (absorbs JIT compilation)
    print("  Warmup (JIT compile)...")
    rng = jax.random.PRNGKey(0)
    out = train_jit(rng, lr, ent_coef, clip_eps)
    jax.block_until_ready(out)
    print("  Warmup done.")

    # Timed trials
    times = []
    for i in range(NUM_TRIALS):
        rng = jax.random.PRNGKey(i + 1)
        t0 = time.time()
        out = train_jit(rng, lr, ent_coef, clip_eps)
        jax.block_until_ready(out)
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"  Trial {i+1:2d}/{NUM_TRIALS}: {elapsed:.3f}s")

    mean_t = float(np.mean(times))
    std_t = float(np.std(times))
    est_full_s = mean_t * (FULL_UPDATES / BENCH_UPDATES)
    est_full_min = est_full_s / 60.0

    print(f"  Mean: {mean_t:.3f}s  Std: {std_t:.3f}s")
    print(f"  Estimated full training ({FULL_UPDATES} updates): {est_full_s:.1f}s ({est_full_min:.2f} min)")

    return {
        "env_name": env_name,
        "mean_5updates_s": round(mean_t, 4),
        "std_5updates_s": round(std_t, 4),
        "est_full_runtime_s": round(est_full_s, 2),
        "est_full_runtime_min": round(est_full_min, 2),
    }


def main() -> None:
    """Benchmark all environments and write results to CSV."""
    print(f"JAX devices: {jax.devices()}")
    print(f"Benchmark: {BENCH_UPDATES} updates, {NUM_TRIALS} trials, extrapolate to {FULL_UPDATES} updates")

    results = []
    for env_name in ENV_NAMES:
        row = benchmark_env(env_name)
        results.append(row)

    # Write CSV
    fieldnames = ["env_name", "mean_5updates_s", "std_5updates_s", "est_full_runtime_s", "est_full_runtime_min"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults written to {OUTPUT_CSV}")
    print("\nSummary:")
    print(f"{'Environment':<20} {'Mean (s)':<12} {'Std (s)':<12} {'Est Full (min)':<16}")
    print("-" * 60)
    for r in results:
        print(f"{r['env_name']:<20} {r['mean_5updates_s']:<12} {r['std_5updates_s']:<12} {r['est_full_runtime_min']:<16}")


if __name__ == "__main__":
    main()
