"""Benchmark steps-per-second (SPS) for all environments with random actions.

Measures how throughput scales with the number of parallel environments.
Outputs results to benchmark_sps.csv.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import csv
import time
from typing import List

import jax
import jax.numpy as jnp

import assistax

ENV_NAMES: List[str] = [
    "scratchitch",
    "bedbathing",
    "armmanipulation",
    "feeding",
    "teethbrushing",
]
NUM_ENVS_LIST: List[int] = [128, 256, 512, 1024, 2048, 4096]
NUM_TRIALS: int = 16
TOTAL_STEPS: int = 1_000_000


def benchmark_env(env_name: str, num_envs: int) -> tuple[float, float]:
    """Benchmark a single (env, num_envs) combination.

    Returns mean and std of SPS across trials.
    """
    env = assistax.envs.create(env_name, batch_size=num_envs)
    rng = jax.random.PRNGKey(0)
    num_step_calls = TOTAL_STEPS // num_envs

    # Reset
    rng, reset_key = jax.random.split(rng)
    state = env.reset(reset_key)

    @jax.jit
    def run_n_steps(state, rng):
        """Run num_step_calls env steps via lax.scan."""
        def step_fn(carry, _):
            state, rng = carry
            rng, act_key, step_key = jax.random.split(rng, 3)
            action = jax.random.uniform(act_key, (num_envs, env.action_size), minval=-1.0, maxval=1.0)
            step_keys = jax.random.split(step_key, num_envs)
            state = env.step(step_keys, state, action)
            return (state, rng), None
        (state, _), _ = jax.lax.scan(step_fn, (state, rng), None, length=num_step_calls)
        return state

    # JIT warmup (compiles the full scan)
    rng, warmup_key = jax.random.split(rng)
    state = run_n_steps(state, warmup_key)
    jax.block_until_ready(state)

    sps_values = []
    for trial in range(NUM_TRIALS):
        rng, trial_key = jax.random.split(rng)
        t_start = time.time()
        state = run_n_steps(state, trial_key)
        jax.block_until_ready(state)
        elapsed = time.time() - t_start
        sps = num_step_calls * num_envs / elapsed
        sps_values.append(sps)

    mean_sps = float(jnp.mean(jnp.array(sps_values)))
    std_sps = float(jnp.std(jnp.array(sps_values)))
    return mean_sps, std_sps


def main() -> None:
    """Run the benchmark across all environments and num_envs configurations."""
    output_path = os.path.join(os.path.dirname(__file__), "benchmark_sps.csv")
    results = []

    for env_name in ENV_NAMES:
        for num_envs in NUM_ENVS_LIST:
            print(f"Benchmarking {env_name} | num_envs={num_envs} ...", flush=True)
            mean_sps, std_sps = benchmark_env(env_name, num_envs)
            print(f"  -> SPS: {mean_sps:.0f} +/- {std_sps:.0f}")
            results.append({
                "env_name": env_name,
                "num_envs": num_envs,
                "mean_sps": mean_sps,
                "std_sps": std_sps,
            })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["env_name", "num_envs", "mean_sps", "std_sps"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
