"""Train and/or evaluate IPPO on any assistax version and record what each agent does, for the human-involvement
audit (human_audit.py). Standalone on purpose: it imports nothing else from reward_design, so it also runs on the
pre-0.1.0 code (058d2f7), whose envs lack what tasks.py relies on.

Each evaluation records, per step: qpos/qvel, dones, every agent's reward, every agent's action as applied (sampled
or mean) and its policy mean, and the scratch itch (scratch itch only). Three modes, all on the same 32 starts per
seed as ippo_run.py: sampled actions, policy means, and sampled with the human's actions set to zero.

Usage, from the checkout whose assistax should run:
  python /path/to/reward_design/audit_run.py train <env> <out_dir> [--steps 4e7] [--seeds 1] [--seed 0]
  python /path/to/reward_design/audit_run.py eval <env> <out_dir> --params <dir with robot/human.safetensors>
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "  # as ippo_run.py

import distrax  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import safetensors.flax  # noqa: E402
from flax.traverse_util import flatten_dict, unflatten_dict  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import assistax  # noqa: E402
from assistax.baselines.utils import _unstack_tree  # noqa: E402

IPPO_DIR = Path(assistax.__file__).parent / "baselines" / "IPPO"
sys.path.insert(0, str(IPPO_DIR))
from ippo_ff_nps import MultiActorCritic, batchify, make_evaluation, make_train, unbatchify  # noqa: E402

MODES = ("sampled", "mean", "human_zero")


def load_config(env_name: str, overrides: list[str]) -> dict:
    """Compose ippo.yaml as `python ippo_run.py ENV_NAME=<env> ...` would, and resolve it."""
    with initialize_config_dir(config_dir=str(IPPO_DIR / "config"), version_base=None):
        cfg = compose("ippo", overrides=[f"ENV_NAME={env_name}", *overrides])
    return OmegaConf.to_container(cfg, resolve=True)


def load_params(run_dir: Path, agents: list[str], seed: int) -> dict:
    """Network params of one training seed: each agent's saved params, stacked on the agent axis."""
    per_agent = []
    for agent in agents:
        flat = safetensors.flax.load_file(str(run_dir / f"{agent}.safetensors"))
        per_agent.append(unflatten_dict({tuple(k.split("/")): jnp.asarray(v[seed]) for k, v in flat.items()}))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *per_agent)


def make_rollout(eval_env, apply_fn, n_episodes: int, mode: str):
    """One evaluation of n_episodes in `mode`, recording states, rewards and both agents' actions."""
    human = eval_env.agents.index("human")

    def rollout(rng: jax.Array, params) -> dict:
        rng_reset, rng = jax.random.split(rng)
        obs, env_state = jax.vmap(eval_env.reset)(jax.random.split(rng_reset, n_episodes))
        last_done = jnp.zeros((eval_env.num_agents, n_episodes), dtype=bool)

        def step(carry, _):
            obs, env_state, last_done, rng = carry
            obs_batch = batchify(obs, eval_env.agents)
            avail = batchify(jax.vmap(eval_env.get_avail_actions)(env_state.env_state), eval_env.agents)
            (mean, std), _ = apply_fn(params, (obs_batch, last_done, avail))
            rng, k_act, k_step = jax.random.split(rng, 3)
            action = distrax.MultivariateNormalDiag(mean, jnp.expand_dims(std, axis=1)).sample(seed=k_act)
            if mode == "mean":
                action = mean
            if mode == "human_zero":
                action = action.at[human].set(0.0)
            obs, env_state, reward, done, _ = jax.vmap(eval_env.step)(
                jax.random.split(k_step, n_episodes), env_state, unbatchify(action, eval_env.agents))
            inner = env_state.env_state
            rec = {"qpos": inner.pipeline_state.qpos, "qvel": inner.pipeline_state.qvel, "done": done["__all__"],
                   "reward": jnp.stack([reward[a] for a in eval_env.agents]), "action": action, "mean": mean}
            if "scratch" in inner.info:
                rec["itch_arm"], rec["itch_pos"] = inner.info["scratch"]["arm"], inner.info["scratch"]["pos"]
            return (obs, env_state, batchify(done, eval_env.agents), rng), rec

        _, recs = jax.lax.scan(step, (obs, env_state, last_done, rng), None, eval_env.episode_length)
        return recs

    return rollout


def evaluate(config: dict, params_dir: Path, out_dir: Path) -> None:
    """All three modes for every training seed in params_dir; writes audit_<mode>.npz (seed axis first)."""
    eval_env, _ = make_evaluation(config)
    apply_fn = MultiActorCritic(config=config).apply
    agents = list(eval_env.agents)
    n_seeds = next(iter(safetensors.flax.load_file(str(params_dir / "robot.safetensors")).values())).shape[0]
    params = [load_params(params_dir, agents, s) for s in range(n_seeds)]
    _, eval_rng = jax.random.split(jax.random.key(config["SEED"]))
    for mode in MODES:
        t0 = time.time()
        rollout = jax.jit(make_rollout(eval_env, apply_fn, config["NUM_EVAL_EPISODES"], mode))
        recs = [jax.device_get(rollout(eval_rng, p)) for p in params]
        np.savez(out_dir / f"audit_{mode}.npz", agents=np.array(agents),
                 **{k: np.stack([np.asarray(r[k]) for r in recs]) for k in recs[0]})
        print(f"[{config['ENV_NAME']}] {mode}: {time.time() - t0:.0f} s", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval"])
    p.add_argument("env")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--steps", type=float, default=None)
    p.add_argument("--seeds", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--params", type=Path, default=None)
    p.add_argument("--version", default=None, help="reward_design.tasks version to install (0.1.0 code only)")
    a = p.parse_args()
    if a.version is not None:  # imported only on request, so the file still runs on pre-0.1.0 code
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from reward_design import tasks
        tasks.install(a.env, a.version)
    overrides = ([f"TOTAL_TIMESTEPS={a.steps}"] * (a.steps is not None) + [f"NUM_SEEDS={a.seeds}"] * (a.seeds is not None)
                 + [f"SEED={a.seed}"] * (a.seed is not None))
    config = load_config(a.env, overrides)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "config.json").write_text(json.dumps(
        {**config, "ASSISTAX_FILE": assistax.__file__, "REWARD_VERSION": a.version}, indent=1))
    params_dir = a.params
    if a.mode == "train":
        train_rng, _ = jax.random.split(jax.random.key(config["SEED"]))
        t0 = time.time()
        train_jit = jax.jit(make_train(config, save_train_state=False), device=jax.devices()[config["DEVICE"]])
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            jax.random.split(train_rng, config["NUM_SEEDS"]), config["LR"], config["ENT_COEF"], config["CLIP_EPS"])
        jax.block_until_ready(out)
        print(f"[{a.env}] trained in {time.time() - t0:.0f} s", flush=True)
        np.savez(a.out_dir / "train_metrics.npz", **{k: np.asarray(v) for k, v in out["metrics"].items()})
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        params = out["runner_state"].train_state.params
        for agent, ap in zip(env.agents, _unstack_tree(jax.tree.map(lambda x: x.swapaxes(0, 1), params))):
            safetensors.flax.save_file(flatten_dict(ap, sep="/"), str(a.out_dir / f"{agent}.safetensors"))
        params_dir = a.out_dir
    evaluate(config, params_dir, a.out_dir)


if __name__ == "__main__":
    main()
