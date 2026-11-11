"""Training scenarios vs fresh ones: replay one seed's final policy from the exact starts it trained on, and from new
random starts, and compare returns and the human's behaviour.

The env's AutoResetWrapper restores each env's first state (pose and, in scratch itch, the itch in ``info``) after
every episode, so a seed trains on only NUM_ENVS fixed scenarios. Training scenario i is env.reset(k_i), with k_i
rebuilt from the key chain in ippo_ff_nps.make_train (seed split, network init split, reset split).

Usage, from the worktree root:
  python -m reward_design.scenario_eval <run dir from audit_run.py> [--n 256] [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

import distrax
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reward_design import audit_run, tasks  # noqa: E402


def training_reset_keys(config: dict) -> jax.Array:
    """The NUM_ENVS reset keys of training seed 0 of a run (audit_run.py train with --seeds 1 --seed SEED)."""
    train_rng, _ = jax.random.split(jax.random.key(config["SEED"]))
    rng = jax.random.split(train_rng, config["NUM_SEEDS"])[0]
    rng, _ = jax.random.split(rng)  # network init
    _, rng_reset = jax.random.split(rng)
    return jax.random.split(rng_reset, config["NUM_ENVS"])


def make_episode(env, apply_fn, qadr: np.ndarray):
    """One episode per reset key with sampled actions: shared return and the human's mean joint angles after 2 s."""
    def episode(keys: jax.Array, params, rng: jax.Array) -> dict:
        obs, state = jax.vmap(env.reset)(keys)
        done = jnp.zeros((env.num_agents, keys.shape[0]), dtype=bool)

        def step(carry, t):
            obs, state, done, rng, ret, ang = carry
            ob = audit_run.batchify(obs, env.agents)
            avail = audit_run.batchify(jax.vmap(env.get_avail_actions)(state.env_state), env.agents)
            (mean, std), _ = apply_fn(params, (ob, done, avail))
            rng, k_act, k_step = jax.random.split(rng, 3)
            act = distrax.MultivariateNormalDiag(mean, jnp.expand_dims(std, axis=1)).sample(seed=k_act)
            obs, state, rew, d, _ = jax.vmap(env.step)(jax.random.split(k_step, keys.shape[0]), state,
                                                      audit_run.unbatchify(act, env.agents))
            q = state.env_state.pipeline_state.qpos[:, qadr]
            return (obs, state, audit_run.batchify(d, env.agents), rng, ret + rew[env.agents[0]],
                    ang + q * (t >= 250)), None

        init = (obs, state, done, rng, jnp.zeros(keys.shape[0]), jnp.zeros((keys.shape[0], len(qadr))))
        (_, _, _, _, ret, ang), _ = jax.lax.scan(step, init, jnp.arange(env.episode_length - 1))
        return {"return": ret, "angles_deg": jnp.degrees(ang / (env.episode_length - 251))}
    return episode


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--json", type=Path, default=None)
    a = p.parse_args()
    config = {k: v for k, v in json.loads((a.run / "config.json").read_text()).items()
              if k not in ("ASSISTAX_FILE", "REWARD_VERSION")}
    version = json.loads((a.run / "config.json").read_text())["REWARD_VERSION"]
    if version:
        tasks.install(config["ENV_NAME"], version)
    env, _ = audit_run.make_evaluation(config)
    apply_fn = audit_run.MultiActorCritic(config=config).apply
    params = audit_run.load_params(a.run, list(env.agents), 0)
    scene = mujoco.MjModel.from_xml_path(str(Path(audit_run.assistax.__file__).parent / "envs/assets/wheelchair_scene.xml"))
    names = ("right_shoulder1", "right_shoulder2", "right_elbow")
    qadr = np.array([scene.jnt_qposadr[scene.joint(n).id] for n in names])
    episode = jax.jit(make_episode(env, apply_fn, qadr))
    out = {}
    fresh = jax.random.split(jax.random.key(12345), a.n)
    for name, keys in (("training", training_reset_keys(config)[:a.n]), ("fresh", fresh)):
        r = jax.device_get(episode(keys, params, jax.random.key(7)))
        out[name] = {"return_mean": float(r["return"].mean()), "return_p10": float(np.percentile(r["return"], 10)),
                     "return_p90": float(np.percentile(r["return"], 90)),
                     **{f"{n.replace('right_', '')}_deg": float(r["angles_deg"][:, i].mean()) for i, n in enumerate(names)}}
        print(a.run.name, name, json.dumps({k: round(v, 1) for k, v in out[name].items()}), flush=True)
    if a.json:
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
