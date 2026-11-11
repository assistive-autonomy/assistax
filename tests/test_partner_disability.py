"""Per-partner disability: in ad hoc teamwork and cross-play, each loaded human acts in the body it was trained in,
even when partners with different disabilities share one vmapped batch (open issue E, 2026-11-11)."""
import types
import unittest

import jax
import jax.numpy as jp
import numpy as np

import assistax
from assistax import envs
from assistax.wrappers.aht import LoadAgentWrapper, LoadNetworkState, _disability_from_config

STATIC = {"joint_idx": 13, "joint_restriction_factor": 0.5, "joint_strength": 0.5, "tremor_magnitude": 0.1}
PER_PARTNER = {"joint_idx": 13, "per_partner": True}


def params(**kw) -> dict:
    """One partner's disability as the arrays the wrappers carry in state.info["disability"]."""
    p = {"joint_idx": 13, "joint_restriction_factor": 1.0, "joint_strength": 1.0, "tremor_magnitude": 0.0, **kw}
    return {k: jp.asarray(v, jp.int32 if k == "joint_idx" else jp.float32) for k, v in p.items()}


def run(env, n_steps: int, disability: dict | None = None, seed: int = 0):
    """Reset (optionally setting the per-partner disability) and take n_steps uniform random actions."""
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    key_reset, key = jax.random.split(jax.random.PRNGKey(seed))
    s = reset(key_reset)
    if disability is not None:
        s = s.replace(info={**s.info, "disability": disability})
    for _ in range(n_steps):
        key, k_act, k_step = jax.random.split(key, 3)
        s = step(k_step, s, jax.random.uniform(k_act, (env.action_size,), minval=-1.0, maxval=1.0))
    return s


class TestPerPartnerDisabilityWrapper(unittest.TestCase):
    def test_matches_the_static_wrapper(self) -> None:
        """Given per episode, a partner's disability acts exactly as the static wrapper it was trained with."""
        static = run(envs.create("scratchitch", disability=STATIC), 20)
        dynamic = run(envs.create("scratchitch", disability=PER_PARTNER), 20,
                      params(joint_restriction_factor=0.5, joint_strength=0.5, tremor_magnitude=0.1))
        np.testing.assert_allclose(np.asarray(dynamic.pipeline_state.qpos), np.asarray(static.pipeline_state.qpos),
                                   atol=1e-4)

    def test_partners_in_one_batch_move_as_their_own_bodies(self) -> None:
        """Two partners in one vmapped batch (elbow restriction 1.0 and 0.4; the reset pose is inside both ranges),
        the elbow driven hard each way: each env's elbow follows the path of a static DisabilityWrapper with that
        partner's restriction, and the two paths differ."""
        factors = (1.0, 0.4)
        n_steps = 150

        def elbow_path(env, sign: float, disability: dict | None, batched: bool) -> np.ndarray:
            """Elbow angle over n_steps with actuator 13 at full effort, envs keyed as in the batched run."""
            adr = env.unwrapped.sys.mj_model.joint("right_elbow").qposadr[0]
            reset, step = (jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))) if batched else \
                (jax.jit(env.reset), jax.jit(env.step))
            keys = jax.random.split(jax.random.PRNGKey(0), 2)
            s = reset(keys if batched else keys[disability])
            if batched:
                s = s.replace(info={**s.info, "disability": jax.tree.map(
                    lambda *x: jp.stack(x), *[params(joint_restriction_factor=f) for f in factors])})
            action = jp.zeros((2, env.action_size) if batched else (env.action_size,)).at[..., 13].set(sign)
            q = []
            for i in range(n_steps):
                k = jax.random.split(jax.random.PRNGKey(i), 2)
                s = step(k if batched else k[disability], s, action)
                q.append(np.asarray(s.pipeline_state.qpos[..., adr]))
            return np.array(q)

        gaps = []
        for sign in (1.0, -1.0):
            batch = elbow_path(envs.create("scratchitch", disability=PER_PARTNER), sign, None, batched=True)
            for i, f in enumerate(factors):
                static_env = envs.create("scratchitch", disability={**STATIC, "joint_restriction_factor": f,
                                                                    "joint_strength": 1.0, "tremor_magnitude": 0.0})
                # vmapped and single-env programs round differently; over 150 contact steps that reaches ~0.003 rad
                np.testing.assert_allclose(batch[:, i], elbow_path(static_env, sign, i, batched=False), atol=1e-2)
            gaps.append(np.abs(batch[:, 0] - batch[:, 1]).max())
        self.assertGreater(max(gaps), 0.1)


def _fake_partners(env, n: int) -> dict:
    """n loaded 'IPPO' humans whose policy outputs zero-mean actions (the network is irrelevant here)."""
    act_dim = env.action_spaces["human"].shape[0]

    def apply_fn(params, hstate, x):
        return types.SimpleNamespace(pi=(jp.zeros((n, act_dim)), jp.ones((n, act_dim))), hstate=None)

    return {"IPPO": {"human": LoadNetworkState(apply_fn=apply_fn, params={"w": jp.zeros(n)}, pop_size=n)}}


class TestLoadAgentWrapperDisability(unittest.TestCase):
    def test_disability_from_zoo_config(self) -> None:
        cfg = {"ENV_KWARGS": {"disability": {"joint_idx": 13, "joint_restriction_factor": 0.5, "joint_strength": 0.0}}}
        self.assertEqual(_disability_from_config(cfg), {"joint_idx": 13, "joint_restriction_factor": 0.5,
                                                        "joint_strength": 0.0, "tremor_magnitude": 0.0})
        self.assertEqual(_disability_from_config({"ENV_KWARGS": {}})["joint_strength"], 1.0)

    def test_sampled_partner_brings_its_disability(self) -> None:
        env = assistax.make("scratchitch", disability=PER_PARTNER)
        confs = [params(joint_restriction_factor=f, joint_strength=s) for f, s in ((1.0, 1.0), (0.5, 0.0), (0.0, 0.5))]
        stacked = {"human": jax.tree.map(lambda *x: jp.stack(x), *confs)}
        wrapped = LoadAgentWrapper(env, _fake_partners(env, 3), disability_configs=stacked)
        reset, step = jax.jit(wrapped.reset), jax.jit(wrapped.step)
        seen = set()
        for seed in range(12):
            _, s = reset(jax.random.PRNGKey(seed))
            i = int(s.ag_idx["human"])
            seen.add(i)
            for _ in range(2):
                acts = {a: jp.zeros(wrapped.action_space(a).shape) for a in wrapped.agents}
                _, s, _, _, _ = step(jax.random.PRNGKey(100 + seed), s, acts)
            got = {k: float(v) for k, v in s._state.info["disability"].items()}
            self.assertEqual(got, {k: float(v[i]) for k, v in stacked["human"].items()})
        self.assertGreater(len(seen), 1)

    def test_disabled_partners_need_a_disability_wrapper(self) -> None:
        env = assistax.make("scratchitch")
        stacked = {"human": jax.tree.map(lambda *x: jp.stack(x), params(), params(joint_strength=0.0))}
        with self.assertRaises(ValueError):
            LoadAgentWrapper(env, _fake_partners(env, 2), disability_configs=stacked)


if __name__ == "__main__":
    unittest.main()
