import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
import safetensors.flax
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict


def _tree_take(pytree, indices, axis=None):
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

def _tree_shape(pytree):
    return jax.tree.map(lambda x: x.shape, pytree)

def _unstack_tree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _stack_tree(pytree_list, axis=0):
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )

def _concat_tree(pytree_list, axis=0):
    return jax.tree.map(
        lambda *leaf: jnp.concat(leaf, axis=axis),
        *pytree_list
    )

def _tree_split(pytree, n, axis=0):
    leaves, treedef = jax.tree.flatten(pytree)
    split_leaves = zip(
        *jax.tree.map(lambda x: jnp.array_split(x,n,axis), leaves)
    )
    return [
        jax.tree.unflatten(treedef, leaves)
        for leaves in split_leaves
    ]

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not (done)
    ]

def _compute_episode_returns(eval_info, time_axis=-2):
    done_arr = eval_info.done["__all__"]
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    undiscounted_returns = jax.tree.map(
        lambda r: (r*(1-episode_done)).sum(axis=time_axis),
        eval_info.reward
    )
    return undiscounted_returns



@hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from mappo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
        case (False, True):
            from mappo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
        case (True, False):
            from mappo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
        case (True, True):
            from mappo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    print(f"Starting training with {config['TOTAL_TIMESTEPS']} timesteps \n num envs: {config['NUM_ENVS']} \n num seeds: {config['NUM_SEEDS']} \n for env: {config['ENV_NAME']}")
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        # first run (includes JIT)
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )

        # SAVE TRAIN METRICS
        EXCLUDED_METRICS = ["train_state"]
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )

        # SAVE PARAMS
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        all_train_states = out["metrics"]["train_state"]
        final_train_state = out["runner_state"].train_state
        safetensors.flax.save_file(
            flatten_dict(all_train_states.actor.params, sep='/'),
            "all_params.safetensors"
        )
        if config["network"]["agent_param_sharing"]:
            safetensors.flax.save_file(
                flatten_dict(final_train_state.actor.params, sep='/'),
                "final_params.safetensors"
            )
        else:
            # split by agent
            split_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state.actor.params)
            )
            for agent, params in zip(env.agents, split_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"{agent}.safetensors",
                )

        # RUN EVALUATION
        # Assume the first 2 dimensions are batch dims
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.actor.params))[:2]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        def _flatten_and_split_trainstate(trainstate):
            # We define this operation and JIT it for memory reasons
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0]*x.shape[1],*x.shape[2:])),
                trainstate
            )
            return _tree_split(flat_trainstate, n_sequential_evals)
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states.actor)
        eval_env, run_eval = make_evaluation(config)
        eval_log_config = EvalInfoLogConfig(
            env_state=False,
            done=True,
            action=False,
            value=False,
            reward=True,
            log_prob=False,
            obs=False,
            info=False,
            avail_actions=False,
        )
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
        evals = _concat_tree([
            eval_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # COMPUTE RETURNS
        first_episode_returns = _compute_episode_returns(evals)
        first_episode_returns = first_episode_returns["__all__"]
        mean_episode_returns = first_episode_returns.mean(axis=-1)

        # SAVE RETURNS
        jnp.save("returns.npy", mean_episode_returns)

        # RENDER
        # Run episodes for render (saving env_state at each timestep)
        render_log_config = EvalInfoLogConfig(
            env_state=True,
            done=True,
            action=False,
            value=False,
            reward=True,
            log_prob=False,
            obs=False,
            info=False,
            avail_actions=False,
        )
        eval_final = eval_jit(eval_rng, _tree_take(final_train_state.actor, 0, axis=0), render_log_config) # not sure why we don't just make final_train_state = out["runner_state"].train_state.actor already
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        worst_idx = episode_argsort.take(0,axis=-1)
        best_idx = episode_argsort.take(-1, axis=-1)
        median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)

        from brax.io import html
        worst_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=worst_idx,
        )
        median_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=median_idx,
        )
        best_episode = _take_episode(
            eval_final.env_state.env_state.pipeline_state, first_episode_done,
            time_idx=-1, eval_idx=best_idx,
        )
        html.save("final_worst.html", eval_env.sys, worst_episode)
        html.save("final_median.html", eval_env.sys, median_episode)
        html.save("final_best.html", eval_env.sys, best_episode)


if __name__ == "__main__":
    main()