import os
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
from base64 import urlsafe_b64encode

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

def _compute_episode_returns(eval_info, common_reward=False, time_axis=-2):
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
    if "__all__" not in undiscounted_returns:
        undiscounted_returns.update({
            "__all__": (sum(undiscounted_returns.values())
                        /(len(undiscounted_returns) if common_reward else 1))
        })
    return undiscounted_returns

def _generate_sweep_axes(rng, config):
    p_lr_rng, q_lr_rng, alpha_lr_rng, tau_rng = jax.random.split(rng, 4)
    sweep_config = config["SWEEP"]
    if sweep_config.get("p_lr", False):
        p_lrs = 10**jax.random.uniform(
            p_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["p_lr"]["min"],
            maxval=sweep_config["p_lr"]["max"],
        )
        p_lr_axis = 0
    else:
        p_lrs = config["POLICY_LR"]
        p_lr_axis = None

    if sweep_config.get("q_lr", False):
        q_lrs = 10**jax.random.uniform(
            q_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["q_lr"]["min"],
            maxval=sweep_config["q_lr"]["max"],
        )
        q_lr_axis = 0
    else:
        q_lrs = config["Q_LR"]
        q_lr_axis = None

    if sweep_config.get("alpha_lr", False):
        alpha_lrs = 10**jax.random.uniform(
            alpha_lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["alpha_lr"]["min"],
            maxval=sweep_config["alpha_lr"]["max"],
        )
        alpha_lr_axis = 0
    else:
        alpha_lrs = config["ALPHA_LR"]
        alpha_lr_axis = None

    if sweep_config.get("tau", False):
        taus = 10**jax.random.uniform(
            tau_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["tau"]["min"],
            maxval=sweep_config["tau"]["max"],
        )
        tau_axis = 0
    else:
        taus = config["TAU"]
        tau_axis = None


    return {
        "p_lr": {"val": p_lrs, "axis": p_lr_axis},
        "q_lr": {"val": q_lrs, "axis": q_lr_axis},
        "alpha_lr": {"val": alpha_lrs, "axis": alpha_lr_axis},
        "tau": {"val": taus, "axis":tau_axis},
    }

@hydra.main(version_base=None, config_path="config", config_name="masac_sweep")
def main(config):
    config_key = hash(config) % 2**62
    config_key = urlsafe_b64encode(
        config_key.to_bytes(
            (config_key.bit_length()+8)//8,
            "big", signed=False
        )
    ).decode("utf-8").replace("=", "")
    os.makedirs(config_key, exist_ok=True)
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    # match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
    #     case (False, False):
    #         from ippo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
    #     case (False, True):
    #         from ippo_ff_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
    #     case (True, False):
    #         from ippo_rnn_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
    #     case (True, True):
    #         from ippo_rnn_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig

    from masac_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    sweep = _generate_sweep_axes(sweep_rng, config)
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        out = jax.vmap(
            jax.vmap(
                train_jit,
                in_axes=(0, None, None, None, None)
            ),
            in_axes=(
                None,
                sweep["p_lr"]["axis"],
                sweep["q_lr"]["axis"],
                sweep["alpha_lr"]["axis"],
                sweep["tau"]["axis"],
            )
        )(
            train_rngs,
            sweep["p_lr"]["val"],
            sweep["q_lr"]["val"],
            sweep["alpha_lr"]["val"],
            sweep["tau"]["val"]
        )

        # SAVE TRAIN METRICS
        EXCLUDED_METRICS = ["actor_train_state", "q1_train_state", "q2_train_state"]
        saveable_metrics = {key: val.copy() for key, val in out["metrics"].items() if key not in EXCLUDED_METRICS}
        
        jnp.save(f"{config_key}/metrics.npy", {
            key: val
            for key, val in saveable_metrics.items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )
        
        # SAVE SWEEP HPARAMS
        jnp.save(f"{config_key}/hparams.npy", {
            "p_lr": sweep["p_lr"]["val"],
            "q_lr": sweep["q_lr"]["val"],
            "alpha_lr": sweep["alpha_lr"]["val"],            
            "tau": sweep["tau"]["val"],
            "num_updates": config["NUM_UPDATES"],
            "total_timesteps": config["TOTAL_TIMESTEPS"],
            "num_envs": config["NUM_ENVS"],
            "num_sac_updates": config["NUM_SAC_UPDATES"],
            "batch_size": config["BATCH_SIZE"],
            "buffer_size": config["BUFFER_SIZE"],
            "rollout_length": config["ROLLOUT_LENGTH"],
            "explore_steps": config["EXPLORE_STEPS"],
            }
        )

        # SAVE PARAMS
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        all_train_states = out["metrics"]["actor_train_state"]

        final_train_state = out["runner_state"].train_states.actor
        safetensors.flax.save_file(
            flatten_dict(all_train_states.params, sep='/'),
            f"{config_key}/all_params.safetensors"
        )
        if config["network"]["agent_param_sharing"]:
            safetensors.flax.save_file(
                flatten_dict(final_train_state.params, sep='/'),
                f"{config_key}/final_params.safetensors"
            )
        else:
            # split by agent
            split_params = _unstack_tree(
                jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), final_train_state.params)
            )
            for agent, params in zip(env.agents, split_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"{config_key}/{agent}.safetensors",
                )

        # RUN EVALUATION
        # Assume the first 3 dimensions are batch dims
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        def _flatten_and_split_trainstate(train_state):
            # We define this operation and JIT it for memory reasons
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0]*x.shape[1]*x.shape[2],*x.shape[3:])),
                train_state
            )
            return _tree_split(flat_trainstate, n_sequential_evals)
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)

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
            static_argnames=["log_eval_info"], # do the eval_state eventually
        )
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
        evals = _concat_tree([
            eval_vmap(eval_rng, ts, False)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # COMPUTE RETURNS
        first_episode_returns = _compute_episode_returns(evals)
        mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)

        # SAVE RETURNS
        jnp.save(f"{config_key}/returns.npy", mean_episode_returns)


if __name__ == "__main__":
    main()