import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
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
    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(rng, 3)
    sweep_config = config["SWEEP"]
    if sweep_config.get("lr", False):
        lrs = 10**jax.random.uniform(
            lr_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["lr"]["min"],
            maxval=sweep_config["lr"]["max"],
        )
        lr_axis = 0
    else:
        lrs = config["LR"]
        lr_axis = None

    if sweep_config.get("ent_coef", False):
        ent_coefs = 10**jax.random.uniform(
            ent_coef_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["ent_coef"]["min"],
            maxval=sweep_config["ent_coef"]["max"],
        )
        ent_coef_axis = 0
    else:
        ent_coefs = config["ENT_COEF"]
        ent_coef_axis = None

    if sweep_config.get("clip_eps", False):
        clip_epss = 10**jax.random.uniform(
            clip_eps_rng,
            shape=(sweep_config["num_configs"],),
            minval=sweep_config["clip_eps"]["min"],
            maxval=sweep_config["clip_eps"]["max"],
        )
        clip_eps_axis = 0
    else:
        clip_epss = config["CLIP_EPS"]
        clip_eps_axis = None

    return {
        "lr": {"val": lrs, "axis": lr_axis},
        "ent_coef": {"val": ent_coefs, "axis":ent_coef_axis},
        "clip_eps": {"val": clip_epss, "axis":clip_eps_axis},
    }


@hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax")
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
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (False, True):
            from ippo_ff_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (True, False):
            from ippo_rnn_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (True, True):
            from ippo_rnn_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, sweep_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    sweep = _generate_sweep_axes(sweep_rng, config)
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=False),
            device=jax.devices()[config["DEVICE"]]
        )
        out = jax.vmap(
            jax.vmap(
                train_jit,
                in_axes=(0, None, None, None)
            ),
            in_axes=(
                None,
                sweep["lr"]["axis"],
                sweep["ent_coef"]["axis"],
                sweep["clip_eps"]["axis"],
            )
        )(
            train_rngs,
            sweep["lr"]["val"],
            sweep["ent_coef"]["val"],
            sweep["clip_eps"]["val"],
        )

        # SAVE TRAIN METRICS
        EXCLUDED_METRICS = ["train_state"]
        jnp.save(f"{config_key}/metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )
        
        # SAVE SWEEP HPARAMS
        jnp.save(f"{config_key}/hparams.npy", {
            "lr": sweep["lr"]["val"],
            "ent_coef": sweep["ent_coef"]["val"],
            "clip_eps": sweep["clip_eps"]["val"],
            "num_steps": config["NUM_STEPS"],
            "num_envs": config["NUM_ENVS"],
            "update_epochs": config["UPDATE_EPOCHS"],
            "num_minibatches": config["NUM_MINIBATCHES"],
            }
        )

        # SAVE PARAMS
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        # all_train_states = out["metrics"]["train_state"]

        final_train_state = out["runner_state"].train_state
        # safetensors.flax.save_file(
        #     flatten_dict(all_train_states.params, sep='/'),
        #     f"{config_key}/all_params.safetensors"
        # )
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
        batch_dims = jax.tree.leaves(_tree_shape(final_train_state.params))[:2]
        # batch_dims = jax.tree.leaves(_tree_shape(all_train_states.params))[:3]
        # n_sequential_evals = int(jnp.ceil(
        #     config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
        #     / config["GPU_ENV_CAPACITY"]
        # ))
        # def _flatten_and_split_trainstate(train_state):
        #     # We define this operation and JIT it for memory reasons
        #     flat_trainstate = jax.tree.map(
        #         lambda x: x.reshape((x.shape[0]*x.shape[1]*x.shape[2],*x.shape[3:])),
        #         train_state
        #     )
        #     return _tree_split(flat_trainstate, n_sequential_evals)
        # split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states)
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
        # # evals = _concat_tree([
        # #     eval_vmap(eval_rng, ts, eval_log_config)
        # #     for ts in tqdm(split_trainstate, desc="Evaluation batches")
        # # ])
        # # evals = jax.tree.map(
        # #     lambda x: x.reshape((*batch_dims, *x.shape[1:])),
        # #     evals
        # # )

        # def eval_mem_efficient():  
        #     split_trainstate = _flatten_and_split_trainstate(all_train_states)
        #     evals = _concat_tree([
        #         eval_vmap(eval_rng, ts, eval_log_config)
        #         for ts in tqdm(split_trainstate, desc="Evaluation batches")
        #     ])
        #     evals = jax.tree.map(
        #         lambda x: x.reshape((*batch_dims, *x.shape[1:])),
        #         evals
        #     )
        #     return evals
        
        # evals = jax.jit(eval_mem_efficient)()
        flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0]*x.shape[1],*x.shape[2:])),
                final_train_state
            )
        eval_final = eval_vmap(eval_rng, flat_trainstate, eval_log_config)
        
        # first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        # first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        # first_episode_returns = first_episode_rewards.sum(axis=0)
        # breakpoint()
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            eval_final
        )
       
        # # COMPUTE RETURNS
        first_episode_returns = _compute_episode_returns(evals)

        mean_episode_returns = first_episode_returns["__all__"].mean(axis=-1)

        # SAVE RETURNS
        jnp.save(f"{config_key}/returns.npy", mean_episode_returns)


if __name__ == "__main__":
    main()
