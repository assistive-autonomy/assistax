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
from jaxmarl.wrappers.aht import ZooManager
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



@hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax_zoo_gen")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from mappo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (False, True):
            from mappo_ff_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (True, False):
            from mappo_rnn_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
        case (True, True):
            from mappo_rnn_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    print(f"Starting training with {config['TOTAL_TIMESTEPS']} timesteps \n num envs: {config['NUM_ENVS']} \n num seeds: {config['NUM_SEEDS']} \n for env: {config['ENV_NAME']}")
    with jax.disable_jit(config["DISABLE_JIT"]):
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        train_jit = jax.jit(
            make_train(config, save_train_state=False),
            device=jax.devices()[config["DEVICE"]]
        )
        out = jax.vmap(train_jit, in_axes=(0, None, None, None))(
            train_rngs,
            config["LR"], config["ENT_COEF"], config["CLIP_EPS"]
        )
        final_train_state = out["runner_state"].train_state.actor.params
        zoo = ZooManager(config["ZOO_PATH"])
        for agent_idx, agent_id in enumerate(env.agents):
            for seed_idx in range(config["NUM_SEEDS"]):
                zoo.save_agent(
                    config=config,
                    param_dict=_tree_take( # agent
                        _tree_take( # seed
                            final_train_state,
                            seed_idx,
                            axis=0,
                        ),
                        agent_idx,
                        axis=0,
                    ),
                    scenario_agent_id=agent_id
                )
        # TODO we might want to save final returns...

if __name__ == "__main__":
    main()
