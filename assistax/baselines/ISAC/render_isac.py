import os
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
import safetensors.flax
import optax
import distrax
import assistax
from assistax.wrappers.baselines import get_space_dim, LogEnvState
from assistax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Callable
from flax import struct

@struct.dataclass
class EvalNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict

def _tree_take(pytree, indices, axis=None):
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)

def _unstack_tree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not (done)
    ]


@hydra.main(version_base=None, config_path="config", config_name="isac")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    from isac_ff_nps import MultiSACActor as NetworkArch
    from isac_ff_nps import make_evaluation as make_evaluation
    from isac_ff_nps import EvalInfoLogConfig

    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    
    with jax.disable_jit(config["DISABLE_JIT"]):

        all_train_states = unflatten_dict(safetensors.flax.load_file(config["eval"]["path"]), sep='/')

        eval_env, run_eval = make_evaluation(config)
        eval_jit = jax.jit(run_eval, 
                        static_argnames=["log_eval_info"],
                        )
        network = NetworkArch(config=config)
        # RENDER
        print(f"Started rendering runs...")
        # Run episodes for render (saving env_state at each timestep)
        # I need to find a way to combine the parameters for the human and robot so I can load just the final parameters
        
        eval_log_config = EvalInfoLogConfig(
            env_state=True,
            done=True,
            action=False,
            reward=True,
            log_prob=False,
            obs=False,
            info=False,
            avail_actions=False,
        )
                
        final_train_state = _tree_take(all_train_states, -1, axis=1)
        final_eval_network_state = EvalNetworkState(apply_fn=network.apply, params=final_train_state)
        final_eval = _tree_take(final_eval_network_state, 0, axis=0)
        # eval_final = eval_jit(eval_rng, _tree_take(final_eval_network_state, 0, axis=0), True)
        
        eval_final = eval_jit(eval_rng, final_eval, eval_log_config)
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        worst_idx = episode_argsort.take(0,axis=-1)
        best_idx = episode_argsort.take(-1, axis=-1)
        median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)
        
        from assistax.render import html
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
        html.save(f"final_worst_r{int(first_episode_returns[worst_idx])}.html", eval_env.sys, worst_episode)
        html.save(f"final_median_r{int(first_episode_returns[median_idx])}.html", eval_env.sys, median_episode)
        html.save(f"final_best_r{int(first_episode_returns[best_idx])}.html", eval_env.sys, best_episode)

        print(f"Rendering Completed!")

if __name__ == "__main__":
    main()