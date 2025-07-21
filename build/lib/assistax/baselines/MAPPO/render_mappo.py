import os
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=true "
#     "--xla_dump_to=xla_dump "
# )
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
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
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Callable
from flax import struct



@struct.dataclass
class EvalNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict

@struct.dataclass
class ActorNetworkState:
    actor: EvalNetworkState

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

def _tree_shape(pytree):
    return jax.tree.map(lambda x: x.shape, pytree)

def _stack_tree(pytree_list, axis=0):
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )

@hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from mappo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_ff_nps_mabrax import MultiActor as NetworkArch
        case (False, True):
            from mappo_ff_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_ff_ps_mabrax import Actor as NetworkArch
        case (True, False):
            from mappo_rnn_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_rnn_nps_mabrax import MultiActorRNN as NetworkArch
        case (True, True):
            from mappo_rnn_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from mappo_rnn_ps_mabrax import ActorRNN as NetworkArch
        case _:
            raise Exception
    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    with jax.disable_jit(config["DISABLE_JIT"]):
        
        if config['ENV_NAME'] == 'pushcoop':
            robot1_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["human"]), sep='/'
            )

            robot2_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["robot"]), sep='/'
            )
            agent_params = {'robot1': robot1_params, 'robot2': robot2_params}
        
        else:
            human_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["human"]), sep='/'
            )

            robot_params = unflatten_dict(
                safetensors.flax.load_file(config["eval"]["path"]["robot"]), sep='/'
            )

            agent_params = {'human': human_params, 'robot': robot_params}

        eval_env, run_eval = make_evaluation(config)
        eval_log_config = EvalInfoLogConfig(
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
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        network = NetworkArch(config=config)

        if config['ENV_NAME'] == 'pushcoop':
            robot1 = _tree_take(
                agent_params["robot1"],
                0,
                axis=0
            )
            robot2 = _tree_take(
                agent_params["robot2"],
                0,
                axis=0
            )
            
            eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=_stack_tree([robot1, robot2]),
            )

        else:    
            robot = _tree_take(
                agent_params["robot"],
                0,
                axis=0
            )
            human = _tree_take(
                agent_params["human"],
                0,
                axis=0
            )
            eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=_stack_tree([robot, human]),
            )

        # final_eval_network_state = ActorNetworkState(
        #     actor=eval_network_state
        # )
        
        # RENDER
        # Run episodes for render (saving env_state at each timeste
    
        breakpoint()
        eval_final = eval_jit(eval_rng, eval_network_state, eval_log_config) # change eval_network_state to final_eval_network_state if needed
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
        html.save(f"final_worst_r{int(first_episode_returns[worst_idx])}.html", eval_env.sys, worst_episode)
        html.save(f"final_median_r{int(first_episode_returns[median_idx])}.html", eval_env.sys, median_episode)
        html.save(f"final_best_r{int(first_episode_returns[best_idx])}.html", eval_env.sys, best_episode)



if __name__ == "__main__":
    main()