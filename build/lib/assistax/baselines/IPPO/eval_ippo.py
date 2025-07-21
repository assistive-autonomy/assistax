import os
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=true "
#     "--xla_dump_to=xla_dump "
# )
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
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


@hydra.main(version_base=None, config_path="config", config_name="ippo_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
        case (False, False):
            from ippo_ff_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_ff_nps_mabrax import MultiActorCritic as NetworkArch
        case (False, True):
            from ippo_ff_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_ff_ps_mabrax import ActorCritic as NetworkArch
        case (True, False):
            from ippo_rnn_nps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_rnn_nps_mabrax import MultiActorCriticRNN as NetworkArch
        case (True, True):
            from ippo_rnn_ps_mabrax import make_train, make_evaluation, EvalInfoLogConfig
            from ippo_rnn_ps_mabrax import ActorCriticRNN as NetworkArch
        case _:
            raise Exception

    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    with jax.disable_jit(config["DISABLE_JIT"]):
        all_train_states = unflatten_dict(safetensors.flax.load_file(config["eval"]["path"]['all']), sep='/')
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states["params"]))[:2]
        
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
        network = NetworkArch(config=config)
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
        def eval_mem_efficient():
            eval_network_state = EvalNetworkState(apply_fn=network.apply, params=all_train_states)
            
            split_trainstate = _flatten_and_split_trainstate(eval_network_state)
            evals = _concat_tree([
                eval_vmap(eval_rng, ts, eval_log_config)
                for ts in tqdm(split_trainstate, desc="Evaluation batches")
            ])
            evals = jax.tree.map(
                lambda x: x.reshape((*batch_dims, *x.shape[1:])),
                evals
            )
            return evals
        evals = jax.jit(eval_mem_efficient)()

   
        first_episode_returns = _compute_episode_returns(evals)
        first_episode_returns = first_episode_returns["__all__"]
        mean_episode_returns = first_episode_returns.mean(axis=-1)

        std_error = first_episode_returns.std(axis=-1) / jnp.sqrt(first_episode_returns.shape[-1])

        ci_lower = mean_episode_returns - 1.96 * std_error
        ci_upper = mean_episode_returns + 1.96 * std_error


        # SAVE RETURNS
        jnp.save("returns.npy", mean_episode_returns)
        jnp.save("returns_ci_lower.npy", ci_lower)
        jnp.save("returns_ci_upper.npy", ci_upper)

if __name__ == "__main__":
    main()
