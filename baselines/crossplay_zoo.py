import os
import os.path as osp
import time
from tqdm import tqdm
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
import safetensors.flax
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
from jaxmarl.wrappers.aht_all import ZooManager, LoadAgentWrapper, extract_uuids_from_eval_results
import hydra
from omegaconf import OmegaConf
import pandas as pd
from typing import Sequence, NamedTuple, Any, Dict, Callable
from hydra import compose, initialize
from hydra.utils import to_absolute_path


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
    unstacked_leaves = list(zip(*leaves))
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

def _flatten_and_split_trainstate(trainstate, n_sequential_evals=1):
    # We define this operation and JIT it for memory reasons
    flat_trainstate = jax.tree.map(
        lambda x: x.reshape((x.shape[0]*x.shape[1],*x.shape[2:])),
        trainstate
    )
    return _tree_split(flat_trainstate, n_sequential_evals)

def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    # TODO make sure this works for the NPS code.
    episodes = _tree_take(pipeline_states, time_idx, axis=0)
    episodes = _tree_take(episodes, eval_idx, axis=1)
    dones = dones.take(time_idx, axis=0)
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


def load_and_merge_algo_config(alg_config: dict):
    """
    Given a dictionary with keys "main" and "network" (paths to YAML files),
    load each config and merge them so that the network config is available
    under the 'network' key in the final config.
    """
    # Resolve absolute paths using Hydra's to_absolute_path
    main_config_path = to_absolute_path(alg_config["main"])
    network_config_path = to_absolute_path(alg_config["network"])
    
    # Load the main and network configs
    main_cfg = OmegaConf.load(main_config_path)
    network_cfg = OmegaConf.load(network_config_path)
    
    # Merge them: here we embed the network config under the key "network".
    # If main_cfg already has a "network" key (for instance if using defaults), 
    # OmegaConf.merge will combine them.
    merged_cfg = OmegaConf.merge(main_cfg, OmegaConf.create({"network": network_cfg}))
    return merged_cfg

@hydra.main(version_base=None, config_path="crossplay_config", config_name="crossplay_zoo")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # Add configuration for parallel batch size
    parallel_batch_size = config.get("PARALLEL_BATCH_SIZE", 1)  # Default to 1 if not specified

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    
    # Dictionary to hold functions per algorithm (this is disgusting so needs to be refactored)
    alg_funcs = {}

    if "IPPO" in config["crossplay"]["robot_algos"]:
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from IPPO.ippo_ff_nps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    MultiActorCritic as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case (False, True):
                from IPPO.ippo_ff_ps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    ActorCritic as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case (True, False):
                from IPPO.ippo_rnn_nps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    MultiActorCriticRNN as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case (True, True):
                from IPPO.ippo_rnn_ps_mabrax import (
                    make_train as ippo_make_train,
                    make_evaluation as ippo_make_evaluation,
                    EvalInfoLogConfig as ippo_EvalInfoLogConfig,
                    ActorCriticRNN as ippo_NetworkArch,
                )
                alg_funcs["IPPO"] = {
                    "make_train": ippo_make_train,
                    "make_evaluation": ippo_make_evaluation,
                    "EvalInfoLogConfig": ippo_EvalInfoLogConfig,
                    "NetworkArch": ippo_NetworkArch,
                }
            case _:
                raise Exception("Invalid network configuration for IPPO")

    if "MAPPO" in config["crossplay"]["robot_algos"]:
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from MAPPO.mappo_ff_nps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    MultiActor as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case (False, True):
                from MAPPO.mappo_ff_ps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    Actor as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case (True, False):
                from MAPPO.mappo_rnn_nps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    MultiActorRNN as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case (True, True):
                from MAPPO.mappo_rnn_ps_mabrax import (
                    make_train as mappo_make_train,
                    make_evaluation as mappo_make_evaluation,
                    EvalInfoLogConfig as mappo_EvalInfoLogConfig,
                    ActorRNN as mappo_NetworkArch,
                )
                alg_funcs["MAPPO"] = {
                    "make_train": mappo_make_train,
                    "make_evaluation": mappo_make_evaluation,
                    "EvalInfoLogConfig": mappo_EvalInfoLogConfig,
                    "NetworkArch": mappo_NetworkArch,
                }
            case _:
                raise Exception("Invalid network configuration for MAPPO")

    if "MASAC" in config["crossplay"]["robot_algos"]:
        from MASAC.masac_ff_nps_mabrax import (
            make_train as masac_make_train,
            make_evaluation as masac_make_evaluation,
            EvalInfoLogConfig as masac_EvalInfoLogConfig,
            MultiSACActor as masac_NetworkArch,
        )
        alg_funcs["MASAC"] = {
            "make_train": masac_make_train,
            "make_evaluation": masac_make_evaluation,
            "EvalInfoLogConfig": masac_EvalInfoLogConfig,
            "NetworkArch": masac_NetworkArch,
        }

    robo_configs = {}
    for alg, paths in config["crossplay"]["algo_configs"].items():
        robo_configs[alg] = load_and_merge_algo_config(paths)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)

    with jax.disable_jit(config["DISABLE_JIT"]):
        zoo = ZooManager(config["ZOO_PATH"])
        scenario = config["ENV_NAME"]

        partner_dict = {}
        for partner_algo in config["PARTNER_ALGORITHMS"]:
            partner_dict[partner_algo] = zoo.index.query(f'algorithm == "{partner_algo}"'
                                                        ).query(f'scenario == "{scenario}"'
                                                        ).query('scenario_agent_id == "human"')
            
        num_humans = sum(len(x) for x in partner_dict.values())

        load_zoo_dict = {algo: {"human": list(partner_dict[algo].agent_uuid)} for algo in partner_dict.keys()}
        robo_filtered = {}
        for alg in config["crossplay"]["robot_algos"]:
            robo_filtered[alg] = zoo.index.query(f'algorithm == "{alg}"'
                                         ).query(f'scenario == "{scenario}"'
                                         ).query('scenario_agent_id == "robot"')
            
        # robo_filtered = {alg: df.head(5) for alg, df in robo_filtered.items()}
        returns_dict = {}
        opponent_info_dict = {}
        # This function actually seems largely obsolute except maybe for batch_uuids
        def add_batch_dim(x):
            # Add a leading dimension to the array
            return jnp.expand_dims(x, axis=0)
        
        def create_batch_network_states(batch_uuids, network, multi_agent=False):
            batch_states = []
            for agent_uuid in batch_uuids:
                # Load parameters for robot and human
                agent_params = unflatten_dict(
                        safetensors.flax.load_file(osp.join(config["ZOO_PATH"], "params", agent_uuid+".safetensors")),
                        sep='/'
                    )
                # # Splice parameters for this agent pair
                # spliced_params = jax.tree.map(
                #     lambda *p: jnp.stack(p, axis=0),
                #     *(agent_params[a] for a in env.agents)
                # )
                
                agent_params = jax.tree_util.tree_map(add_batch_dim, agent_params)

                # Create network state
                eval_network_state = EvalNetworkState(
                    apply_fn=network.apply,
                    params=agent_params,
                )
                batch_states.append(eval_network_state)
            return batch_states

        for alg, robo_agents in robo_filtered.items():
            inner_returns_dict = {}
            inner_opponent_info = {}  
            
            # Get all agent UUIDs for this algorithm
            agent_uuids = list(robo_agents.agent_uuid)
            
            # Create batches of agent UUIDs
            batches = [agent_uuids[i:i + parallel_batch_size] for i in range(0, len(agent_uuids), parallel_batch_size)]
            
            # Create network for this algorithm
            network = alg_funcs[alg]["NetworkArch"](config=robo_configs[alg])
            
            # Set up eval environment and function
            eval_env, run_eval = alg_funcs[alg]["make_evaluation"](robo_configs[alg], load_zoo=load_zoo_dict, crossplay=True) #testings this
            
            # Configure logging based on algorithm
            if alg == "MASAC":
                eval_log_config = alg_funcs[alg]["EvalInfoLogConfig"](
                    env_state=False, done=True, action=False, reward=True,
                    log_prob=False, obs=False, info=True, avail_actions=False,
                )
            else:
                eval_log_config = alg_funcs[alg]["EvalInfoLogConfig"](
                    env_state=False, done=True, action=False, value=False, reward=True,
                    log_prob=False, obs=False, info=True, avail_actions=False,
                )
            
            # Create jitted evaluation function
            eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info", "num_episodes"])
            
            # Create vmapped evaluation function
            # This will apply eval_jit to each network state in parallel
            eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
            multi_agent = True if alg in ["MASAC", "MAPPO"] else False
            for batch in batches:
                # Create a batch of network states
                batch_network_states = create_batch_network_states(batch, network, multi_agent=multi_agent)
                # Stack network states for vmapping
                # We need to handle the structure correctly for vmapping
                # stacked_params = jax.tree.map(
                #     lambda *p: jnp.stack(p),
                #     *[state.params for state in batch_network_states]
                # )
                stacked_params = jax.tree.map(
                    lambda *p: jnp.expand_dims(jnp.stack(p), axis=0),
                    *[state.params for state in batch_network_states]
                )

                # Run vmapped evaluation
                # We need to be careful about the structure of the network states
                # For vmapping, we need the first dimension to be the batch dimension
                batch_rngs = jax.random.split(eval_rng, len(batch))
                episode_rngs = jax.random.split(eval_rng, num_humans)
                
                batch_dims = jax.tree.leaves(_tree_shape(stacked_params["params"]))[:2]
                breakpoint()
                def eval_mem_efficient():
                    eval_network_state = EvalNetworkState(apply_fn=network.apply, params=stacked_params)
                    split_trainstate = _flatten_and_split_trainstate(eval_network_state)
                    breakpoint
                    evals = _concat_tree([
                        eval_vmap(episode_rngs, ts, eval_log_config)
                        for ts in tqdm(split_trainstate, desc="Evaluation batches")
                    ])
                    evals = jax.tree.map(
                        lambda x: x.reshape((*batch_dims, *x.shape[1:])),
                        evals
                    )
                    return evals
                # batch_eval_states = EvalNetworkState(
                #     apply_fn=network.apply,
                #     params=stacked_params
                # )

                # batch_evals = eval_vmap(episode_rngs, batch_eval_states, eval_log_config) # here is where I get the error 
                batch_evals = jax.jit(eval_mem_efficient)()

                # Process results for each agent in the batch
                for i, agent_uuid in enumerate(batch):
                    if isinstance(batch_evals, list):
                        agent_evals = batch_evals[i]
                    else:
                        agent_evals = jax.tree.map(
                            lambda x: x[i] if hasattr(x, '__getitem__') and i < len(x) else x,
                            batch_evals
                        )
                    
                    # Compute returns
                    episode_returns = _compute_episode_returns(agent_evals)
                    mean_returns = episode_returns["__all__"].mean(axis=-1)
 
                    opponent_uuids = extract_uuids_from_eval_results(eval_env, agent_evals)

                    # Store returns and opponent info
                    inner_returns_dict[agent_uuid] = mean_returns
                    inner_opponent_info[agent_uuid] = opponent_uuids
                            
            returns_dict[alg] = inner_returns_dict
            opponent_info_dict[alg] = inner_opponent_info
    jnp.save("crossplay_test_results.npy", returns_dict, allow_pickle=True)
    breakpoint()
    # Now you can use returns_dict for analysis
    print("Evaluation complete!")
    return returns_dict

if __name__ == "__main__":
    main()
