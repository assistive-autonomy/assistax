import os
import os.path as osp
import jax
import jax.numpy as jnp
import hydra
import safetensors.flax
import pandas as pd
from tqdm import tqdm
from flax import struct
from flax.traverse_util import unflatten_dict
from omegaconf import OmegaConf
from assistax.wrappers.aht import ZooManager, extract_uuids_from_eval_results
from hydra.utils import to_absolute_path
from typing import Dict, List, Any, Callable, Tuple
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode, _compute_episode_returns,
    _tree_shape, _stack_tree, _concat_tree, _tree_split,
    )
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@struct.dataclass
class EvalNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict


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

def sample_teams(
    robot_df: pd.DataFrame,
    human_df: pd.DataFrame,
    n_teams: int | None,
    rng_key: jax.Array,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Subsample co-trained robot-human pairs by team_uuid."""
    common_teams = list(set(robot_df.team_uuid) & set(human_df.team_uuid))
    if n_teams is None or n_teams >= len(common_teams):
        return robot_df, human_df
    indices = jax.random.choice(rng_key, len(common_teams), shape=(n_teams,), replace=False)
    selected = [common_teams[int(i)] for i in indices]
    return (
        robot_df[robot_df.team_uuid.isin(selected)],
        human_df[human_df.team_uuid.isin(selected)],
    )


@hydra.main(version_base=None, config_path="config", config_name="crossplay_zoo")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE

    # Dictionary to hold functions per algorithm (this is disgusting so needs to be refactored)
    alg_funcs = {}

    if "IPPO" in config["crossplay"]["robot_algos"]:
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from IPPO.ippo_ff_nps import (
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
                from IPPO.ippo_ff_ps import (
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
                from IPPO.ippo_rnn_nps import (
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
                from IPPO.ippo_rnn_ps import (
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
                from MAPPO.mappo_ff_nps import (
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
                from MAPPO.mappo_ff_ps import (
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
                from MAPPO.mappo_rnn_nps import (
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
                from MAPPO.mappo_rnn_ps import (
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
        from ..MASAC.masac_ff_nps import (
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
        # Strip preference_rewards — human prefs come from LoadEvalAgentWrapper
        if "preference_rewards" in robo_configs[alg].get("ENV_KWARGS", {}):
            robo_configs[alg]["ENV_KWARGS"] = {
                k: v for k, v in robo_configs[alg]["ENV_KWARGS"].items()
                if k != "preference_rewards"
            }
        # Inject crossplay config values into algo configs
        robo_configs[alg]["NUM_EVAL_EPISODES"] = config["NUM_EVAL_EPISODES"]
        robo_configs[alg]["ZOO_PATH"] = config["ZOO_PATH"]
        robo_configs[alg]["NUM_ENVS"] = config["NUM_ENVS"]
        robo_configs[alg]["NUM_STEPS"] = config["NUM_STEPS"]
        robo_configs[alg]["DISABLE_JIT"] = config["DISABLE_JIT"]

    rng = jax.random.PRNGKey(config["SEED"])
    max_pairs = config.get("MAX_CROSSPLAY_PAIRS", None)

    with jax.disable_jit(config["DISABLE_JIT"]):
        zoo = ZooManager(config["ZOO_PATH"])
        scenario = config["ENV_NAME"]

        partner_dict = {}
        for partner_algo in config["PARTNER_ALGORITHMS"]:
            partner_dict[partner_algo] = zoo.index.query(f'algorithm == "{partner_algo}"'
                                                        ).query(f'scenario == "{scenario}"'
                                                        ).query('scenario_agent_id == "human"')

        robo_filtered = {}

        for alg in config["crossplay"]["robot_algos"]:
            robo_filtered[alg] = zoo.index.query(f'algorithm == "{alg}"'
                                         ).query(f'scenario == "{scenario}"'
                                         ).query('scenario_agent_id == "robot"')

        # Team-based sampling: subsample co-trained robot-human pairs
        for alg in config["crossplay"]["robot_algos"]:
            if alg in partner_dict:
                rng, sample_key = jax.random.split(rng)
                robo_filtered[alg], partner_dict[alg] = sample_teams(
                    robo_filtered[alg], partner_dict[alg], max_pairs, sample_key
                )
                if max_pairs is not None:
                    print(f"Sampled {len(robo_filtered[alg])} robots and "
                          f"{len(partner_dict[alg])} humans for {alg}")

        # Rebuild after sampling
        load_zoo_dict = {algo: {"human": list(partner_dict[algo].agent_uuid)} for algo in partner_dict.keys()}

        human_uuids = []
        human_team_uuids = []
        for algo in partner_dict.keys():
            human_uuids.extend(list(partner_dict[algo].agent_uuid))
            human_team_uuids.extend(list(partner_dict[algo].team_uuid))

        num_humans = len(human_uuids)

        results = {}

        for alg, robo_agents in robo_filtered.items():
            inner_returns_dict = {}
            inner_returns_full_dict = {}

            agent_uuids = list(robo_agents.agent_uuid)
            robot_team_uuids = list(robo_agents.team_uuid)

            network = alg_funcs[alg]["NetworkArch"](config=robo_configs[alg])

            eval_env, run_eval = alg_funcs[alg]["make_evaluation"](
                robo_configs[alg], load_zoo=load_zoo_dict, crossplay=True
            )

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

            eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
            # Vmap over seeds: axis 0 of rngs, robot params shared across seeds
            eval_seed_vmap = jax.vmap(eval_jit, in_axes=(0, None, None))

            for robot_uuid in tqdm(agent_uuids, desc=f"Evaluating {alg} robots"):
                # Load single robot params
                robot_params = unflatten_dict(
                    safetensors.flax.load_file(
                        osp.join(config["ZOO_PATH"], "params", robot_uuid + ".safetensors")
                    ),
                    sep='/'
                )
                robot_params = jax.tree.map(lambda x: jnp.expand_dims(x, 0), robot_params)
                robot_eval_state = EvalNetworkState(
                    apply_fn=network.apply, params=robot_params
                )

                # Create per-seed RNGs: shape (N_SEEDS, num_humans, 2)
                seed_keys = jax.random.split(rng, config["NUM_SEEDS"])
                episode_rngs = jax.vmap(
                    lambda k: jax.random.split(k, num_humans)
                )(seed_keys)

                # Run: vmapped over N_SEEDS, internally scans over num_humans
                agent_evals = eval_seed_vmap(
                    episode_rngs, robot_eval_state, eval_log_config
                )

                # Compute returns: (N_SEEDS, num_humans, NUM_EVAL_EPISODES)
                episode_returns = _compute_episode_returns(agent_evals)
                full_returns = episode_returns["__all__"]
                inner_returns_full_dict[robot_uuid] = full_returns
                # Mean over seeds and episodes: (num_humans,)
                inner_returns_dict[robot_uuid] = full_returns.mean(axis=(0, 2))

            results[alg] = {
                "returns": inner_returns_dict,
                "returns_full": inner_returns_full_dict,
                "robot_uuids": agent_uuids,
                "human_uuids": human_uuids,
                "robot_team_uuids": robot_team_uuids,
                "human_team_uuids": human_team_uuids,
                "num_seeds": config["NUM_SEEDS"],
                "num_eval_episodes": config["NUM_EVAL_EPISODES"],
            }

    if max_pairs is not None:
        output_filename = f"crossplay_results_{max_pairs}pairs.npy"
    else:
        output_filename = "crossplay_test_results.npy"

    jnp.save(output_filename, results, allow_pickle=True)

    print(f"Evaluation complete! Results saved to {output_filename}")
    return results

if __name__ == "__main__":
    main()
