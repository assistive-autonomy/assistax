"""
Multi-Algorithm Crossplay Evaluation Script

This module implements comprehensive crossplay evaluation across different multi-agent reinforcement 
learning algorithms (IPPO, MAPPO, MASAC). Crossplay evaluation measures how well agents trained 
with one algorithm perform when paired with partners trained using different algorithms, providing 
insights into generalization, robustness, and algorithm compatibility.

Usage:
    python masac_crossplay.py [hydra options]
    
Configuration should specify:
- robot_algos: List of algorithms for robot agents to evaluate
- PARTNER_ALGORITHMS: List of algorithms for partner agents
- algo_configs: Configuration files for each algorithm
- ZOO_PATH: Path to zoo containing trained agents
"""

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
    _tree_shape, _stack_tree, _concat_tree, _tree_split
    )


# ================================ DATA STRUCTURES ================================

@struct.dataclass
class EvalNetworkState:
    """
    Lightweight network state container for crossplay evaluation.
    
    Contains only the essential components needed for policy evaluation:
    the network's apply function and trained parameters.
    
    Attributes:
        apply_fn: Neural network forward pass function
        params: Trained network parameters
    """
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict

# ================================ ALGORITHM LOADING UTILITIES ================================

def load_and_merge_algo_config(alg_config: Dict[str, str]) -> Dict:
    """
    Load and merge algorithm configuration files.
    
    Given a dictionary with keys "main" and "network" (paths to YAML files),
    load each config and merge them so that the network config is available
    under the 'network' key in the final config.
    
    Args:
        alg_config: Dictionary with "main" and "network" config file paths
        
    Returns:
        Merged configuration dictionary
    """
    # Resolve absolute paths using Hydra's to_absolute_path
    main_config_path = to_absolute_path(alg_config["main"])
    network_config_path = to_absolute_path(alg_config["network"])
    
    # Load the main and network configs
    main_cfg = OmegaConf.load(main_config_path)
    network_cfg = OmegaConf.load(network_config_path)
    
    # Merge them: embed the network config under the key "network"
    merged_cfg = OmegaConf.merge(main_cfg, OmegaConf.create({"network": network_cfg}))
    return merged_cfg


def load_algorithm_functions(config: Dict) -> Dict[str, Dict]:
    """
    Dynamically load algorithm functions based on configuration.
    
    Imports the appropriate make_train, make_evaluation, EvalInfoLogConfig,
    and NetworkArch functions for each algorithm based on network configuration.
    
    Args:
        config: Configuration dictionary containing algorithm and network settings
        
    Returns:
        Dictionary mapping algorithm names to their function dictionaries
    """
    alg_funcs = {}
    
    # ===== IPPO ALGORITHM LOADING =====
    if "IPPO" in config["crossplay"]["robot_algos"]:
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from IPPO.ippo_ff_nps import (
                    make_train, make_evaluation, EvalInfoLogConfig, MultiActorCritic as NetworkArch
                )
            case (False, True):
                from IPPO.ippo_ff_ps import (
                    make_train, make_evaluation, EvalInfoLogConfig, ActorCritic as NetworkArch
                )
            case (True, False):
                from IPPO.ippo_rnn_nps import (
                    make_train, make_evaluation, EvalInfoLogConfig, MultiActorCriticRNN as NetworkArch
                )
            case (True, True):
                from IPPO.ippo_rnn_ps import (
                    make_train, make_evaluation, EvalInfoLogConfig, ActorCriticRNN as NetworkArch
                )
            case _:
                raise ValueError("Invalid network configuration for IPPO")
        
        alg_funcs["IPPO"] = {
            "make_train": make_train,
            "make_evaluation": make_evaluation,
            "EvalInfoLogConfig": EvalInfoLogConfig,
            "NetworkArch": NetworkArch,
        }
        print("✓ Loaded IPPO algorithm functions")

    # ===== MAPPO ALGORITHM LOADING =====
    if "MAPPO" in config["crossplay"]["robot_algos"]:
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from MAPPO.mappo_ff_nps import (
                    make_train, make_evaluation, EvalInfoLogConfig, MultiActor as NetworkArch
                )
            case (False, True):
                from MAPPO.mappo_ff_ps import (
                    make_train, make_evaluation, EvalInfoLogConfig, Actor as NetworkArch
                )
            case (True, False):
                from MAPPO.mappo_rnn_nps import (
                    make_train, make_evaluation, EvalInfoLogConfig, MultiActorRNN as NetworkArch
                )
            case (True, True):
                from MAPPO.mappo_rnn_ps import (
                    make_train, make_evaluation, EvalInfoLogConfig, ActorRNN as NetworkArch
                )
            case _:
                raise ValueError("Invalid network configuration for MAPPO")
        
        alg_funcs["MAPPO"] = {
            "make_train": make_train,
            "make_evaluation": make_evaluation,
            "EvalInfoLogConfig": EvalInfoLogConfig,
            "NetworkArch": NetworkArch,
        }
        print("✓ Loaded MAPPO algorithm functions")

    # ===== MASAC ALGORITHM LOADING =====
    if "MASAC" in config["crossplay"]["robot_algos"]:
        from baselines.MASAC.masac_ff_nps import (
            make_train, make_evaluation, EvalInfoLogConfig, MultiSACActor as NetworkArch
        )
        
        alg_funcs["MASAC"] = {
            "make_train": make_train,
            "make_evaluation": make_evaluation,
            "EvalInfoLogConfig": EvalInfoLogConfig,
            "NetworkArch": NetworkArch,
        }
        print("✓ Loaded MASAC algorithm functions")

    return alg_funcs


# ================================ CROSSPLAY EVALUATION UTILITIES ================================

def setup_partner_zoo(zoo: ZooManager, scenario: str, partner_algorithms: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Set up partner agent pools from zoo for crossplay evaluation.
    
    Queries the zoo to find all available partner agents for each algorithm
    that will be used in crossplay evaluation.
    
    Args:
        zoo: ZooManager instance
        scenario: Environment/scenario name
        partner_algorithms: List of algorithms to load partners from
        
    Returns:
        Dictionary mapping algorithm names to DataFrames of available partners
    """
    partner_dict = {}
    
    for partner_algo in partner_algorithms:
        partners = zoo.index.query(f'algorithm == "{partner_algo}"'
                                 ).query(f'scenario == "{scenario}"'
                                 ).query('scenario_agent_id == "human"')
        
        partner_dict[partner_algo] = partners
        print(f"Found {len(partners)} {partner_algo} human partners")
    
    total_partners = sum(len(partners) for partners in partner_dict.values())
    print(f"Total partner agents available: {total_partners}")
    
    return partner_dict


def setup_robot_agents(zoo: ZooManager, scenario: str, robot_algorithms: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Set up robot agent pools from zoo for crossplay evaluation.
    
    Queries the zoo to find all available robot agents for each algorithm
    that will be evaluated in crossplay scenarios.
    
    Args:
        zoo: ZooManager instance
        scenario: Environment/scenario name
        robot_algorithms: List of algorithms to load robot agents from
        
    Returns:
        Dictionary mapping algorithm names to DataFrames of available robots
    """
    robot_dict = {}
    
    for robot_algo in robot_algorithms:
        robots = zoo.index.query(f'algorithm == "{robot_algo}"'
                                ).query(f'scenario == "{scenario}"'
                                ).query('scenario_agent_id == "robot"')
        
        robot_dict[robot_algo] = robots
        print(f"Found {len(robots)} {robot_algo} robot agents")
    
    total_robots = sum(len(robots) for robots in robot_dict.values())
    print(f"Total robot agents to evaluate: {total_robots}")
    
    return robot_dict


def add_batch_dim(x):
    """
    Add a leading batch dimension to an array.
    
    Helper function for preparing parameters for batch evaluation.
    
    Args:
        x: Input array
        
    Returns:
        Array with added leading dimension
    """
    return jnp.expand_dims(x, axis=0)


def create_batch_network_states(batch_uuids: List[str], network: Any, zoo_path: str) -> List[EvalNetworkState]:
    """
    Create a batch of network states for parallel evaluation.
    
    Loads parameters for each agent UUID and creates EvalNetworkState objects
    that can be used for batch evaluation.
    
    Args:
        batch_uuids: List of agent UUIDs to load
        network: Network architecture instance
        zoo_path: Path to zoo directory containing agent parameters
        
    Returns:
        List of EvalNetworkState objects for batch evaluation
    """
    batch_states = []
    
    for agent_uuid in batch_uuids:
        # Load parameters for this agent
        param_path = osp.join(zoo_path, "params", agent_uuid + ".safetensors")
        agent_params = unflatten_dict(
            safetensors.flax.load_file(param_path),
            sep='/'
        )
        
        # Add batch dimension for compatibility
        agent_params = jax.tree_util.tree_map(add_batch_dim, agent_params)

        # Create network state
        eval_network_state = EvalNetworkState(
            apply_fn=network.apply,
            params=agent_params,
        )
        batch_states.append(eval_network_state)
    
    return batch_states


def run_crossplay_evaluation(
    algorithm: str,
    robot_agents: pd.DataFrame,
    alg_funcs: Dict,
    robo_configs: Dict,
    load_zoo_dict: Dict,
    config: Dict,
    eval_rng: Any
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Run crossplay evaluation for a specific algorithm.
    
    Evaluates all robot agents from the specified algorithm against
    all available partner types in a memory-efficient manner.
    
    Args:
        algorithm: Name of the algorithm being evaluated
        robot_agents: DataFrame of robot agents to evaluate
        alg_funcs: Dictionary of algorithm functions
        robo_configs: Dictionary of robot configurations
        load_zoo_dict: Dictionary of partner zoo configurations
        config: Main configuration dictionary
        eval_rng: Random number generator key
        
    Returns:
        Tuple of (returns_dict, opponent_info_dict) for this algorithm
    """
    print(f"\nEvaluating {algorithm} agents...")
    
    inner_returns_dict = {}
    inner_opponent_info = {}
    
    # Get algorithm-specific components
    parallel_batch_size = config.get("PARALLEL_BATCH_SIZE", 1)
    agent_uuids = list(robot_agents.agent_uuid)
    
    # Create network for this algorithm
    network = alg_funcs[algorithm]["NetworkArch"](config=robo_configs[algorithm])
    
    # Set up evaluation environment and function
    eval_env, run_eval = alg_funcs[algorithm]["make_evaluation"](
        robo_configs[algorithm], 
        load_zoo=load_zoo_dict, 
        crossplay=True
    )
    
    # Configure logging based on algorithm type
    if algorithm == "MASAC":
        eval_log_config = alg_funcs[algorithm]["EvalInfoLogConfig"](
            env_state=False, done=True, action=False, reward=True,
            log_prob=False, obs=False, info=True, avail_actions=False,
        )
    else:
        eval_log_config = alg_funcs[algorithm]["EvalInfoLogConfig"](
            env_state=False, done=True, action=False, value=False, reward=True,
            log_prob=False, obs=False, info=True, avail_actions=False,
        )
    
    # Create JIT compiled evaluation functions
    eval_jit = jax.jit(run_eval, static_argnames=["log_eval_info"])
    eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
    
    # Process agents in batches for memory efficiency
    batches = [agent_uuids[i:i + parallel_batch_size] 
               for i in range(0, len(agent_uuids), parallel_batch_size)]
    
    print(f"Processing {len(agent_uuids)} agents in {len(batches)} batches...")
    
    for batch_idx, batch in enumerate(tqdm(batches, desc=f"{algorithm} batches")):
        # Create batch of network states
        batch_network_states = create_batch_network_states(
            batch, network, config["ZOO_PATH"]
        )
        
        # Stack parameters for vmapping
        stacked_params = jax.tree.map(
            lambda *p: jnp.expand_dims(jnp.stack(p), axis=0),
            *[state.params for state in batch_network_states]
        )
        
        # Prepare evaluation
        num_humans = sum(len(partners) for partners in load_zoo_dict.values())
        episode_rngs = jax.random.split(eval_rng, num_humans)
        
        # Memory-efficient evaluation function
        def eval_mem_efficient():
            eval_network_state = EvalNetworkState(
                apply_fn=network.apply, 
                params=stacked_params
            )
            
            # Get batch dimensions for reshaping
            batch_dims = jax.tree.leaves(_tree_shape(stacked_params["params"]))[:2]
            
            # Split for sequential evaluation if needed
            split_trainstate = _flatten_and_split_trainstate(eval_network_state)
            
            # Run evaluation in chunks
            evals = _concat_tree([
                eval_vmap(episode_rngs, ts, eval_log_config)
                for ts in split_trainstate
            ])
            
            # Reshape results back to original batch structure
            evals = jax.tree.map(
                lambda x: x.reshape((*batch_dims, *x.shape[1:])),
                evals
            )
            
            return evals
        
        # Execute evaluation
        batch_evals = jax.jit(eval_mem_efficient)()
        
        # Process results for each agent in the batch
        for i, agent_uuid in enumerate(batch):
            # Extract evaluation results for this agent
            if isinstance(batch_evals, list):
                agent_evals = batch_evals[i]
            else:
                agent_evals = jax.tree.map(
                    lambda x: x[i] if hasattr(x, '__getitem__') and i < len(x) else x,
                    batch_evals
                )
            
            # Compute episode returns
            episode_returns = _compute_episode_returns(agent_evals)
            mean_returns = episode_returns["__all__"].mean(axis=-1)
            
            # Extract opponent information
            opponent_uuids = extract_uuids_from_eval_results(eval_env, agent_evals)
            
            # Store results
            inner_returns_dict[agent_uuid] = mean_returns
            inner_opponent_info[agent_uuid] = opponent_uuids
    
    print(f"✓ Completed {algorithm} evaluation")
    return inner_returns_dict, inner_opponent_info


# ================================ MAIN CROSSPLAY ORCHESTRATION ================================

@hydra.main(version_base=None, config_path="config", config_name="crossplay_zoo")
def main(config):
    """
    Main orchestration function for multi-algorithm crossplay evaluation.
    
    This function implements comprehensive crossplay evaluation across multiple
    RL algorithms to understand inter-algorithm compatibility and generalization.
    
    The evaluation process:
    1. Loads algorithm functions and configurations
    2. Sets up partner and robot agent pools from zoo
    3. Runs systematic crossplay evaluation
    4. Computes performance matrices and saves results
    
    Args:
        config: Hydra configuration object containing crossplay parameters
    """
    config = OmegaConf.to_container(config, resolve=True)
    
    print("="*70)
    print("MULTI-ALGORITHM CROSSPLAY EVALUATION")
    print("="*70)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Zoo path: {config['ZOO_PATH']}")
    print(f"Robot algorithms: {config['crossplay']['robot_algos']}")
    print(f"Partner algorithms: {config['PARTNER_ALGORITHMS']}")
    print(f"Parallel batch size: {config.get('PARALLEL_BATCH_SIZE', 1)}")
    
    # ===== ALGORITHM LOADING =====
    print("\nLoading algorithm functions...")
    alg_funcs = load_algorithm_functions(config)
    
    # Load algorithm configurations
    print("Loading algorithm configurations...")
    robo_configs = {}
    for alg, paths in config["crossplay"]["algo_configs"].items():
        robo_configs[alg] = load_and_merge_algo_config(paths)
        print(f"✓ Loaded {alg} configuration")
    
    # ===== RANDOM NUMBER GENERATOR SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    
    # ===== ZOO SETUP =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        print("\nSetting up zoo and agent pools...")
        zoo = ZooManager(config["ZOO_PATH"])
        scenario = config["ENV_NAME"]
        
        # Set up partner agents (e.g., "human" behavioral policies)
        partner_dict = setup_partner_zoo(zoo, scenario, config["PARTNER_ALGORITHMS"])
        
        # Create zoo loading configuration for partners
        load_zoo_dict = {
            algo: {"human": list(partner_dict[algo].agent_uuid)} 
            for algo in partner_dict.keys()
        }
        
        # Set up robot agents to be evaluated
        robot_dict = setup_robot_agents(zoo, scenario, config["crossplay"]["robot_algos"])
        
        # ===== CROSSPLAY EVALUATION EXECUTION =====
        print("\nStarting crossplay evaluation...")
        returns_dict = {}
        opponent_info_dict = {}
        
        # Evaluate each robot algorithm against all partner types
        for alg, robot_agents in robot_dict.items():
            if len(robot_agents) == 0:
                print(f"⚠ No robot agents found for {alg}, skipping...")
                continue
            
            # Run crossplay evaluation for this algorithm
            inner_returns_dict, inner_opponent_info = run_crossplay_evaluation(
                algorithm=alg,
                robot_agents=robot_agents,
                alg_funcs=alg_funcs,
                robo_configs=robo_configs,
                load_zoo_dict=load_zoo_dict,
                config=config,
                eval_rng=eval_rng
            )
            
            returns_dict[alg] = inner_returns_dict
            opponent_info_dict[alg] = inner_opponent_info
        
        # ===== SAVE RESULTS =====
        print("\nSaving crossplay evaluation results...")
        jnp.save("crossplay_results.npy", returns_dict, allow_pickle=True)
        jnp.save("crossplay_opponent_info.npy", opponent_info_dict, allow_pickle=True)
        
        # ===== DISPLAY RESULTS SUMMARY =====
        print("\n" + "="*70)
        print("CROSSPLAY EVALUATION RESULTS SUMMARY")
        print("="*70)
        
        # Compute summary statistics
        total_evaluations = 0
        algorithm_stats = {}
        
        for alg, results in returns_dict.items():
            num_agents = len(results)
            if num_agents > 0:
                all_returns = jnp.concatenate([returns for returns in results.values()])
                mean_performance = float(all_returns.mean())
                std_performance = float(all_returns.std())
                
                algorithm_stats[alg] = {
                    "num_agents": num_agents,
                    "mean_performance": mean_performance,
                    "std_performance": std_performance
                }
                total_evaluations += num_agents
                
        
        # Cross-algorithm comparison
        if len(algorithm_stats) > 1:
            print("Cross-Algorithm Performance Comparison:")
            sorted_algs = sorted(algorithm_stats.items(), 
                               key=lambda x: x[1]["mean_performance"], reverse=True)
            
            for rank, (alg, stats) in enumerate(sorted_algs, 1):
                print(f"  {rank}. {alg}: {stats['mean_performance']:.2f} ± {stats['std_performance']:.2f}")
            
            best_alg = sorted_algs[0][0]
            worst_alg = sorted_algs[-1][0]
            performance_gap = (sorted_algs[0][1]["mean_performance"] - 
                             sorted_algs[-1][1]["mean_performance"])
            
            print(f"\nKey Findings:")
            print(f"  🏆 Best performing algorithm: {best_alg}")
            print(f"  📉 Largest performance gap: {performance_gap:.2f}")
            print(f"  🔄 Total crossplay evaluations: {total_evaluations}")
        
        print("Multi-algorithm crossplay evaluation completed successfully!")
        
    return returns_dict

    # TODO: Add rendering functionality to visualize crossplay 

if __name__ == "__main__":
    main()

