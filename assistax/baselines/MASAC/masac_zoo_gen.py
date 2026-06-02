"""
MASAC Zoo Generation Script

This module trains Multi-Agent Soft Actor-Critic (MASAC) agents and saves them to a "zoo" 
for later use in mixed training scenarios. The zoo is a collection of trained agent policies
that can be loaded and used as training partners, enabling diverse multi-agent interactions
and robust policy development.

Usage:
    python masac_zoo_gen.py [hydra options] ZOO_PATH=path/to/zoo
    
The script will train agents and automatically save them to the zoo with proper
organization by agent ID and training seed.
"""

import jax
import jax.numpy as jnp
import hydra
import assistax
from tqdm import tqdm
from omegaconf import OmegaConf
from assistax.wrappers.aht import ZooManager
from typing import Dict, Any
from assistax.baselines.utils import (
    _tree_take, _unstack_tree, _take_episode,
    _tree_shape, _stack_tree, _concat_tree, _tree_split,
    generate_preference_configs, extract_pref_config_at_index,
    )
import copy
import uuid


def _compute_episode_returns(eval_info, common_reward=False, time_axis=-2):
    """
    Compute undiscounted episode returns from evaluation information.
    
    Handles episode boundaries correctly by resetting cumulative rewards
    when episodes end and start new ones. Supports both individual agent
    rewards and common team rewards.
    
    Args:
        eval_info: Evaluation information containing rewards and done flags
        common_reward: Whether to treat rewards as shared across agents
        time_axis: Axis representing time dimension (default: -2)
        
    Returns:
        Dictionary of undiscounted returns per agent (and "__all__" for total)
    """
    done_arr = eval_info.done["__all__"]
    
    # Create mask for episode boundaries
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    
    # Sum rewards within episodes only
    undiscounted_returns = jax.tree.map(
        lambda r: (r * (1 - episode_done)).sum(axis=time_axis),
        eval_info.reward
    )
    
    # Add total reward if not present
    if "__all__" not in undiscounted_returns:
        undiscounted_returns.update({
            "__all__": (sum(undiscounted_returns.values())
                       / (len(undiscounted_returns) if common_reward else 1))
        })
    
    return undiscounted_returns


# ================================ HYPERPARAMETER SWEEP UTILITIES ================================

def _generate_sweep_axes(rng, config):
    """
    Generate hyperparameter configurations for sweep.
    
    Creates random samples for each hyperparameter marked for sweeping in the
    configuration. Uses log-uniform sampling for learning rates and tau to
    cover multiple orders of magnitude effectively.
    
    Note: This function is included for compatibility but not used in zoo generation.
    
    Args:
        rng: Random number generator key
        config: Configuration dictionary containing sweep specifications
        
    Returns:
        Dictionary containing sampled values and vmap axes for each hyperparameter
    """
    p_lr_rng, q_lr_rng, alpha_lr_rng, tau_rng = jax.random.split(rng, 4)
    sweep_config = config["SWEEP"]
    
    # ===== POLICY LEARNING RATE SWEEP =====
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

    # ===== Q-NETWORK LEARNING RATE SWEEP =====
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

    # ===== TEMPERATURE LEARNING RATE SWEEP =====
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

    # ===== TAU (SOFT UPDATE) SWEEP =====
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
        "tau": {"val": taus, "axis": tau_axis},
    }


# ================================ MAIN ZOO GENERATION ORCHESTRATION ================================

@hydra.main(version_base=None, config_path="config", config_name="masac_zoo_gen")
def main(config):
    """
    Main orchestration function for MASAC zoo generation.
    
    This function:
    1. Trains MASAC agents across multiple random seeds
    2. Extracts individual agent parameters from trained models
    3. Saves each agent to the zoo with proper metadata and organization
    4. Creates a diverse collection of trained agents for future use
    
    The zoo enables training against diverse partners, curriculum learning,
    and robust policy evaluation in multi-agent environments.
    
    Args:
        config: Hydra configuration object containing training and zoo parameters
    """
    config = OmegaConf.to_container(config, resolve=True)

    pref_sweep_config = config.get("PREFERENCE_SWEEP", None)
    use_dynamic_prefs = pref_sweep_config is not None

    print(f"Starting MASAC zoo generation")
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Number of seeds: {config['NUM_SEEDS']}")
    print(f"Zoo path: {config['ZOO_PATH']}")
    print(f"Total timesteps per agent: {config['TOTAL_TIMESTEPS']}")
    if use_dynamic_prefs:
        print(f"Preference sweep: {pref_sweep_config['num_configs']} configs")

    # ===== IMPORT ALGORITHM COMPONENTS =====
    from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig

    # ===== ENVIRONMENT SETUP =====
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    print(f"Environment agents: {env.agents}")

    # ===== RANDOM NUMBER GENERATOR SETUP =====
    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng, pref_rng = jax.random.split(rng, 3)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])

    # ===== TRAINING EXECUTION =====
    with jax.disable_jit(config["DISABLE_JIT"]):
        if use_dynamic_prefs:
            # === PREFERENCE SWEEP PATH ===
            pref_configs = generate_preference_configs(pref_rng, pref_sweep_config, config)
            print(f"Sampled {pref_sweep_config['num_configs']} preference configs")

            print("Compiling training function (dynamic preferences)...")
            train_jit = jax.jit(
                make_train(config, save_train_state=False, dynamic_preferences=True),
                device=jax.devices()[config["DEVICE"]]
            )

            # Nested vmap: outer=preference configs, inner=seeds
            out = jax.vmap(
                jax.vmap(train_jit, in_axes=(0, None, None, None, None, None)),  # seeds
                in_axes=(None, None, None, None, None, 0),  # pref configs
            )(train_rngs, config["POLICY_LR"], config["Q_LR"],
              config["ALPHA_LR"], config["TAU"], pref_configs)

            # Shape: (num_pref_configs, num_seeds, num_agents, ...)
            final_train_state = out["runner_state"].train_states.actor.params
            print(f"Training completed! Final parameters shape: {jax.tree.leaves(_tree_shape(final_train_state))[0]}")

            # Three-level extraction and per-agent saving
            print("Setting up zoo management...")
            zoo = ZooManager(config["ZOO_PATH"])
            total_agents_saved = 0

            for pref_idx in range(pref_sweep_config["num_configs"]):
                agent_config = copy.deepcopy(config)
                pref_at_idx = extract_pref_config_at_index(pref_configs, pref_idx)
                agent_config["ENV_KWARGS"]["preference_rewards"]["preference_weights"] = {
                    "speed_preference": pref_at_idx["w_speed"],
                    "force_preference": pref_at_idx["w_force"],
                    "touch_penalty": pref_at_idx["w_touch"],
                }
                agent_config["ENV_KWARGS"]["preference_rewards"]["preference_ranges"] = {
                    "speed_range": [pref_at_idx["speed_range_min"], pref_at_idx["speed_range_max"]],
                    "force_range": [pref_at_idx["force_range_min"], pref_at_idx["force_range_max"]],
                }
                print(f"Pref config {pref_idx}: w_speed={pref_at_idx['w_speed']:.3f}, "
                      f"w_force={pref_at_idx['w_force']:.3f}")

                pref_weights_for_index = {
                    "w_speed": round(pref_at_idx["w_speed"], 4),
                    "w_force": round(pref_at_idx["w_force"], 4),
                    "w_touch": round(pref_at_idx["w_touch"], 4),
                }

                for seed_idx in range(config["NUM_SEEDS"]):
                    team_uuid = str(uuid.uuid4())
                    for agent_idx, agent_id in enumerate(env.agents):
                        agent_params = _tree_take(
                            _tree_take(
                                _tree_take(final_train_state, pref_idx, axis=0),
                                seed_idx, axis=0,
                            ),
                            agent_idx, axis=0,
                        )
                        zoo.save_agent(
                            config=agent_config,
                            param_dict=agent_params,
                            scenario_agent_id=agent_id,
                            team_uuid=team_uuid,
                            preference_weights=pref_weights_for_index,
                        )
                        total_agents_saved += 1

        else:
            # === ORIGINAL PATH (unchanged) ===
            print("Compiling training function...")
            train_jit = jax.jit(
                make_train(config, save_train_state=False),
                device=jax.devices()[config["DEVICE"]]
            )

            out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
                train_rngs,
                config["POLICY_LR"],
                config["Q_LR"],
                config["ALPHA_LR"],
                config["TAU"],
            )

            final_train_state = out["runner_state"].train_states.actor.params
            print(f"Training completed! Final parameters shape: {jax.tree.leaves(_tree_shape(final_train_state))[0]}")

            print("Setting up zoo management...")
            zoo = ZooManager(config["ZOO_PATH"])
            total_agents_saved = 0

            for seed_idx in range(config["NUM_SEEDS"]):
                team_uuid = str(uuid.uuid4())
                for agent_idx, agent_id in enumerate(env.agents):
                    agent_params = _tree_take(
                        _tree_take(final_train_state, seed_idx, axis=0),
                        agent_idx, axis=0,
                    )
                    zoo.save_agent(
                        config=config,
                        param_dict=agent_params,
                        scenario_agent_id=agent_id,
                        team_uuid=team_uuid,
                    )
                    total_agents_saved += 1

        # ===== DISPLAY COMPLETION SUMMARY =====
        print("\n" + "="*60)
        print("ZOO GENERATION COMPLETED")
        print("="*60)
        print(f"Total agents saved to zoo: {total_agents_saved}")
        print(f"Agent types: {', '.join(env.agents)}")
        print(f"Seeds per type: {config['NUM_SEEDS']}")
        if use_dynamic_prefs:
            print(f"Preference configs: {pref_sweep_config['num_configs']}")
        print(f"Zoo location: {config['ZOO_PATH']}")
        print("="*60)


if __name__ == "__main__":
    main()
