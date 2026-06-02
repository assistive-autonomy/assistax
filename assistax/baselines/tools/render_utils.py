"""This is a utility file for rendering already trained policies (mainly since rendering fails due to memory constraints)"""
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
from assistax.baselines.tree_utils import (
    _tree_take, _unstack_tree, _tree_shape, _stack_tree,
)
from assistax.baselines.utils import (
    _take_episode, _compute_episode_returns, _concat_tree, _tree_split,
)
from flax import struct
import argparse


@struct.dataclass
class EvalNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict


# Tree/episode helpers are imported above from tree_utils / utils (no local copies).


def render_episodes(path: str):
    """
    Render episodes from trained policies.
    
    Args:
        path: Path to the directory containing .safetensor files and .hydra/config.yaml
    """
    # Ensure path doesn't end with trailing slash for consistency
    path = path.rstrip('/')
    
    config_path = os.path.join(path, ".hydra", "config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = OmegaConf.to_container(
        OmegaConf.load(config_path), resolve=True
    )
    
    # Dynamically import the correct module based on config
    if config["ALG"] == "IPPO":
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from assistax.baselines.IPPO.ippo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
                from assistax.baselines.IPPO.ippo_ff_nps import MultiActorCritic as NetworkArch
                print("Using: Feedforward Networks with No Parameter Sharing")
            case (False, True):
                from assistax.baselines.IPPO.ippo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig
                from assistax.baselines.IPPO.ippo_ff_ps import ActorCritic as NetworkArch
                print("Using: Feedforward Networks with Parameter Sharing")
            case (True, False):
                from assistax.baselines.IPPO.ippo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig, NetworkArch
                from assistax.baselines.IPPO.ippo_rnn_nps import MultiActorCriticRNN as NetworkArch
                print("Using: Recurrent Networks with No Parameter Sharing")
            case (True, True):
                from assistax.baselines.IPPO.ippo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig, NetworkArch
                from assistax.baselines.IPPO.ippo_rnn_ps import ActorCriticRNN as NetworkArch
                print("Using: Recurrent Networks with Parameter Sharing")

    elif config["ALG"] == "MAPPO":
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from assistax.baselines.MAPPO.mappo_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
                from assistax.baselines.MAPPO.mappo_ff_nps import MultiActor as NetworkArch
                print("Using: Feedforward Networks with No Parameter Sharing")
            case (False, True):
                from assistax.baselines.MAPPO.mappo_ff_ps import make_train, make_evaluation, EvalInfoLogConfig 
                from assistax.baselines.MAPPO.mappo_ff_ps import Actor as NetworkArch
                print("Using: Feedforward Networks with Parameter Sharing")
            case (True, False):
                from assistax.baselines.MAPPO.mappo_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig
                from assistax.baselines.MAPPO.mappo_rnn_nps import MultiActorRNN as NetworkArch
                print("Using: Recurrent Networks with No Parameter Sharing")
            case (True, True):
                from assistax.baselines.MAPPO.mappo_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig
                from assistax.baselines.MAPPO.mappo_rnn_ps import ActorRNN as NetworkArch
                print("Using: Recurrent Networks with Parameter Sharing")

    elif config["ALG"] == "MASAC":
        match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
            case (False, False):
                from assistax.baselines.MASAC.masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig, NetworkArch
                print("Using: Feedforward Networks with No Parameter Sharing")
           # case (False, True):
           #     from assistax.baselines.MASAC.masac_ff_ps import make_train, make_evaluation, EvalInfoLogConfig, NetworkArch
           #     print("Using: Feedforward Networks with Parameter Sharing")
           # case (True, False):
           #     from assistax.baselines.MASAC.masac_rnn_nps import make_train, make_evaluation, EvalInfoLogConfig, NetworkArch
           #     print("Using: Recurrent Networks with No Parameter Sharing")
           # case (True, True):
           #     from assistax.baselines.MASAC.masac_rnn_ps import make_train, make_evaluation, EvalInfoLogConfig, NetworkArch
           #     print("Using: Recurrent Networks with Parameter Sharing")
    else:
        raise ValueError(f"Unknown algorithm: {config['ALG']}")
    
    # Set number of evaluation episodes
    config["NUM_EVAL_EPISODES"] = 3 
    rng = jax.random.PRNGKey(config["SEED"])
    rng, eval_rng = jax.random.split(rng)
    
    print(f"Loading parameters from {path}")
    
    # TODO: make this general to cases where the params are called differently e.g. robot1 and robot2
    final_params_name = "final_params.safetensors" if config["network"]["agent_param_sharing"] else "all_params.safetensors"
    all_params_path = os.path.join(path, final_params_name) 
    if not config["network"]["agent_param_sharing"] and config["ENV_NAME"] not in ["pushcoop", "handover"]:
        human_path = os.path.join(path, "human.safetensors")
        robot_path = os.path.join(path, "robot.safetensors")
    
        if not os.path.exists(human_path):
            raise FileNotFoundError(f"Human parameters not found at {human_path}")
        if not os.path.exists(robot_path):
            raise FileNotFoundError(f"Robot parameters not found at {robot_path}")
    
    elif not config["network"]["agent_param_sharing"] and config["ENV_NAME"] in ["pushcoop", "handover"]:
        robot1_path = os.path.join(path, "robot1.safetensors")
        robot2_path = os.path.join(path, "robot2.safetensors")
    
        if not os.path.exists(robot1_path):
            raise FileNotFoundError(f"Robot1 parameters not found at {robot1_path}")
        if not os.path.exists(robot2_path):
            raise FileNotFoundError(f"Robot2 parameters not found at {robot2_path}")

    if not os.path.exists(all_params_path):
        raise FileNotFoundError(f"All parameters not found at {all_params_path}")

    with jax.disable_jit(config.get("DISABLE_JIT", False)):
        
        if config["network"]["agent_param_sharing"]:
            agent_params = unflatten_dict(
                safetensors.flax.load_file(all_params_path), sep='/'
        )
    
        else:
                    
            if config["ENV_NAME"] in ["pushcoop", "handover"]: # TODO double check the naming here

                robot1_params = unflatten_dict(
                    safetensors.flax.load_file(robot1_path), sep='/'
                )
                robot2_params = unflatten_dict(
                    safetensors.flax.load_file(robot2_path), sep='/'
                )
                agent_params = {'robot1': robot1_params, 'robot2': robot2_params}
           
            else:
                
                human_params = unflatten_dict(
                safetensors.flax.load_file(human_path), sep='/'
                )
                robot_params = unflatten_dict(
                    safetensors.flax.load_file(robot_path), sep='/'
                )
                agent_params = {'human': human_params, 'robot': robot_params}
        
        # Create evaluation environment
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
        
        # Prepare network state
        if config["ENV_NAME"] in ["scratchithch", "bedbathing", "armmanipulation"] and not config["network"]["agent_param_sharing"]:
            robot = _tree_take(agent_params["robot"], 0, axis=0)
            human = _tree_take(agent_params["human"], 0, axis=0)
            final_eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=_stack_tree([robot, human]),
            )
        elif config["network"]["agent_param_sharing"]:
            agent_params = _tree_take(agent_params, 0, axis=0)
            final_eval_network_state = EvalNetworkState(
                apply_fn=network.apply,
                params=agent_params,
            )

        elif config["ENV_NAME"] in ["pushcoop", "handover"]: # TODO double check the naming here
           robot1 = _tree_take(agent_params["robot1"], 0, axis=0)
           robot2 = _tree_take(agent_params["robot2"], 0, axis=0)
           final_eval_network_state = EvalNetworkState(
               apply_fn=network.apply,
               params=_stack_tree([robot1, robot2]),
           )
        print("Running evaluation...")
        # Run evaluation
        eval_final = eval_jit(eval_rng, final_eval_network_state, eval_log_config)
        
        # Extract episodes
        first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        first_episode_rewards = eval_final.reward["__all__"] * (1 - first_episode_done)
        first_episode_returns = first_episode_rewards.sum(axis=0)
        episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        
        worst_idx = episode_argsort.take(0, axis=-1)
        best_idx = episode_argsort.take(-1, axis=-1)
        median_idx = episode_argsort.take(episode_argsort.shape[-1] // 2, axis=-1)
        
        print(f"Episode returns - Worst: {first_episode_returns[worst_idx]:.2f}, "
              f"Median: {first_episode_returns[median_idx]:.2f}, "
              f"Best: {first_episode_returns[best_idx]:.2f}")
        
        from brax.io import html
        
        # Extract episodes
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
        
        # Render and save HTML files
        print("Rendering episodes...")
        
        worst_html = html.render(
            eval_env.env.sys.tree_replace({'opt.timestep': eval_env.env.dt}),
            worst_episode
        )
        worst_html_path = os.path.join(path, "worst_episode.html")
        with open(worst_html_path, 'w') as f:
            f.write(worst_html)
        print(f"Saved worst episode to {worst_html_path}")
        
        median_html = html.render(
            eval_env.env.sys.tree_replace({'opt.timestep': eval_env.env.dt}),
            median_episode
        )
        median_html_path = os.path.join(path, "median_episode.html")
        with open(median_html_path, 'w') as f:
            f.write(median_html)
        print(f"Saved median episode to {median_html_path}")
        
        best_html = html.render(
            eval_env.env.sys.tree_replace({'opt.timestep': eval_env.env.dt}),
            best_episode
        )
        best_html_path = os.path.join(path, "best_episode.html")
        with open(best_html_path, 'w') as f:
            f.write(best_html)
        print(f"Saved best episode to {best_html_path}")
        
        print(f"\nAll renders saved successfully to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render trained policy episodes from assistax training runs"
    )
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Path to the directory containing .safetensor files and .hydra/config.yaml"
    )
    
    args = parser.parse_args()
    render_episodes(args.path)


if __name__ == "__main__":
    main()