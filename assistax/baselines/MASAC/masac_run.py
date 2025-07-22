import os
import sys
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


@hydra.main(version_base=None, config_path="config", config_name="masac_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
        
    from masac_ff_nps import make_train, make_evaluation, EvalInfoLogConfig
    from masac_ff_nps import MultiSACActor as NetworkArch

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])

    print(f"Starting training with {config['TOTAL_TIMESTEPS']} timesteps")
    print(f"Num environments: {config['NUM_ENVS']}")
    print(f"Num seeds: {config['NUM_SEEDS']}")
    print(f"Environment: {config['ENV_NAME']}")

    with jax.disable_jit(config["DISABLE_JIT"]):
        
        train_jit = jax.jit(
            make_train(config, save_train_state=True, load_zoo=False),
            device=jax.devices()[config["DEVICE"]]
        )
        # Execute training across all seeds (includes JIT compilation on first run)
        print("Running training...")
        out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
            train_rngs,
            config["POLICY_LR"], config["Q_LR"], config["ALPHA_LR"], config["TAU"]
        )

        # ===== SAVE TRAINING METRICS =====
        print("Saving training metrics...")
        EXCLUDED_METRICS = ["actor_train_state", "q1_train_state", "q2_train_state"]
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
        }, allow_pickle=True)

        # ===== SAVE MODEL PARAMETERS =====
        print("Saving model parameters...")
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        all_train_states_actor = out["metrics"]["actor_train_state"]
        all_train_states_q1 = out["metrics"]["q1_train_state"]
        all_train_states_q2 = out["metrics"]["q2_train_state"]
        final_train_state_actor = out["runner_state"].train_states.actor
        final_train_state_q1 = out["runner_state"].train_states.q1
        final_train_state_q2 = out["runner_state"].train_states.q2

        # Save all training states (for analysis across training)
        actor_all_path = "actor_all_params.safetensors"
        safetensors.flax.save_file(
            flatten_dict(all_train_states_actor.params, sep='/'),
            actor_all_path
        )
        actor_all_path = os.path.abspath(actor_all_path) # getting this to run render later
        safetensors.flax.save_file(
            flatten_dict(all_train_states_q1.params, sep='/'),
            "q1_all_params.safetensors"
        )
        safetensors.flax.save_file(
            flatten_dict(all_train_states_q2.params, sep='/'),
            "q2_all_params.safetensors"
        )

        # Save final parameters
        if config["network"]["agent_param_sharing"]:
            # For parameter sharing: single set of shared parameters
            safetensors.flax.save_file(
                flatten_dict(final_train_state_actor.params, sep='/'),
                "actor_final_params.safetensors"
            )

            safetensors.flax.save_file(
                flatten_dict(final_train_state_q1.params, sep='/'),
                "q1_final_params.safetensors"
            )
            safetensors.flax.save_file(
                flatten_dict(final_train_state_q2.params, sep='/'),
                "q2_final_params.safetensors"
            )
        else:
            # For independent parameters: split by agent
            split_actor_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_actor.params)
            )
            for agent, params in zip(env.agents, split_actor_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"actor_{agent}.safetensors",
                )

            split_q1_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_q1.params)
            )
            for agent, params in zip(env.agents, split_q1_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q1_{agent}.safetensors",
                )

            split_q2_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0, 1), final_train_state_q2.params)
            )
            for agent, params in zip(env.agents, split_q2_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q2_{agent}.safetensors",
                )

        # ===== EVALUATION SETUP =====
        print("Setting up evaluation...")
        
        # Calculate evaluation batching for memory efficiency
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states_actor.params))[:2]
        n_sequential_evals = int(jnp.ceil(
            config["NUM_EVAL_EPISODES"] * jnp.prod(jnp.array(batch_dims))
            / config["GPU_ENV_CAPACITY"]
        ))
        
        def _flatten_and_split_trainstate(trainstate):
            """Flatten and split training states for sequential evaluation."""
            flat_trainstate = jax.tree.map(
                lambda x: x.reshape((x.shape[0] * x.shape[1], *x.shape[2:])),
                trainstate
            )
            return _tree_split(flat_trainstate, n_sequential_evals)

        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states_actor)
        
        # ===== EVALUATION EXECUTION =====
        print("Running evaluation...")
        eval_env, run_eval = make_evaluation(config)
        
        # Configure evaluation logging
        eval_log_config = EvalInfoLogConfig(
            env_state=False,
            done=True,
            action=False,
            reward=True,
            log_prob=False,
            obs=False,
            info=False,
            avail_actions=False,
        )
        
        # JIT compile evaluation functions
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
        # Run evaluation in batches for memory efficiency
        evals = _concat_tree([
            eval_vmap(eval_rng, ts, eval_log_config)
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        
        # Reshape evaluation results back to original batch structure
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # ===== COMPUTE PERFORMANCE METRICS =====
        print("Computing performance metrics...")
        first_episode_returns = _compute_episode_returns(evals)
        first_episode_returns = first_episode_returns["__all__"]
        mean_episode_returns = first_episode_returns.mean(axis=-1)

        # ===== SAVE EVALUATION RESULTS =====
        print("Saving evaluation results...")
        jnp.save("returns.npy", mean_episode_returns)
        
        print(f"Mean episode return: {mean_episode_returns.mean():.2f} ± {mean_episode_returns.std():.2f}")
        print("Training and evaluation completed successfully!")

        if config["VIZ_POLICY"]:
            
            actor_all_path = actor_all_path 
            current_output_dir = os.getcwd()
            script_directory = os.path.dirname(os.path.abspath(__file__))
            render_script_path = os.path.join(script_directory, "render_masac.py") # Because hydra changes the dir
            breakpoint()
            os.execv(sys.executable, 
                    [sys.executable,
                    render_script_path,
                    f'eval.path={actor_all_path}',
                    f'hydra.run.dir={current_output_dir}',
                    f'NUM_EVAL_EPISODES={config["N_RENDER_EPISODES"]}',
                    ])

if __name__ == "__main__":
    main()