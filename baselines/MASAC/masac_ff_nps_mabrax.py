import os
# os.environ["XLA_FLAGS"] = (
#     "--xla_gpu_enable_triton_softmax_fusion=true "
#     "--xla_gpu_triton_gemm_any=true "
#     "--xla_dump_to=xla_dump "
# )
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false" # TODO: get rid of these 
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.95"
import jax
import jax.numpy as jnp
import flax.linen as nn
from tqdm import tqdm
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax import struct
import optax
import distrax
import sys
import numpy as np
import jaxmarl
# from jaxmarl.distributions.tanh_distribution import TanhTransformedDistribution # try new distribution
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper, LogCrossplayWrapper
from jaxmarl.wrappers.aht_all import ZooManager, LoadAgentWrapper, LoadEvalAgentWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, TypeAlias, Any, Dict, Optional
import os
import functools
from functools import partial
# import tensorflow_probability.substrates.jax.distributions as tfd
from flax.core.scope import FrozenVariableDict
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
import flashbax as fbx
import safetensors.flax
from flax.traverse_util import flatten_dict
import sys



# Helper functions remain the same
def _tree_take(pytree, indices, axis=None):
    return jax.tree_util.tree_map(lambda x: x.take(indices, axis=axis), pytree)

def _tree_shape(pytree):
    return jax.tree_util.tree_map(lambda x: x.shape, pytree)

def _unstack_tree(pytree):
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = list(zip(*leaves))
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

def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """Convert dict of arrays to batched array."""
    return jnp.stack(tuple(qty[a] for a in agents))

def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array to dict of arrays."""
    return dict(zip(agents, qty))

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
    
@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiSACActor(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        obs, done, avail_actions = x
        # actor Network
        actor_hidden = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(obs)
        actor_hidden = activation(actor_hidden)
        actor_hidden = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_hidden)
        actor_hidden = activation(actor_hidden)
        
        # output mean
        actor_mean = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_hidden)
        
        # log std
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.config["ACT_DIM"],)
        )
        actor_log_std = jnp.broadcast_to(log_std, actor_mean.shape)

        return actor_mean, jnp.exp(actor_log_std) # could try softplus instead or just return log_std and then do the transformation after 

# remove vmap so that we have a shared crtiric network

class SACQNetwork(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, x, action):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs = x
        x = jnp.concatenate([obs, action], axis=-1)
        
        x = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(x)
        x = activation(x)
        x = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(x)
        
        return jnp.squeeze(x, axis=-1)

class Transition(NamedTuple):
    obs: jnp.ndarray
    obs_global: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    next_obs_global: jnp.ndarray


class SACTrainStates(NamedTuple):
    actor: TrainState
    q1: TrainState
    q2: TrainState
    q1_target: Dict
    q2_target: Dict
    log_alpha: jnp.ndarray
    alpha_opt_state: optax.OptState

BufferState: TypeAlias = TrajectoryBufferState[Transition]

class RunnerState(NamedTuple):
    train_states: SACTrainStates
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    t: int
    buffer_state: BufferState
    rng: jnp.ndarray
    total_env_steps: int  # Add this
    total_grad_updates: int # Add this
    update_t: int
    ag_idx: Optional[int] = None

class EvalState(NamedTuple):
    train_states: SACTrainStates
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    update_step: int
    rng: jnp.ndarray
    ag_idx: Optional[int] = None

class EvalInfo(NamedTuple):
    env_state: LogEnvState
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray
    ag_idx: Optional[jnp.ndarray]

# TODO: add eval info log config to determine what to log for evaluation (improve the memroy efficienty)

# class TransitionNames(NamedTuple):
#     obs: str
#     obs_global: str
#     action: str
#     reward: str
#     done: str
#     next_obs: str
#     next_obs_global: str

@struct.dataclass
class EvalInfoLogConfig:
    env_state: bool = True,
    done: bool = True,
    action: bool = True,
    reward: bool = True,
    log_prob: bool = True,
    obs: bool = True,
    info: bool = True,
    avail_actions: bool = True,
    
def reshape_for_buffer(x, f):
    if f not in ["obs_global", "next_obs_global"]:
        x = x.swapaxes(1,2)
    # x = x.swapaxes(1,2)
    timesteps = x.shape[0]
    num_envs = x.shape[1]
    return x.reshape(timesteps * num_envs, *x.shape[2:])

def flatten_actions(x):
    x = x.swapaxes(0,1)
    n_envs = x.shape[0]
    n_agents = x.shape[1]
    act_dim = x.shape[2]
    return x.reshape(n_envs, n_agents * act_dim)


def make_train(config, save_train_state=True, load_zoo=False): 
    if load_zoo:
        zoo = ZooManager(config["ZOO_PATH"])
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_UPDATES"] = int(jnp.ceil(
        config["TOTAL_TIMESTEPS"] / config["ROLLOUT_LENGTH"] / config["NUM_ENVS"])
    ) # round up to do at least config["TOTAL_TIMESTEPS"]
    config["TOTAL_TIMESTEPS"] = int(config["NUM_UPDATES"] * config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]) # recalculate actual total timesteps
    config["SCAN_STEPS"] = config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"]
    config["EXPLORE_SCAN_STEPS"] = config["EXPLORE_STEPS"] // config["NUM_ENVS"]
    print(f"TOTAL_TIMESTEPS: {config['TOTAL_TIMESTEPS']} \n NUM_UPDATES: {config['NUM_UPDATES']} \n SCAN_STEPS: {config['SCAN_STEPS']} \n EXPLORE_STEPS: {config['EXPLORE_STEPS']} \n NUM_CHECKPOINTS: {config['NUM_CHECKPOINTS']}")
    print(f"Jax Running on: {jax.devices()}")
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
    env = LogWrapper(env, replace_info=True)
    
    def train(rng, p_lr, q_lr, alpha_lr, tau): # TODO: Update these hparams for sweeping sac 
        # Could sweep over alpha if we don't learn it. 
        # Non contineous sweeps 
        # - NUM SAC UPDATES
        # - Batch Size 
        # - Buffer Size 
        # - Alpha when not using learning

        actor = MultiSACActor(config=config)
        q = SACQNetwork(config=config)

        rng, actor_rng, q1_rng, q2_rng = jax.random.split(rng, num=4)

        init_x = (
            jnp.zeros( # obs
                (env.num_agents, 1, config["OBS_DIM"])
            ),
            jnp.zeros( # done
                (env.num_agents, 1)
            ),
            jnp.zeros( # avail_actions
                (env.num_agents, 1, config["ACT_DIM"])
            ),
        )
        init_x_q = jnp.zeros((1, config["GOBS_DIM"]))
        
        actor_params = actor.init(actor_rng, init_x)
        dummy_action = jnp.zeros((env.num_agents, 1, config["ACT_DIM"]))
        dummy_action = flatten_actions(dummy_action)
        q1_params = q.init(q1_rng, init_x_q, dummy_action)
        q2_params = q.init(q2_rng, init_x_q, dummy_action)

        rng, env_rng = jax.random.split(rng)
        reset_rng = jax.random.split(env_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_dones = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool)

        action_space = env.action_space(env.agents[0])
        action = jnp.zeros((env.num_agents, config["NUM_ENVS"], action_space.shape[0]))


        init_transition = Transition(
            obs=jnp.zeros((env.num_agents, get_space_dim(env.observation_space(env.agents[0]))), dtype=float),
            obs_global=jnp.zeros(obsv["global"].shape[1], dtype=float),
            action=jnp.zeros((env.num_agents, get_space_dim(action_space)), dtype=float),
            reward=jnp.zeros((env.num_agents,), dtype=float),
            done=jnp.zeros((env.num_agents,), dtype=bool),
            next_obs=jnp.zeros((env.num_agents, get_space_dim(env.observation_space(env.agents[0]))), dtype=float),
            next_obs_global=jnp.zeros(obsv["global"].shape[1], dtype=float),
        )
        rb = fbx.make_item_buffer(
            max_length=int(config["BUFFER_SIZE"]), # adding for sweeping functionality to use 1e6 for buffer size in hydra multirun
            min_length=config["EXPLORE_STEPS"],
            sample_batch_size=int(config["BATCH_SIZE"]),
            add_batches=True,
        )

        buffer_state = rb.init(init_transition)

        target_entropy = -config["TARGET_ENTROPY_SCALE"] * config["ACT_DIM"]
        target_entropy = jnp.repeat(target_entropy, env.num_agents) 
        target_entropy = target_entropy[:, jnp.newaxis]


        if config["AUTOTUNE"]:
            log_alpha = jnp.zeros_like(target_entropy)
        else: 
            log_alpha = jnp.log(config["INIT_ALPHA"])
            log_alpha = jnp.broadcast_to(log_alpha, target_entropy.shape)

    
        grad_clip = optax.clip_by_global_norm(config["MAX_GRAD_NORM"]) # Change if I want to sweep

        actor_opt = optax.chain(grad_clip, optax.adam(p_lr))

        q1_opt = optax.chain(grad_clip, optax.adam(q_lr))
        # Testing with 2 separate optimizers for each q function 
        q2_opt = optax.chain(grad_clip, optax.adam(q_lr))
        
        alpha_opt = optax.chain(grad_clip, optax.adam(alpha_lr))
        alpha_opt_state = alpha_opt.init(log_alpha)

        tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
        
        tau = tau

        actor_train_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=actor_opt,
        )
        q1_train_state = TrainState.create(
            apply_fn=q.apply,
            params=q1_params,
            tx=q1_opt,
        )

        q2_train_state = TrainState.create(
            apply_fn=q.apply,
            params=q2_params,
            tx=q2_opt,
        )
        
        train_states = SACTrainStates(
            actor=actor_train_state,
            q1=q1_train_state,
            q2=q2_train_state,
            q1_target=q1_params,
            q2_target=q2_params,
            log_alpha=log_alpha,
            alpha_opt_state=alpha_opt_state,
        )

        runner_state = RunnerState(
            train_states=train_states,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            t=0,
            buffer_state=buffer_state,
            rng=rng,
            total_env_steps=0,
            total_grad_updates=0,
            update_t=0,
        )

        
        def _explore(runner_state, unused):
            
            rng, explore_rng = jax.random.split(runner_state.rng)
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            
            avail_actions_shape = batchify(avail_actions, env.agents).shape
            action = jax.random.uniform(explore_rng, avail_actions_shape, minval=-1, maxval=1)
            env_act = unbatchify(action, env.agents)
            rng_step = jax.random.split(explore_rng, config["NUM_ENVS"])
           
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
            
            t = runner_state.t + config["NUM_ENVS"]

            last_obs_batch = batchify(runner_state.last_obs, env.agents)
            done_batch = batchify(done, env.agents)

            transition = Transition(
                    obs = last_obs_batch,
                    obs_global = runner_state.last_obs["global"],
                    action = action,
                    reward = batchify(reward, env.agents),
                    done = done_batch,
                    next_obs = batchify(obsv, env.agents),
                    next_obs_global = obsv["global"],
                )
            
            new_total_steps = runner_state.total_env_steps + config["NUM_ENVS"]
            
            runner_state = RunnerState(
                train_states=runner_state.train_states,
                env_state=env_state,
                last_obs = obsv,
                last_done=done_batch,
                t = t,
                buffer_state=runner_state.buffer_state,
                rng=rng,
                total_env_steps = new_total_steps,
                total_grad_updates = runner_state.total_grad_updates,
                update_t=runner_state.update_t,
            )

            return runner_state, transition 
        
        
        def _checkpoint_step(runner_state, unused):
            """ Used to reduce amount of parameters we save during training. """

            def _update_step(runner_state, unused):
                """ The SAC update"""
            

                def _env_step(runner_state, unused):
                    """ Step the environment """

                    rng = runner_state.rng
                    obs_batch = batchify(runner_state.last_obs, env.agents)
                    avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                    avail_actions = jax.lax.stop_gradient(
                        batchify(avail_actions, env.agents)
                    )
                    ac_in = (obs_batch, runner_state.last_done, avail_actions)

                    # SELECT ACTION
                    
                    rng, action_rng = jax.random.split(rng)
                    (actor_mean, actor_std) = runner_state.train_states.actor.apply_fn(
                        runner_state.train_states.actor.params, 
                        ac_in
                        )
                    
                    pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                    
                    pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)    
                    action = pi_tanh.sample(seed=action_rng)
                    env_act = unbatchify(action, env.agents)

                    #STEP ENV
                    rng, step_rng = jax.random.split(rng)
                    rng_step = jax.random.split(step_rng, config["NUM_ENVS"])
                    obsv, env_state, reward, done, _ = jax.vmap(env.step)(
                        rng_step, runner_state.env_state, env_act,
                    )
                    done_batch = batchify(done, env.agents)
                    # info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info)
                    # breakpoint()
                    transition = Transition(
                        obs = obs_batch,
                        obs_global = runner_state.last_obs["global"],
                        action = action,
                        reward = batchify(reward, env.agents),
                        done = done_batch,
                        next_obs = batchify(obsv, env.agents),
                        next_obs_global = obsv["global"],
                    )

                    t = runner_state.t + config["NUM_ENVS"]

                    new_total_steps = runner_state.total_env_steps + config["NUM_ENVS"]

                    runner_state = RunnerState(
                        train_states=runner_state.train_states,
                        env_state=env_state,
                        last_obs=obsv,
                        last_done=done_batch,
                        t=t,
                        buffer_state=runner_state.buffer_state,
                        rng=rng,
                        total_env_steps = new_total_steps,
                        total_grad_updates = runner_state.total_grad_updates,
                        update_t=runner_state.update_t,
                    )

                    return runner_state, transition
                
                runner_state, traj_batch  = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
                )
                
                traj_batch_reshaped = jax.tree_util.tree_map(
                    lambda x, f: reshape_for_buffer(x, f),
                    traj_batch, 
                    type(traj_batch)(*[name for name in traj_batch._fields])
                )
                
                new_buffer_state = rb.add(
                    runner_state.buffer_state,
                    traj_batch_reshaped, # move batch axis to start
                )

                runner_state = runner_state._replace(buffer_state=new_buffer_state)
                # breakpoint()
                def _update_networks(runner_state, rng): 
                    rng, batch_sample_rng, q_update_rng, actor_update_rng = jax.random.split(rng, 4)
                    train_state = runner_state.train_states
                    buffer_state = runner_state.buffer_state
                    # # # jax.debug.print('updating networks')
                    batch = rb.sample(buffer_state, batch_sample_rng).experience
                    
                    batch = jax.tree_util.tree_map(
                        lambda x, f: x.swapaxes(0, 1) if not ('global' in f) else x,
                        batch,
                        type(batch)(*[name for name in batch._fields])
                    )

                    #UPDATE Q_NETWORKS
                    def q_loss_fn(q1_online_params, q2_online_params, obs, action, target_q):

                        current_q1 = train_state.q1.apply_fn(
                            q1_online_params, 
                            obs, action
                        )
                        current_q2 = train_state.q2.apply_fn(
                            q2_online_params, 
                            obs, action
                        )
                    
                        # MSE loss for both Q-networks
                        q1_loss = jnp.mean(jnp.square(current_q1 - target_q))
                        q2_loss = jnp.mean(jnp.square(current_q2 - target_q))
                        return q1_loss + q2_loss, (q1_loss, q2_loss)
                    
                    # loss for the actor
                    def actor_loss_fn(actor_params, q1_params, q2_params, obs, obs_global, dones, alpha, rng, avail_actions):

                        next_ac_in = (obs, dones, avail_actions)

                        actor_mean, actor_std = train_state.actor.apply_fn(
                            actor_params, 
                            next_ac_in
                        )

                        pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                        
                        # act_distribution = tfd.Normal(loc=actor_mean, scale=actor_std)
                        # pi = tfd.Independent(
                        #     TanhTransformedDistribution(act_distribution),
                        #     reinterpreted_batch_ndims=1,
                        # )
                        # act_loss_action = pi.sample(seed=rng)
                        # log_prob = pi.log_prob(act_loss_action)
                            
                        pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)
                        act_loss_action, log_prob = pi_tanh.sample_and_log_prob(seed=rng) 

                        q1_values = train_state.q1.apply_fn(
                            q1_params, 
                            obs_global, flatten_actions(act_loss_action)
                        )
                        q2_values = train_state.q2.apply_fn(
                            q2_params,
                            obs_global, flatten_actions(act_loss_action)        
                        )
                        q_value = jnp.minimum(q1_values, q2_values)
                        
                        # actor loss with entropy
                        actor_loss = jnp.mean((alpha * log_prob) - q_value)

                        return actor_loss, log_prob
                    
                    def alpha_loss_fn(log_alpha, log_pi, target_entropy):

                        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))
                    
                    
                    # Q networks loss and gradient 
                    obs = batch.obs
                    obs_global = batch.obs_global
                    dones = batch.done
                    action = batch.action
                    next_obs = batch.next_obs
                    next_obs_global = batch.next_obs_global
                    reward = batch.reward

                    avail_actions =  jnp.zeros( # avail_actions
                            (env.num_agents, config["NUM_ENVS"]*config["BATCH_SIZE"], config["ACT_DIM"])
                        ) # this is unused for assistax but useful in other implementations
                    
                    avail_actions = jax.lax.stop_gradient(avail_actions)

                    def update_q(train_state):
                    
                        next_act_mean, next_act_std = train_state.actor.apply_fn(
                            train_state.actor.params, 
                            (next_obs, dones, avail_actions),
                        )

                        next_pi = distrax.MultivariateNormalDiag(next_act_mean, next_act_std)

                        # next_act_distribution = tfd.Normal(loc=next_act_mean, scale=next_act_std)
                        # next_pi = tfd.Independent(
                        #     TanhTransformedDistribution(next_act_distribution),
                        #     reinterpreted_batch_ndims=1,
                        # )
                        # next_action = next_pi.sample(seed=q_update_rng)
                        # next_log_prob = next_pi.log_prob(next_action)
                        next_pi_tanh = distrax.Transformed(next_pi, bijector=tanh_bijector)
                        next_action, next_log_prob = next_pi_tanh.sample_and_log_prob(seed=rng)
                    
                        # compute q target
                        next_q1 = train_state.q1.apply_fn(
                            train_state.q1_target, 
                            next_obs_global, flatten_actions(next_action) # change to global obs and falttened actions
                        )
                        next_q2 = train_state.q2.apply_fn(
                            train_state.q2_target, 
                            next_obs_global, flatten_actions(next_action) # change to global obs and falttened actions
                        )
            
                        next_q = jnp.minimum(next_q1, next_q2)
                        next_q = next_q - jnp.exp(train_state.log_alpha) * next_log_prob
                        target_q = reward + config["GAMMA"] * (1.0 - dones) * next_q

                        q_grad_fun = jax.value_and_grad(q_loss_fn, argnums=(0,1), has_aux=True)
                        (q_loss, (q1_loss, q2_loss)), (q1_grads, q2_grads) = q_grad_fun(
                            train_state.q1.params, 
                            train_state.q2.params, 
                            obs_global, # global_obs,
                            flatten_actions(action),
                            target_q,
                            )
                        
                        new_q1_train_state = train_state.q1.apply_gradients(grads=q1_grads)
                        new_q2_train_state = train_state.q2.apply_gradients(grads=q2_grads)
                        new_q1_target = optax.incremental_update(
                            new_q1_train_state.params,
                            train_state.q1_target,
                            tau,
                        )
                        new_q2_target = optax.incremental_update(
                            new_q2_train_state.params,
                            train_state.q2_target,
                            tau,
                        )

                        q_update_train_state = SACTrainStates( # TODO: use ._replace method instead
                            actor=train_state.actor,
                            q1=new_q1_train_state,
                            q2=new_q2_train_state,
                            q1_target=new_q1_target,
                            q2_target=new_q2_target,
                            log_alpha=train_state.log_alpha,
                            alpha_opt_state=train_state.alpha_opt_state,
                        )
                        
                        q_metrics = {
                            'critic_loss': q_loss,
                            'q1_loss': q1_loss,
                            'q2_loss': q2_loss,
                            'next_log_prob': next_log_prob
                            }

                        return q_update_train_state, q_metrics

                    # actor loss and gradient 
                    def _update_actor_and_alpha(carry, _):
                        train_state, dummy_metrics = carry
                        actor_grad_fun = jax.value_and_grad(actor_loss_fn, has_aux=True)
                        (actor_loss, log_prob), actor_grads = actor_grad_fun(
                            train_state.actor.params,
                            train_state.q1.params,
                            train_state.q2.params,
                            obs,
                            obs_global,
                            dones,
                            jnp.exp(train_state.log_alpha),
                            actor_update_rng,
                            avail_actions,
                        )

                        # alphaloss and gradient update
                        temperature_loss = 0.0
                        new_log_alpha = log_alpha
                        new_alpha_opt_state = alpha_opt_state
                        if config["AUTOTUNE"]:
                            alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
                            temperature_loss, alpha_grad = alpha_grad_fn(train_state.log_alpha, log_prob, target_entropy)
                            alpha_updates, new_alpha_opt_state = alpha_opt.update(alpha_grad, train_state.alpha_opt_state)
                            new_log_alpha = optax.apply_updates(train_state.log_alpha, alpha_updates)
                    
                        new_actor_train_state = train_state.actor.apply_gradients(grads=actor_grads)
                        
                        act_update_train_state = SACTrainStates(
                            actor=new_actor_train_state,
                            q1=train_state.q1,
                            q2=train_state.q2,
                            q1_target=train_state.q1_target,
                            q2_target=train_state.q2_target,
                            log_alpha=new_log_alpha,
                            alpha_opt_state=new_alpha_opt_state,
                        )

                        actor_metrics = {
                            "actor_loss": actor_loss, 
                            "alpha_loss": temperature_loss, 
                            "mean_log_prob": log_prob.mean(), 
                            }
                        # jax.debug.print("Updating Actor")
                        return (act_update_train_state, actor_metrics), _
                    
                    q_update_train_state, q_info = update_q(train_state)

                    (actor_update_train_state, actor_info), _ = jax.lax.cond(
                        runner_state.update_t % config["POLICY_UPDATE_DELAY"] == 0, # changed the t we use because of coprime thing 
                        lambda: jax.lax.scan(_update_actor_and_alpha, 
                                             (q_update_train_state, {'actor_loss': 0.0, 'alpha_loss': 0.0, 'mean_log_prob': 0.0}),
                                             None,
                                             length=config["POLICY_UPDATE_DELAY"]), 
                        lambda: ((q_update_train_state, 
                                {'actor_loss': 0.0, 'alpha_loss': 0.0, 'mean_log_prob': 0.0}), None)
                    )

                    new_train_state = SACTrainStates(
                        actor=actor_update_train_state.actor,
                        q1=q_update_train_state.q1,
                        q2=q_update_train_state.q2,
                        q1_target=q_update_train_state.q1_target,
                        q2_target=q_update_train_state.q2_target,
                        log_alpha=actor_update_train_state.log_alpha,
                        alpha_opt_state=actor_update_train_state.alpha_opt_state,
                    )

                    metrics = {
                        'critic_loss': q_info["critic_loss"],
                        'q1_loss': q_info["q1_loss"],
                        'q2_loss': q_info["q2_loss"],
                        'actor_loss': actor_info["actor_loss"],
                        'alpha_loss': actor_info["alpha_loss"],
                        'alpha': jnp.exp(actor_update_train_state.log_alpha),
                        'log_probs': actor_info["mean_log_prob"],
                        "next_log_probs": q_info["next_log_prob"].mean(),
                        "actor_update_step": actor_update_train_state.actor.step,
                        "q1_update_step": q_update_train_state.q1.step,
                        "q2_update_step": q_update_train_state.q2.step,
                        "step_counter": runner_state.t 
                    }
                    new_update_t = runner_state.update_t + 1 # count the updates 
                    runner_state = RunnerState(
                        train_states=new_train_state,
                        env_state=runner_state.env_state,
                        last_obs=runner_state.last_obs,
                        last_done=runner_state.last_done,
                        t=runner_state.t,
                        buffer_state=buffer_state,
                        rng=runner_state.rng,
                        total_env_steps=runner_state.total_env_steps,
                        total_grad_updates=new_train_state.actor.step,
                        update_t=new_update_t
                    )

                    return runner_state, metrics
                
                _, u_rng = jax.random.split(runner_state.rng)

                update_rngs = jax.random.split(u_rng, config["NUM_SAC_UPDATES"])

                runner_state, metrics = jax.lax.scan(_update_networks, runner_state, update_rngs)
                metrics = jax.tree.map(lambda x: x.mean(), metrics)
                
                return runner_state, metrics
            
            runner_state, metrics = jax.lax.scan(
                _update_step, runner_state, None, config["SCAN_STEPS"]
            )
            metrics = jax.tree.map(lambda x: x.mean(), metrics)

            if save_train_state:
                metrics.update({"actor_train_state": runner_state.train_states.actor})
                metrics.update({"q1_train_state": runner_state.train_states.q1})
                metrics.update({"q2_train_state": runner_state.train_states.q2})

            return runner_state, metrics
        
        # Exploration before training
        explore_runner_state, explore_traj_batch = jax.lax.scan(
                _explore, runner_state, None, config["EXPLORE_SCAN_STEPS"]
            )

        explore_traj_batch = jax.tree_util.tree_map(
            lambda x, f: reshape_for_buffer(x,f), 
            explore_traj_batch,
            type(explore_traj_batch)(*[name for name in explore_traj_batch._fields]), # this is a hack to get the field names 
        )

        explore_buffer_state = rb.add(
            runner_state.buffer_state,
            explore_traj_batch
        ) 
        
        explore_runner_state = RunnerState(
            train_states=explore_runner_state.train_states,
            env_state=explore_runner_state.env_state,
            last_obs=explore_runner_state.last_obs,
            last_done=explore_runner_state.last_done,
            t=explore_runner_state.t,
            buffer_state= explore_buffer_state,
            rng=explore_runner_state.rng,
            total_env_steps=explore_runner_state.total_env_steps,
            total_grad_updates=explore_runner_state.total_grad_updates,
            update_t=explore_runner_state.update_t

        )

        final_runner_state, checkpoint_metrics = jax.lax.scan(
            _checkpoint_step, explore_runner_state, None, config["NUM_CHECKPOINTS"]
        ) 

        return {"runner_state": final_runner_state, "metrics": checkpoint_metrics}
    
    return train

# def make_evaluation(config, load_zoo=False, crossplay=False):
#     if load_zoo:
#         zoo = ZooManager(config["ZOO_PATH"])
#         env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#         if crossplay:
#             env = LoadEvalAgentWrapper.load_from_zoo(env, zoo, load_zoo, crossplay)
#         else:
#             env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
#     else:
#         env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#     config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
#     config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
#     env = LogWrapper(env, replace_info=True)
#     max_steps = env.episode_length
#     tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
#     det_eval = config["DETERMINISTIC_EVAL"]

def make_evaluation(config, load_zoo=False, crossplay=False):
    if load_zoo:
        zoo = ZooManager(config["ZOO_PATH"])
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        if crossplay:
            env = LoadEvalAgentWrapper.load_from_zoo(env, zoo, load_zoo)
        else:
            env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["OBS_DIM"] = int(get_space_dim(env.observation_space(env.agents[0])))
    config["ACT_DIM"] = int(get_space_dim(env.action_space(env.agents[0])))
    if crossplay:
        env = LogCrossplayWrapper(env, replace_info=True, crossplay_info=crossplay) # this is stupid we do not need crossplay infor to be determined by crossplay
    else:
        env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length
    tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
    det_eval = config["DETERMINISTIC_EVAL"]

    def run_evaluation(rngs, train_state, log_eval_info=EvalInfoLogConfig()):
        
        if crossplay:
            rng_reset, rng_env = jax.random.split(rngs[0])
        else:
            rng_reset, rng_env = jax.random.split(rngs)
        
        # rng_reset, rng_env = jax.random.split(rngs[0])
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        # obsv, env_state = jax.vmap(env.reset)(rngs_reset)
        init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)
        if crossplay:
            init_obsv, init_env_state = jax.vmap(env.reset, in_axes=(0, None))(rngs_reset, None)
            init_runner_state = EvalState(
                train_states=train_state,
                env_state=init_env_state,
                last_obs=init_obsv,
                last_done=init_dones,
                update_step=0,
                rng=rng_env,
                ag_idx=init_env_state.env_state.ag_idx
            )

        else:
            init_obsv, init_env_state = jax.vmap(env.reset)(rngs_reset)
            init_runner_state = EvalState(
                train_states=train_state,
                env_state=init_env_state,
                last_obs=init_obsv,
                last_done=init_dones,
                update_step=0,
                rng=rng_env,
            )
        
        def _run_episode(runner_state, episode_rng):

            rng_reset, rng_env = jax.random.split(episode_rng)
            rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
            init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)
            if crossplay:
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(rngs_reset, runner_state.ag_idx)
                runner_state = EvalState(
                    train_states=runner_state.train_states,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=init_dones,
                    update_step=runner_state.update_step,
                    rng=rng_env,
                    ag_idx=env_state.env_state.ag_idx
                )
            else:
                obsv, env_state = jax.vmap(env.reset)(rngs_reset)
                runner_state = RunnerState(
                    train_states=runner_state.train_states,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=init_dones,
                    update_step=runner_state.update_step,
                    rng=rng_env,
                ) 


            def _env_step(runner_state, unused):

                rng = runner_state.rng
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                ac_in = (obs_batch, runner_state.last_done, avail_actions)

                rng, action_rng = jax.random.split(rng)
                (actor_mean, actor_std) = runner_state.train_states.apply_fn(
                    runner_state.train_states.params, 
                    ac_in
                    )
                # SELECT ACTION
                if det_eval:
                    action = jnp.tanh(actor_mean)
                
                else:
                    pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
    
                    pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)

                    action, log_prob = pi_tanh.sample_and_log_prob(seed=action_rng)

                env_act = unbatchify(action, env.agents)

                #STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
                done_batch = batchify(done, env.agents)
                info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info)
                            
                eval_info = EvalInfo(
                    env_state=(env_state if log_eval_info.env_state else None),
                    done=(done if log_eval_info.done else None),
                    action=(action if log_eval_info.action else None),
                    reward=(reward if log_eval_info.reward else None),
                    log_prob=(log_prob if log_eval_info.log_prob else None),
                    obs=(obs_batch if log_eval_info.obs else None),
                    info=(info if log_eval_info.info else None),
                    avail_actions=(avail_actions if log_eval_info.avail_actions else None),
                    ag_idx=(runner_state.ag_idx if crossplay else None),
                )
                runner_state = EvalState(
                    train_states=runner_state.train_states,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    update_step=runner_state.update_step,
                    rng=rng,
                    ag_idx=(runner_state.ag_idx if crossplay else None),
                )

                return runner_state, eval_info
            
            runner_state, episode_eval_info = jax.lax.scan(
                _env_step, runner_state, None, max_steps
            )

            return runner_state, episode_eval_info
        
        # runner_state, all_episode_eval_infos = jax.lax.scan(
        #     _run_episode, init_runner_state, rngs
        # )

        if crossplay:
            runner_state, all_episode_eval_infos = jax.lax.scan(
                _run_episode, init_runner_state, rngs
        )
        else:
            runner_state, all_episode_eval_infos = _run_episode(init_runner_state, rngs)

        return all_episode_eval_infos
    
    return env, run_evaluation

@hydra.main(version_base=None, config_path="config", config_name="masac_mabrax")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    # IMPORT FUNCTIONS BASED ON ARCHITECTURE
    # match (config["network"]["recurrent"], config["network"]["agent_param_sharing"]):
    #     case (False, False):
    #         from ippo_ff_nps_mabrax import make_train as make_train
    #         from ippo_ff_nps_mabrax import make_evaluation as make_evaluation
    #     case (False, True):
    #         from ippo_ff_ps_mabrax import make_train as make_train
    #         from ippo_ff_ps_mabrax import make_evaluation as make_evaluation
    #     case (True, False):
    #         from ippo_rnn_nps_mabrax import make_train as make_train
    #         from ippo_rnn_nps_mabrax import make_evaluation as make_evaluation
    #     case (True, True):
    #         from ippo_rnn_ps_mabrax import make_train as make_train
    #         from ippo_rnn_ps_mabrax import make_evaluation as make_evaluation

    rng = jax.random.PRNGKey(config["SEED"])
    train_rng, eval_rng = jax.random.split(rng)
    train_rngs = jax.random.split(train_rng, config["NUM_SEEDS"])    
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config, save_train_state=True),
            device=jax.devices()[config["DEVICE"]]
        )
        # first run (includes JIT)
        out = jax.vmap(train_jit, in_axes=(0, None, None, None, None))(
            train_rngs,
            config["POLICY_LR"], config["Q_LR"], config["ALPHA_LR"], config["TAU"] # TODO: Change these for SAC sweep
        )

        # SAVE TRAIN METRICS
        EXCLUDED_METRICS = ["actor_train_state", "q1_train_state", "q2_train_state"]
        jnp.save("metrics.npy", {
            key: val
            for key, val in out["metrics"].items()
            if key not in EXCLUDED_METRICS
            },
            allow_pickle=True
        )

        # SAVE PARAMS
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])

        all_train_states_actor = out["metrics"]["actor_train_state"]
        all_train_states_q1 = out["metrics"]["q1_train_state"]
        all_train_states_q2 = out["metrics"]["q2_train_state"]
        final_train_state_actor = out["runner_state"].train_states.actor
        final_train_state_q1 = out["runner_state"].train_states.q1
        final_train_state_q2 = out["runner_state"].train_states.q2

        safetensors.flax.save_file(
            flatten_dict(all_train_states_actor.params, sep='/'),
            "actor_all_params.safetensors"
        )

        safetensors.flax.save_file(
            flatten_dict(all_train_states_q1.params, sep='/'),
            "q1_all_params.safetensors"
        )

        safetensors.flax.save_file(
            flatten_dict(all_train_states_q2.params, sep='/'),
            "q2_all_params.safetensors"
        )

        if config["network"]["agent_param_sharing"]:
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
            # split by agent
            split_actor_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state_actor.params)
            )
            for agent, params in zip(env.agents, split_actor_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"actor_{agent}.safetensors",
                )

            split_q1_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state_q1.params)
            )
            for agent, params in zip(env.agents, split_q1_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q1_{agent}.safetensors",
                )

            split_q2_params = _unstack_tree(
                jax.tree.map(lambda x: x.swapaxes(0,1), final_train_state_q2.params)
            )
            for agent, params in zip(env.agents, split_q2_params):
                safetensors.flax.save_file(
                    flatten_dict(params, sep='/'),
                    f"q2_{agent}.safetensors",
                )
            
            

       
        # Assume the first 2 dimensions are batch dims
        batch_dims = jax.tree.leaves(_tree_shape(all_train_states_actor.params))[:2]
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
        split_trainstate = jax.jit(_flatten_and_split_trainstate)(all_train_states_actor)
        eval_env, run_eval = make_evaluation(config)
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
        eval_jit = jax.jit(
            run_eval,
            static_argnames=["log_eval_info"],
        )
        eval_vmap = jax.vmap(eval_jit, in_axes=(None, 0, None))
 
        evals = _concat_tree([
            eval_vmap(eval_rng, ts, eval_log_config) # Changed to true for rendering but get OOM 
            for ts in tqdm(split_trainstate, desc="Evaluation batches")
        ])
        evals = jax.tree.map(
            lambda x: x.reshape((*batch_dims, *x.shape[1:])),
            evals
        )

        # COMPUTE RETURNS
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

        # RENDER
        # Run episodes for render (saving env_state at each timestep)
        # eval_final = eval_jit(eval_rng, _tree_take(final_train_state_actor, 0, axis=0), True)
        # first_episode_done = jnp.cumsum(eval_final.done["__all__"], axis=0, dtype=bool)
        # first_episode_rewards = eval_final.reward["__all__"] * (1-first_episode_done)
        # first_episode_returns = first_episode_rewards.sum(axis=0)
        # episode_argsort = jnp.argsort(first_episode_returns, axis=-1)
        # worst_idx = episode_argsort.take(0,axis=-1)
        # best_idx = episode_argsort.take(-1, axis=-1)
        # median_idx = episode_argsort.take(episode_argsort.shape[-1]//2, axis=-1)

        # from brax.io import html
        # worst_episode = _take_episode(
        #     eval_final.env_state.env_state.pipeline_state, first_episode_done,
        #     time_idx=-1, eval_idx=worst_idx,
        # )
        # median_episode = _take_episode(
        #     eval_final.env_state.env_state.pipeline_state, first_episode_done,
        #     time_idx=-1, eval_idx=median_idx,
        # )
        # best_episode = _take_episode(
        #     eval_final.env_state.env_state.pipeline_state, first_episode_done,
        #     time_idx=-1, eval_idx=best_idx,
        # )
        # breakpoint() # maybe test eval_env.sys.geom_rgba 
        # html.save("final_worst.html", eval_env.sys, worst_episode)
        # html.save("final_median.html", eval_env.sys, median_episode)
        # html.save("final_best.html", eval_env.sys, best_episode)


if __name__ == "__main__":
    main()
