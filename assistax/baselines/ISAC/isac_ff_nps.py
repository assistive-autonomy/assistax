"""
Independent Soft Actor-Critic (ISAC) - Complete Algorithm Implementation

This module implements ISAC, an off-policy multi-agent reinforcement learning algorithm
based on Soft Actor-Critic (SAC). ISAC extends SAC to multi-agent environments where
each agent learns independently with its own networks and experience.

"""

import os
import functools
from functools import partial
from typing import Sequence, NamedTuple, TypeAlias, Any, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import tensorflow_probability.substrates.jax.distributions as tfd
from flax.core.scope import FrozenVariableDict
import flashbax as fbx
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
import assistax
from assistax.wrappers.baselines import  get_space_dim, LogEnvState, LogWrapper


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _tree_take(pytree, indices, axis=None):
    """Take elements from pytree along specified axis."""
    return jax.tree_util.tree_map(lambda x: x.take(indices, axis=axis), pytree)


def _tree_shape(pytree):
    """Get shapes of all leaves in pytree for debugging."""
    return jax.tree_util.tree_map(lambda x: x.shape, pytree)


def _unstack_tree(pytree):
    """
    Unstack a pytree along its first axis.
    
    Converts a tree with arrays of shape (n, ...) to a list of n trees
    with arrays of shape (...).
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = list(zip(*leaves))
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]


def _take_episode(pipeline_states, dones, time_idx=-1, eval_idx=0):
    """Extract episode data from pipeline states for visualization."""
    episodes = _tree_take(pipeline_states, eval_idx, axis=1)
    dones = dones.take(eval_idx, axis=1)
    return [
        state
        for state, done in zip(_unstack_tree(episodes), dones)
        if not done
    ]


def _concat_tree(pytree_list, axis=0):
    """Concatenate a list of pytrees along specified axis."""
    return jax.tree_util.tree_map(
        lambda *leaf: jnp.concat(leaf, axis=axis),
        *pytree_list
    )


def _tree_split(pytree, n, axis=0):
    """Split pytree into n parts along specified axis."""
    leaves, treedef = jax.tree.flatten(pytree)
    split_leaves = zip(
        *jax.tree.map(lambda x: jnp.array_split(x, n, axis), leaves)
    )
    return [
        jax.tree.unflatten(treedef, leaves)
        for leaves in split_leaves
    ]


def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """Convert dictionary of agent-specific arrays to batched array."""
    return jnp.stack(tuple(qty[a] for a in agents))


def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array back to dictionary of agent-specific arrays."""
    return dict(zip(agents, qty))


def _compute_episode_returns(eval_info, time_axis=-2):
    """
    Compute undiscounted episode returns from evaluation info.
    
    Args:
        eval_info: Evaluation information containing rewards and done flags
        time_axis: Axis along which time is indexed
        
    Returns:
        Dictionary of undiscounted returns per agent
    """
    done_arr = eval_info.done["__all__"]
    
    # Create episode mask to separate different episodes
    first_timestep = [slice(None) for _ in range(done_arr.ndim)]
    first_timestep[time_axis] = 0
    episode_done = jnp.cumsum(done_arr, axis=time_axis, dtype=bool)
    episode_done = jnp.roll(episode_done, 1, axis=time_axis)
    episode_done = episode_done.at[tuple(first_timestep)].set(False)
    
    # Compute returns by masking out rewards from previous episodes
    undiscounted_returns = jax.tree.map(
        lambda r: (r * (1 - episode_done)).sum(axis=time_axis),
        eval_info.reward
    )
    
    return undiscounted_returns


def reshape_for_buffer(x):
    """
    Reshape trajectories for experience replay buffer.
    
    Converts from (time, agents, envs, ...) to (time*envs, agents, ...)
    """
    x = x.swapaxes(1, 2)  # (time, envs, agents, ...)
    timesteps = x.shape[0]
    num_envs = x.shape[1]
    return x.reshape(timesteps * num_envs, *x.shape[2:])


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiSACActor(nn.Module):
    """
    Multi-agent SAC actor network with independent parameters per agent.
    
    This network learns stochastic policies that output Gaussian action distributions.
    The use of nn.vmap enables efficient parallel computation across agents while
    maintaining separate parameters for each agent.
    
    Key Features:
    - Independent parameters per agent (no parameter sharing)
    - Outputs mean and standard deviation for Gaussian policies
    - Learns exploration through entropy regularization
    - Tanh activation is applied during action selection (not here)
    """
    config: Dict
    
    @nn.compact
    def __call__(self, x):
        """
        Forward pass through actor network.
        
        Args:
            x: Tuple of (observations, done_flags, available_actions)
            
        Returns:
            Tuple of (action_mean, action_std)
        """
        # Select activation function
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
            
        obs, done, avail_actions = x
        
        # Actor network - two hidden layers
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
        
        # Output mean of action distribution
        actor_mean = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),  # Small initialization for stable policy
            bias_init=constant(0.0)
        )(actor_hidden)
        
        # Learnable log standard deviation (shared across actions)
        log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.config["ACT_DIM"],)
        )
        actor_log_std = jnp.broadcast_to(log_std, actor_mean.shape)

        return actor_mean, jnp.exp(actor_log_std)


@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiSACQNetwork(nn.Module):
    """
    Multi-agent SAC Q-network with independent parameters per agent.
    
    This network estimates Q-values given state-action pairs. ISAC uses two
    identical Q-networks (Q1, Q2) to reduce overestimation bias common in
    Q-learning algorithms.
    
    Key Features:
    - Independent parameters per agent (no parameter sharing)
    - Takes concatenated (observation, action) as input
    - Outputs scalar Q-values
    - Used in double Q-learning setup for stability
    """
    config: Dict
    
    @nn.compact
    def __call__(self, x, action):
        """
        Forward pass through Q-network.
        
        Args:
            x: Tuple of (observations, done_flags, available_actions)
            action: Action array to evaluate
            
        Returns:
            Q-value estimates for the given state-action pairs
        """
        # Select activation function
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x
        
        # Concatenate observations and actions
        x = jnp.concatenate([obs, action], axis=-1)
        
        # Q-network - two hidden layers
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
        
        # Output Q-value
        x = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(x)
        
        return jnp.squeeze(x, axis=-1)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Transition(NamedTuple):
    """Single transition for experience replay buffer."""
    obs: jnp.ndarray           # Current observations
    action: jnp.ndarray        # Actions taken
    reward: jnp.ndarray        # Rewards received
    done: jnp.ndarray          # Episode termination flags
    next_obs: jnp.ndarray      # Next observations


class SACTrainStates(NamedTuple):
    """Training states for all ISAC networks."""
    actor: TrainState          # Actor network training state
    q1: TrainState             # Q1 network training state
    q2: TrainState             # Q2 network training state
    q1_target: Dict            # Q1 target network parameters
    q2_target: Dict            # Q2 target network parameters
    log_alpha: jnp.ndarray     # Log entropy temperature
    alpha_opt_state: optax.OptState  # Entropy temperature optimizer state


# Type alias for buffer state
BufferState: TypeAlias = TrajectoryBufferState[Transition]


class RunnerState(NamedTuple):
    """State maintained by the training runner."""
    train_states: SACTrainStates
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    t: int                     # Environment timesteps
    buffer_state: BufferState  # Experience replay buffer state
    rng: jnp.ndarray          # Random number generator
    total_env_steps: int       # Total environment steps taken
    total_grad_updates: int    # Total gradient updates performed
    update_t: int              # Update counter for policy delay


class EvalState(NamedTuple):
    """State maintained during evaluation."""
    train_states: SACTrainStates
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    update_step: int
    rng: jnp.ndarray


class EvalInfo(NamedTuple):
    """Information logged during evaluation."""
    env_state: LogEnvState
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray


@struct.dataclass
class EvalInfoLogConfig:
    """Configuration for what information to log during evaluation."""
    env_state: bool = True
    done: bool = True
    action: bool = True
    reward: bool = True
    log_prob: bool = True
    obs: bool = True
    info: bool = True
    avail_actions: bool = True


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def make_train(config, save_train_state=True):
    """
    Create the main ISAC training function.
    
    This function sets up the complete ISAC training pipeline including:
    - Network initialization
    - Experience replay buffer
    - Exploration phase
    - Main training loop with policy updates
    
    Args:
        config: Training configuration dictionary
        save_train_state: Whether to save training states in metrics
        
    Returns:
        Compiled training function
    """
    # Initialize environment and configuration
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Calculate training parameters
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LENGTH"] // config["NUM_ENVS"]
    )
    config["SCAN_STEPS"] = config["NUM_UPDATES"] // config["NUM_CHECKPOINTS"]
    config["EXPLORE_SCAN_STEPS"] = config["EXPLORE_STEPS"] // config["NUM_ENVS"]
    
    print(f"ISAC Training Configuration:")
    print(f"  NUM_UPDATES: {config['NUM_UPDATES']}")
    print(f"  SCAN_STEPS: {config['SCAN_STEPS']}")
    print(f"  EXPLORE_STEPS: {config['EXPLORE_STEPS']}")
    print(f"  NUM_CHECKPOINTS: {config['NUM_CHECKPOINTS']}")
    
    # Set observation and action dimensions
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    
    # Wrap environment for logging
    env = LogWrapper(env, replace_info=True)
    
    def train(rng, p_lr, q_lr, alpha_lr, tau):
        """
        Main ISAC training function.
        
        Args:
            rng: Random number generator key
            p_lr: Policy (actor) learning rate
            q_lr: Q-function learning rate
            alpha_lr: Entropy temperature learning rate
            tau: Soft update coefficient for target networks
            
        Returns:
            Dictionary containing final runner state and training metrics
        """
        
        # ====================================================================
        # INITIALIZE NETWORKS
        # ====================================================================
        
        actor = MultiSACActor(config=config)
        q = MultiSACQNetwork(config=config)

        rng, actor_rng, q1_rng, q2_rng = jax.random.split(rng, num=4)

        # Initialize dummy inputs for parameter initialization
        init_x = (
            jnp.zeros((env.num_agents, 1, config["OBS_DIM"])),     # obs
            jnp.zeros((env.num_agents, 1)),                        # done
            jnp.zeros((env.num_agents, 1, config["ACT_DIM"])),     # avail_actions
        )
        
        # Initialize network parameters
        actor_params = actor.init(actor_rng, init_x)
        dummy_action = jnp.zeros((env.num_agents, 1, config["ACT_DIM"]))
        q1_params = q.init(q1_rng, init_x, dummy_action)
        q2_params = q.init(q2_rng, init_x, dummy_action)

        # ====================================================================
        # INITIALIZE ENVIRONMENT
        # ====================================================================
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_dones = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool)

        # ====================================================================
        # INITIALIZE EXPERIENCE REPLAY BUFFER
        # ====================================================================
        
        # Create template transition for buffer initialization
        action_space = env.action_space(env.agents[0])
        init_transition = Transition(
            obs=jnp.zeros((env.num_agents, get_space_dim(env.observation_space(env.agents[0]))), dtype=float),
            action=jnp.zeros((env.num_agents, get_space_dim(action_space)), dtype=float),
            reward=jnp.zeros((env.num_agents,), dtype=float),
            done=jnp.zeros((env.num_agents,), dtype=bool),
            next_obs=jnp.zeros((env.num_agents, get_space_dim(env.observation_space(env.agents[0]))), dtype=float)
        )

        # Initialize experience replay buffer
        rb = fbx.make_item_buffer(
            max_length=config["BUFFER_SIZE"],
            min_length=config["EXPLORE_STEPS"],
            sample_batch_size=int(config["BATCH_SIZE"]),
            add_batches=True,
        )
        buffer_state = rb.init(init_transition)

        # ====================================================================
        # INITIALIZE ENTROPY REGULARIZATION
        # ====================================================================
        
        # Target entropy for automatic entropy tuning
        target_entropy = -config["TARGET_ENTROPY_SCALE"] * config["ACT_DIM"]
        target_entropy = jnp.repeat(target_entropy, env.num_agents)
        target_entropy = target_entropy[:, jnp.newaxis]

        # Initialize entropy temperature (alpha)
        if config["AUTOTUNE"]:
            log_alpha = jnp.zeros_like(target_entropy)
        else:
            log_alpha = jnp.log(config["INIT_ALPHA"])
            log_alpha = jnp.broadcast_to(log_alpha, target_entropy.shape)

        # ====================================================================
        # INITIALIZE OPTIMIZERS
        # ====================================================================
        
        grad_clip = optax.clip_by_global_norm(config["MAX_GRAD_NORM"])

        actor_opt = optax.chain(grad_clip, optax.adam(p_lr))
        q1_opt = optax.chain(grad_clip, optax.adam(q_lr))
        q2_opt = optax.chain(grad_clip, optax.adam(q_lr))
        alpha_opt = optax.chain(grad_clip, optax.adam(alpha_lr))
        alpha_opt_state = alpha_opt.init(log_alpha)

        # Tanh bijector for action space transformation
        tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)

        # ====================================================================
        # INITIALIZE TRAINING STATES
        # ====================================================================
        
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

        # ====================================================================
        # EXPLORATION PHASE
        # ====================================================================
        
        def _explore(runner_state, unused):
            """
            Exploration phase: collect random experiences to initialize buffer.
            
            This phase is crucial for off-policy learning as it provides initial
            data before policy learning begins.
            """
            rng, explore_rng = jax.random.split(runner_state.rng)
            
            # Get available actions (unused for continuous control)
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            avail_actions_shape = batchify(avail_actions, env.agents).shape
            
            # Sample random actions uniformly in [-1, 1]
            action = jax.random.uniform(explore_rng, avail_actions_shape, minval=-1, maxval=1)
            env_act = unbatchify(action, env.agents)

            # Step environment
            rng_step = jax.random.split(explore_rng, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, runner_state.env_state, env_act,
            )
            
            # Prepare transition for buffer
            last_obs_batch = batchify(runner_state.last_obs, env.agents)
            done_batch = batchify(done, env.agents)

            transition = Transition(
                obs=last_obs_batch,
                action=action,
                reward=batchify(reward, env.agents),
                done=done_batch,
                next_obs=batchify(obsv, env.agents),
            )

            # Update runner state
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
                total_env_steps=new_total_steps,
                total_grad_updates=runner_state.total_grad_updates,
                update_t=runner_state.update_t,
            )

            return runner_state, transition

        # ====================================================================
        # MAIN TRAINING LOOP
        # ====================================================================
        
        def _checkpoint_step(runner_state, unused):
            """
            Checkpoint step: reduce parameters saved during training.
            
            This function groups multiple update steps to reduce memory usage
            when saving training states.
            """

            def _update_step(runner_state, unused):
                """Single update step: collect experience and update networks."""
            
                def _env_step(runner_state, unused):
                    """Single environment step during training."""
                    rng = runner_state.rng
                    
                    # Prepare observations and available actions
                    obs_batch = batchify(runner_state.last_obs, env.agents)
                    avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                    avail_actions = jax.lax.stop_gradient(
                        batchify(avail_actions, env.agents)
                    )
                    ac_in = (obs_batch, runner_state.last_done, avail_actions)

                    # ========================================================
                    # SELECT ACTIONS USING CURRENT POLICY
                    # ========================================================
                    
                    rng, action_rng = jax.random.split(rng)
                    (actor_mean, actor_std) = runner_state.train_states.actor.apply_fn(
                        runner_state.train_states.actor.params, 
                        ac_in
                    )
                    
                    # Sample actions from policy with tanh transformation
                    pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                    pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)
                    action, log_prob = pi_tanh.sample_and_log_prob(seed=action_rng)
                    env_act = unbatchify(action, env.agents)

                    # ========================================================
                    # STEP ENVIRONMENT
                    # ========================================================
                    
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                    obsv, env_state, reward, done, info = jax.vmap(env.step)(
                        rng_step, runner_state.env_state, env_act,
                    )
                    done_batch = batchify(done, env.agents)
                    info = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), info)

                    # Store transition
                    transition = Transition(
                        obs=obs_batch,
                        action=action,
                        reward=batchify(reward, env.agents),
                        done=done_batch,
                        next_obs=batchify(obsv, env.agents),
                    )

                    # Update runner state
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
                        total_env_steps=new_total_steps,
                        total_grad_updates=runner_state.total_grad_updates,
                        update_t=runner_state.update_t,
                    )

                    return runner_state, transition
                
                # Collect rollout and add to buffer
                runner_state, traj_batch = jax.lax.scan(
                    _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
                )
                
                # Reshape trajectories for buffer storage
                traj_batch_reshaped = jax.tree_util.tree_map(
                    lambda x: reshape_for_buffer(x),
                    traj_batch,
                )
                
                # Add experiences to replay buffer
                new_buffer_state = rb.add(
                    runner_state.buffer_state,
                    traj_batch_reshaped,
                )
                runner_state = runner_state._replace(buffer_state=new_buffer_state)

                # ============================================================
                # NETWORK UPDATES
                # ============================================================

                def _update_networks(runner_state, rng):
                    """Update all ISAC networks using sampled batch."""

                    rng, batch_sample_rng, q_sample_rng, actor_update_rng = jax.random.split(rng, 4)
                    train_state = runner_state.train_states
                    buffer_state = runner_state.buffer_state
                    
                    # Sample batch from replay buffer
                    batch = rb.sample(buffer_state, batch_sample_rng).experience
                    batch = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), batch)

                    # ================================================
                    # Q-NETWORK LOSS FUNCTIONS
                    # ================================================
                    
                    def q_loss_fn(q1_online_params, q2_online_params, obs, dones, action, target_q, avail_actions):
                        """Compute Q-network losses using target Q-values."""
                        
                        # Current Q-values from both networks
                        current_q1 = train_state.q1.apply_fn(
                            q1_online_params, 
                            (obs, dones, avail_actions), action
                        )
                        current_q2 = train_state.q2.apply_fn(
                            q2_online_params, 
                            (obs, dones, avail_actions), action
                        )
                    
                        # MSE loss for both Q-networks
                        q1_loss = jnp.mean(jnp.square(current_q1 - target_q))
                        q2_loss = jnp.mean(jnp.square(current_q2 - target_q))

                        return q1_loss + q2_loss, (q1_loss, q2_loss)
                    
                    # ================================================
                    # ACTOR LOSS FUNCTION
                    # ================================================
                    
                    def actor_loss_fn(actor_params, q1_params, q2_params, obs, dones, alpha, rng, avail_actions):
                        """Compute actor loss with entropy regularization."""

                        next_ac_in = (obs, dones, avail_actions)

                        # Get action distribution from current policy
                        actor_mean, actor_std = train_state.actor.apply_fn(
                            actor_params, 
                            next_ac_in
                        )

                        # Sample actions and compute log probabilities
                        pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                        pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)
                        action, log_prob = pi_tanh.sample_and_log_prob(seed=rng)

                        # Compute Q-values for sampled actions
                        q1_values = train_state.q1.apply_fn(
                            q1_params, 
                            (obs, dones, avail_actions), action
                        )
                        q2_values = train_state.q2.apply_fn(
                            q2_params, 
                            (obs, dones, avail_actions), action
                        )
                        q_value = jnp.minimum(q1_values, q2_values)
                        
                        # Actor loss: maximize Q-value minus entropy penalty
                        actor_loss = jnp.mean((alpha * log_prob) - q_value)

                        return actor_loss, log_prob
                    
                    # ================================================
                    # ENTROPY TEMPERATURE LOSS
                    # ================================================
                    
                    def alpha_loss_fn(log_alpha, log_pi, target_entropy):
                        """Compute entropy temperature loss for automatic tuning."""
                        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

                    # Extract batch components
                    obs = batch.obs
                    dones = batch.done
                    action = batch.action
                    next_obs = batch.next_obs
                    reward = batch.reward

                    # Available actions (unused for continuous control but kept for compatibility)
                    avail_actions = jnp.zeros(
                        (env.num_agents, config["NUM_ENVS"]*config["BATCH_SIZE"], config["ACT_DIM"])
                    )
                    avail_actions = jax.lax.stop_gradient(avail_actions)

                    # ================================================
                    # UPDATE Q-NETWORKS
                    # ================================================

                    def update_q(train_state):
                        """Update Q-networks using target Q-values."""

                        # Compute next actions using current policy
                        next_act_mean, next_act_std = train_state.actor.apply_fn(
                            train_state.actor.params, 
                            (next_obs, dones, avail_actions),
                        )

                        next_pi = distrax.MultivariateNormalDiag(next_act_mean, next_act_std)
                        next_pi_tanh = distrax.Transformed(next_pi, bijector=tanh_bijector)
                        next_action, next_log_prob = next_pi_tanh.sample_and_log_prob(seed=rng)
                        
                        # Compute target Q-values using target networks
                        next_q1 = train_state.q1.apply_fn(
                            train_state.q1_target, 
                            (next_obs, dones, avail_actions), next_action
                        )
                        next_q2 = train_state.q2.apply_fn(
                            train_state.q2_target, 
                            (next_obs, dones, avail_actions), next_action
                        )
                
                        # Take minimum to reduce overestimation
                        next_q = jnp.minimum(next_q1, next_q2)

                        # Subtract entropy term for target Q-value
                        next_q = next_q - jnp.exp(train_state.log_alpha) * next_log_prob

                        # Compute target Q-value with Bellman equation
                        target_q = reward + config["GAMMA"] * (1.0 - dones) * next_q
                        
                        # Compute gradients and update Q-networks
                        q_grad_fun = jax.value_and_grad(q_loss_fn, has_aux=True)
                        (q_loss, (q1_loss, q2_loss)), q_grads = q_grad_fun(
                            train_state.q1.params, 
                            train_state.q2.params, 
                            obs, 
                            dones, 
                            action, 
                            target_q,
                            avail_actions,
                        )
                        
                        # Apply gradients
                        new_q1_train_state = train_state.q1.apply_gradients(grads=q_grads)
                        new_q2_train_state = train_state.q2.apply_gradients(grads=q_grads)

                        # Soft update target networks
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

                        # Update training state
                        q_update_train_state = SACTrainStates(
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
                    
                    # ================================================
                    # UPDATE ACTOR AND ENTROPY TEMPERATURE
                    # ================================================
                    
                    def _update_actor_and_alpha(carry, _):
                        """Update actor and entropy temperature."""
                        
                        # Compute actor gradients
                        actor_grad_fun = jax.value_and_grad(actor_loss_fn, has_aux=True)
                        (actor_loss, log_prob), actor_grads = actor_grad_fun(
                            train_state.actor.params,
                            train_state.q1.params,
                            train_state.q2.params,
                            obs,
                            dones,
                            jnp.exp(train_state.log_alpha),
                            actor_update_rng,
                            avail_actions,
                        )
                        
                        # Update entropy temperature if using automatic tuning
                        temperature_loss = 0.0
                        new_log_alpha = train_state.log_alpha
                        new_alpha_opt_state = train_state.alpha_opt_state
                        
                        if config["AUTOTUNE"]:
                            alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
                            temperature_loss, alpha_grad = alpha_grad_fn(
                                train_state.log_alpha, log_prob, target_entropy
                            )
                            alpha_updates, new_alpha_opt_state = alpha_opt.update(
                                alpha_grad, train_state.alpha_opt_state
                            )
                            new_log_alpha = optax.apply_updates(train_state.log_alpha, alpha_updates)
                    
                        # Apply actor gradients
                        new_actor_train_state = train_state.actor.apply_gradients(grads=actor_grads)
                        
                        # Update training state
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
        
                        return (act_update_train_state, actor_metrics), _
                        
                    # First update Q-networks
                    q_update_train_state, q_info = update_q(train_state)
                    
                    # Update actor with policy delay (common SAC practice)
                    (actor_update_train_state, actor_info), _ = jax.lax.cond(
                        runner_state.update_t % config["POLICY_UPDATE_DELAY"] == 0,
                        lambda: jax.lax.scan(
                            _update_actor_and_alpha, 
                            (q_update_train_state, {'actor_loss': 0.0, 'alpha_loss': 0.0, 'mean_log_prob': 0.0}),
                            None,
                            length=config["POLICY_UPDATE_DELAY"]
                        ), 
                        lambda: ((q_update_train_state, 
                                {'actor_loss': 0.0, 'alpha_loss': 0.0, 'mean_log_prob': 0.0}), None)
                    )
                    
                    # Combine training states
                    new_train_state = SACTrainStates(
                        actor=actor_update_train_state.actor,
                        q1=q_update_train_state.q1,
                        q2=q_update_train_state.q2,
                        q1_target=q_update_train_state.q1_target,
                        q2_target=q_update_train_state.q2_target,
                        log_alpha=actor_update_train_state.log_alpha,
                        alpha_opt_state=actor_update_train_state.alpha_opt_state,
                    )

                    # Collect metrics
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
                    
                    # Update runner state
                    new_update_t = runner_state.update_t + 1
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
                
                # Perform multiple SAC updates per environment step
                _, sample_rng = jax.random.split(runner_state.rng)
                update_rngs = jax.random.split(sample_rng, config["NUM_SAC_UPDATES"])
                runner_state, metrics = jax.lax.scan(_update_networks, runner_state, update_rngs)
                metrics = jax.tree.map(lambda x: x.mean(), metrics)
            
                return runner_state, metrics
            
            # Run multiple update steps per checkpoint
            runner_state, metrics = jax.lax.scan(
                _update_step, runner_state, None, config["SCAN_STEPS"]
            )
            metrics = jax.tree.map(lambda x: x.mean(), metrics)

            # Optionally save training states
            if save_train_state:
                metrics.update({"actor_train_state": runner_state.train_states.actor})
                metrics.update({"q1_train_state": runner_state.train_states.q1})
                metrics.update({"q2_train_state": runner_state.train_states.q2})

            return runner_state, metrics
        
        # ====================================================================
        # RUN EXPLORATION AND TRAINING
        # ====================================================================
        
        # Exploration phase: populate buffer with random experiences
        explore_runner_state, explore_traj_batch = jax.lax.scan(
            _explore, runner_state, None, config["EXPLORE_SCAN_STEPS"]
        )

        # Reshape and add exploration data to buffer
        explore_traj_batch = jax.tree_util.tree_map(
            lambda x: reshape_for_buffer(x), explore_traj_batch
        )
  
        explore_buffer_state = rb.add(
            runner_state.buffer_state,
            explore_traj_batch
        )
                
        # Update runner state with exploration data
        explore_runner_state = RunnerState(
            train_states=explore_runner_state.train_states,
            env_state=explore_runner_state.env_state,
            last_obs=explore_runner_state.last_obs,
            last_done=explore_runner_state.last_done,
            t=explore_runner_state.t,
            buffer_state=explore_buffer_state,
            rng=explore_runner_state.rng,
            total_env_steps=explore_runner_state.total_env_steps,
            total_grad_updates=explore_runner_state.total_grad_updates,
            update_t=explore_runner_state.update_t
        )

        # Main training phase
        final_runner_state, checkpoint_metrics = jax.lax.scan(
            _checkpoint_step, explore_runner_state, None, config["NUM_CHECKPOINTS"]
        )
        
        return {"runner_state": final_runner_state, "metrics": checkpoint_metrics}
    
    return train


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def make_evaluation(config):
    """
    Create evaluation function for trained ISAC agents.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (environment, evaluation_function)
    """
    # Initialize environment
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length
    
    # Tanh bijector for action space transformation
    tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)
    det_eval = config["DETERMINISTIC_EVAL"]

    def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig()):
        """
        Run evaluation episodes with trained ISAC agents.
        
        Args:
            rng: Random number generator key
            train_state: Trained actor network state
            log_eval_info: Configuration for what to log
            
        Returns:
            Evaluation information across all episodes
        """
        # Initialize evaluation episodes
        rng_reset, rng_env = jax.random.split(rng)
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        obsv, env_state = jax.vmap(env.reset)(rngs_reset)
        init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)

        runner_state = EvalState(
            train_states=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            update_step=0,
            rng=rng_env,
        )
        
        def _env_step(runner_state, unused):
            """Single environment step during evaluation."""
            rng = runner_state.rng
            
            # Prepare observations
            obs_batch = batchify(runner_state.last_obs, env.agents)
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents)
            )
            ac_in = (obs_batch, runner_state.last_done, avail_actions)

            # Select actions
            rng, action_rng = jax.random.split(rng)
            (actor_mean, actor_std) = runner_state.train_states.apply_fn(
                runner_state.train_states.params, 
                ac_in
            )
            
            # Deterministic vs stochastic evaluation
            if det_eval:
                # Use mean action for deterministic evaluation
                action = jnp.tanh(actor_mean)
                log_prob = jnp.zeros_like(action[..., 0])  # Dummy log_prob
            else:
                # Sample from policy for stochastic evaluation
                pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                pi_tanh = distrax.Transformed(pi, bijector=tanh_bijector)
                action, log_prob = pi_tanh.sample_and_log_prob(seed=action_rng)

            env_act = unbatchify(action, env.agents)

            # Step environment
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, runner_state.env_state, env_act,
            )
            done_batch = batchify(done, env.agents)
            info = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), info)
                        
            # Log evaluation info
            eval_info = EvalInfo(
                env_state=(env_state if log_eval_info.env_state else None),
                done=(done if log_eval_info.done else None),
                action=(action if log_eval_info.action else None),
                reward=(reward if log_eval_info.reward else None),
                log_prob=(log_prob if log_eval_info.log_prob else None),
                obs=(obs_batch if log_eval_info.obs else None),
                info=(info if log_eval_info.info else None),
                avail_actions=(avail_actions if log_eval_info.avail_actions else None),
            )
            
            # Update evaluation state
            runner_state = EvalState(
                train_states=runner_state.train_states,
                env_state=env_state,
                last_obs=obsv,
                last_done=done_batch,
                update_step=runner_state.update_step,
                rng=rng,
            )
            return runner_state, eval_info

        # Run evaluation episodes
        _, eval_info = jax.lax.scan(_env_step, runner_state, None, max_steps)
        return eval_info
        
    return env, run_evaluation

       



