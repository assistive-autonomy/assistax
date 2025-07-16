"""
Multi-Agent Proximal Policy Optimization (MAPPO) Implementation with Feedforward Networks and Parameter Sharing

This module implements MAPPO for multi-agent reinforcement learning using JAX/Flax where all agents
share the same network parameters. This combines the CTDE paradigm with parameter sharing benefits,
resulting in memory-efficient training while maintaining the advantages of centralized critic training.

Based on the JaxMARL Implementation of MAPPO
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
from jaxmarl.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Optional
import functools


# ================================ NETWORK ARCHITECTURE ================================

class Actor(nn.Module):
    """
    Shared actor network for all agents.
    
    In MAPPO with parameter sharing, all agents use the same actor network parameters
    while processing their local observations independently. This approach reduces
    memory usage significantly while still allowing decentralized execution.
    
    Args:
        config: Configuration dictionary containing network hyperparameters
    """
    config: Dict

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through shared actor network.
        
        Args:
            x: Tuple of (local_observations, done_flags, available_actions)
               where observations are concatenated across all agents
               
        Returns:
            Policy distribution for continuous actions
        """
        # Select activation function
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        # ===== SHARED ACTOR NETWORK =====
        # All agents process their observations through the same network weights
        actor_mean = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        actor_mean = activation(actor_mean)
        
        actor_mean = nn.Dense(
            self.config["network"]["actor_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        
        # Final layer with small initialization for stable policy updates
        actor_mean = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)
        
        # Shared learnable log standard deviation parameter
        actor_log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.config["ACT_DIM"],)
        )
        
        # Create policy distribution directly
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))
        return pi


class Critic(nn.Module):
    """
    Centralized critic network using global state information.
    
    The critic maintains the same centralized structure as in non-parameter sharing
    MAPPO, using global observations to provide better value estimates and credit
    assignment for all agents.
    
    Args:
        config: Configuration dictionary containing network hyperparameters
    """
    config: Dict

    @nn.compact
    def __call__(self, x):
        """
        Forward pass through centralized critic network.
        
        Args:
            x: Global observations containing full environment state
            
        Returns:
            State value estimates for the global state
        """
        # Select activation function
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs = x

        # ===== CENTRALIZED CRITIC NETWORK =====
        # Processes global observations for improved credit assignment
        critic = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        critic = activation(critic)
        
        critic = nn.Dense(
            self.config["network"]["critic_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)
        
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        return jnp.squeeze(critic, axis=-1)


# ================================ DATA STRUCTURES ================================

class Transition(NamedTuple):
    """Single transition containing all information needed for MAPPO updates."""
    done: jnp.ndarray              # Individual agent done flags
    all_done: jnp.ndarray          # Environment-level done flags (tiled to all agents)
    action: jnp.ndarray            # Actions taken by each agent
    value: jnp.ndarray             # Critic value estimates (tiled to all agents)
    reward: jnp.ndarray            # Rewards received by each agent
    log_prob: jnp.ndarray          # Log probabilities of actions
    obs: jnp.ndarray               # Local observations for each agent (concatenated)
    global_obs: jnp.ndarray        # Global observations for centralized critic (tiled)
    info: jnp.ndarray              # Environment info (metrics, etc.)
    avail_actions: jnp.ndarray     # Available actions mask


class ActorCriticTrainState(NamedTuple):
    """Combined training state for shared actor and centralized critic networks."""
    actor: TrainState              # Training state for shared actor
    critic: TrainState             # Training state for centralized critic


class RunnerState(NamedTuple):
    """State maintained throughout training/evaluation runs."""
    train_state: ActorCriticTrainState        # Combined actor-critic training state
    env_state: LogEnvState                     # Environment state
    last_obs: Dict[str, jnp.ndarray]          # Most recent observations (local + global)
    last_done: jnp.ndarray                     # Most recent done flags (concatenated)
    update_step: int                           # Current update iteration
    rng: jnp.ndarray                          # Random number generator state


class UpdateState(NamedTuple):
    """State used during network parameter updates."""
    train_state: ActorCriticTrainState        # Combined training state
    traj_batch: Transition                     # Batch of trajectory data
    advantages: jnp.ndarray                   # GAE advantages
    targets: jnp.ndarray                      # Value function targets
    rng: jnp.ndarray                         # Random number generator state


class UpdateBatch(NamedTuple):
    """Batch data structure for minibatch updates."""
    traj_batch: Transition        # Trajectory data
    advantages: jnp.ndarray       # GAE advantages
    targets: jnp.ndarray          # Value function targets


class EvalInfo(NamedTuple):
    """Information logged during evaluation runs."""
    env_state: Optional[LogEnvState]
    done: Optional[jnp.ndarray]
    action: Optional[jnp.ndarray]
    value: Optional[jnp.ndarray]
    reward: Optional[jnp.ndarray]
    log_prob: Optional[jnp.ndarray]
    obs: Optional[jnp.ndarray]
    info: Optional[jnp.ndarray]
    avail_actions: Optional[jnp.ndarray]


@struct.dataclass
class EvalInfoLogConfig:
    """Configuration for what information to log during evaluation."""
    env_state: bool = True
    done: bool = True
    action: bool = True
    value: bool = True
    reward: bool = True
    log_prob: bool = True
    obs: bool = True
    info: bool = True
    avail_actions: bool = True


# ================================ UTILITY FUNCTIONS ================================

def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """
    Convert agent-indexed dictionary to concatenated array for parameter sharing.
    
    With parameter sharing, agents are processed through the same network,
    so we concatenate all agent data into a single batch dimension.
    
    Args:
        qty: Dictionary with agent names as keys
        agents: Ordered sequence of agent names
        
    Returns:
        Concatenated array with all agents in batch dimension
    """
    return jnp.concatenate(tuple(qty[a] for a in agents))


def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """
    Convert concatenated array back to agent-indexed dictionary.
    
    Splits the concatenated batch back into per-agent arrays.
    
    Args:
        qty: Concatenated array with all agents in batch dimension
        agents: Ordered sequence of agent names
        
    Returns:
        Dictionary with agent names as keys
    """
    return dict(zip(agents, jnp.split(qty, len(agents))))


# ================================ TRAINING FUNCTION ================================

def make_train(config, save_train_state=False):
    """
    Create a training function for MAPPO with parameter sharing.
    
    Args:
        config: Configuration dictionary with all hyperparameters
        save_train_state: Whether to save training state in metrics
        
    Returns:
        Compiled training function
    """
    # Environment setup
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Configuration calculations
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))  # Global obs for centralized critic
    env = LogWrapper(env, replace_info=True)

    def linear_schedule(initial_lr):
        """Create linear learning rate schedule."""
        def _linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return initial_lr * frac
        return _linear_schedule

    def train(rng, lr, ent_coef, clip_eps):
        """
        Main training function for MAPPO with parameter sharing.
        
        Args:
            rng: Random number generator key
            lr: Learning rate
            ent_coef: Entropy coefficient
            clip_eps: PPO clipping parameter
            
        Returns:
            Dictionary containing final runner state and training metrics
        """
        # ===== NETWORK INITIALIZATION =====
        # Separate shared actor and centralized critic networks
        actor_network = Actor(config=config)
        critic_network = Critic(config=config)
        rng, actor_network_rng, critic_network_rng = jax.random.split(rng, 3)
        
        # Initialize actor network with concatenated local observations (no agent dimension)
        init_x_actor = (
            jnp.zeros((1, config["OBS_DIM"])),      # local observations (single agent shape)
            jnp.zeros((1,)),                        # done flags
            jnp.zeros((1, config["ACT_DIM"])),      # available actions
        )
        
        # Initialize critic network with global observations
        init_x_critic = jnp.zeros((1, config["GOBS_DIM"]))
        
        actor_network_params = actor_network.init(actor_network_rng, init_x_actor)
        critic_network_params = critic_network.init(critic_network_rng, init_x_critic)
        
        # Optimizer setup for both networks
        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr, eps=config["ADAM_EPS"])
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr, eps=config["ADAM_EPS"])
            )
        
        # PPO clipping parameter adjustments
        if config["SCALE_CLIP_EPS"]:
            clip_eps /= env.num_agents
        
        if config["RATIO_CLIP_EPS"]:
            # Use ratio-based clipping bounds
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0 / (1.0 - clip_eps)
        else:
            # Use symmetric clipping bounds
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0 + clip_eps
        
        # Create training states for shared actor and centralized critic
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )
        
        # Combined training state for MAPPO
        train_state = ActorCriticTrainState(
            actor=actor_train_state,
            critic=critic_train_state,
        )

        # ===== ENVIRONMENT INITIALIZATION =====
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        
        # For parameter sharing, done flags are concatenated (num_agents * num_envs,)
        init_dones = jnp.zeros((env.num_agents * config["NUM_ENVS"],), dtype=bool)

        # ===== MAIN TRAINING LOOP =====
        def _update_step(runner_state, unused):
            """Single update step: collect trajectories and update networks."""
            
            # === TRAJECTORY COLLECTION ===
            def _env_step(runner_state, unused):
                """Single environment step during trajectory collection."""
                rng = runner_state.rng
                
                # Prepare network inputs (concatenated across agents)
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                
                # Actor input (concatenated local observations)
                actor_in = (obs_batch, runner_state.last_done, avail_actions)
                
                # Critic input (global observations)
                critic_in = runner_state.last_obs["global"]
                
                # === ACTION SELECTION (SHARED ACTOR) ===
                pi = runner_state.train_state.actor.apply_fn(
                    runner_state.train_state.actor.params,
                    actor_in,
                )
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                env_act = unbatchify(action, env.agents)

                # === VALUE ESTIMATION (CENTRALIZED CRITIC) ===
                value = runner_state.train_state.critic.apply_fn(
                    runner_state.train_state.critic.params,
                    critic_in,
                )
                # Tile value to all agents (same value for all since centralized)
                value = jnp.tile(value, env.num_agents)

                # Execute environment step
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
                
                # Process outputs (concatenate instead of stack due to parameter sharing)
                done_batch = batchify(done, env.agents)
                all_done = jnp.tile(done["__all__"], env.num_agents)  # Tile environment done to all agents
                info = jax.tree_util.tree_map(jnp.concatenate, info)
                
                transition = Transition(
                    done=done_batch,
                    all_done=all_done,
                    action=action,
                    value=value,
                    reward=batchify(reward, env.agents),
                    log_prob=log_prob,
                    obs=obs_batch,
                    global_obs=jnp.tile(runner_state.last_obs["global"], (env.num_agents, 1)),  # Tile global obs
                    info=info,
                    avail_actions=avail_actions,
                )
                
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # === ADVANTAGE CALCULATION ===
            # Get final value from centralized critic
            critic_in = runner_state.last_obs["global"]
            last_val = runner_state.train_state.critic.apply_fn(
                runner_state.train_state.critic.params,
                critic_in,
            )
            last_val = jnp.tile(last_val, env.num_agents)  # Tile to all agents

            def _calculate_gae(traj_batch, last_val):
                """Calculate Generalized Advantage Estimation using all_done flags."""
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.all_done,  # Use environment-level done flags
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=config["ADVANTAGE_UNROLL_DEPTH"],
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # === NETWORK UPDATES ===
            def _update_epoch(update_state, unused):
                """Single epoch of network parameter updates."""
                
                def _update_minbatch(train_state, batch_info):
                    """Single minibatch parameter update with separate actor/critic losses."""

                    def _actor_loss_fn(params, traj_batch, gae):
                        """PPO loss function for shared actor network."""
                        actor_in = (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                        )
                        pi = train_state.actor.apply_fn(params, actor_in)
                        log_prob = pi.log_prob(traj_batch.action)

                        # PPO clipped objective
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        
                        # Normalize advantages across all agents and timesteps
                        gae = (
                            (gae - gae.mean())
                            / (gae.std() + 1e-8)
                        )
                        
                        pg_loss1 = ratio * gae
                        pg_loss2 = (
                            jnp.clip(
                                ratio,
                                clip_eps_min,
                                clip_eps_max,
                            )
                            * gae
                        )
                        pg_loss = -jnp.minimum(pg_loss1, pg_loss2)
                        pg_loss = pg_loss.mean()
                        entropy = pi.entropy().mean()
                        
                        # Debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac_min = jnp.mean(ratio < clip_eps_min)
                        clip_frac_max = jnp.mean(ratio > clip_eps_max)
                        
                        actor_loss = (
                            pg_loss.sum()
                            - ent_coef * entropy.sum()
                        )
                        return actor_loss, (
                            pg_loss,
                            entropy,
                            approx_kl,
                            clip_frac_min,
                            clip_frac_max,
                        )

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        """Value loss function for centralized critic."""
                        critic_in = traj_batch.global_obs
                        value = train_state.critic.apply_fn(
                            critic_params,
                            critic_in,
                        )
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss,)

                    # Compute gradients and update both networks
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        train_state.actor.params, batch_info.traj_batch, batch_info.advantages
                    )
                    
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        train_state.critic.params, batch_info.traj_batch, batch_info.targets
                    )
                    
                    # Apply gradients to both networks
                    train_state = ActorCriticTrainState(
                        actor=train_state.actor.apply_gradients(grads=actor_grads),
                        critic=train_state.critic.apply_gradients(grads=critic_grads),
                    )

                    loss_info = {
                        "total_loss": actor_loss[0] + critic_loss[0],
                        "actor_loss": actor_loss[1][0],
                        "critic_loss": critic_loss[1][0],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][2],
                        "clip_frac_min": actor_loss[1][3],
                        "clip_frac_max": actor_loss[1][4],
                    }
                    return train_state, loss_info

                rng = update_state.rng

                # Prepare minibatches - includes agent dimension in batch size
                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"] * env.num_agents
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                assert (
                    batch_size % minibatch_size == 0
                ), "Batch size must be divisible by number of minibatches"
                
                batch = UpdateBatch(
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                )
                
                # Shuffle and create minibatches
                # Note: Simpler axis manipulation due to concatenated structure
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                
                # Reshape for minibatch processing
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size, *x.shape[2:])),  # Flatten (step, agent*env, ...)
                    batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    ),
                    shuffled_batch
                )
                
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, update_state.train_state, minibatches
                )
                
                update_state = UpdateState(
                    train_state=train_state,
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    rng=rng,
                )
                return update_state, loss_info

            runner_rng, update_rng = jax.random.split(runner_state.rng)
            update_state = UpdateState(
                train_state=runner_state.train_state,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=update_rng,
            )
            
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            # Prepare metrics
            update_step = runner_state.update_step + 1
            metric = traj_batch.info
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            
            metric = {
                **metric,
                **loss_info,
                "update_step": update_step,
                "env_step": update_step * config["NUM_STEPS"] * config["NUM_ENVS"],
            }
            
            if save_train_state:
                metric.update({"train_state": update_state.train_state})
            
            runner_state = RunnerState(
                train_state=update_state.train_state,
                env_state=runner_state.env_state,
                last_obs=runner_state.last_obs,
                last_done=runner_state.last_done,
                update_step=update_step,
                rng=runner_rng,
            )
            
            return runner_state, metric

        # Initialize training
        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            update_step=0,
            rng=_rng,
        )
        
        # Execute training loop
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ================================ EVALUATION FUNCTION ================================

def make_evaluation(config):
    """
    Create an evaluation function for trained MAPPO agents with parameter sharing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (environment, evaluation_function)
    """
    # Environment setup
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
    env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length

    def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig()):
        """
        Run evaluation episodes with trained shared MAPPO agent.
        
        Args:
            rng: Random number generator key
            train_state: Trained model state (ActorCriticTrainState)
            log_eval_info: Configuration for what to log
            
        Returns:
            Evaluation information from all episodes
        """
        rng_reset, rng_env = jax.random.split(rng)
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        obsv, env_state = jax.vmap(env.reset)(rngs_reset)
        
        # Concatenated done flags for parameter sharing
        init_dones = jnp.zeros((env.num_agents * config["NUM_EVAL_EPISODES"],), dtype=bool)

        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            update_step=0,
            rng=rng_env,
        )

        def _env_step(runner_state, unused):
            """Single environment step during evaluation."""
            rng = runner_state.rng
            
            # Prepare inputs and get actions from shared actor network
            obs_batch = batchify(runner_state.last_obs, env.agents)
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents)
            )
            actor_in = (obs_batch, runner_state.last_done, avail_actions)

            # Get actions from shared actor network
            pi = runner_state.train_state.actor.apply_fn(
                runner_state.train_state.actor.params,
                actor_in,
            )
            rng, act_rng = jax.random.split(rng)
            action, log_prob = pi.sample_and_log_prob(seed=act_rng)
            env_act = unbatchify(action, env.agents)

            # Compute value if requested
            if config.get("eval", {}).get("compute_value", True):
                critic_in = runner_state.last_obs["global"]
                value = runner_state.train_state.critic.apply_fn(
                    runner_state.train_state.critic.params,
                    critic_in,
                )
                value = jnp.tile(value, env.num_agents)
            else:
                value = None

            # Execute environment step
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, runner_state.env_state, env_act,
            )
            
            done_batch = batchify(done, env.agents)
            info = jax.tree_util.tree_map(jnp.concatenate, info)
            
            # Log evaluation information based on configuration
            eval_info = EvalInfo(
                env_state=(env_state if log_eval_info.env_state else None),
                done=(done if log_eval_info.done else None),
                action=(action if log_eval_info.action else None),
                value=(value if log_eval_info.value else None),
                reward=(reward if log_eval_info.reward else None),
                log_prob=(log_prob if log_eval_info.log_prob else None),
                obs=(obs_batch if log_eval_info.obs else None),
                info=(info if log_eval_info.info else None),
                avail_actions=(avail_actions if log_eval_info.avail_actions else None),
            )
            
            runner_state = RunnerState(
                train_state=runner_state.train_state,
                env_state=env_state,
                last_obs=obsv,
                last_done=done_batch,
                update_step=runner_state.update_step,
                rng=rng,
            )
            
            return runner_state, eval_info

        _, eval_info = jax.lax.scan(
            _env_step, runner_state, None, max_steps
        )

        return eval_info
    
    return env, run_evaluation


# ================================ MAIN FUNCTION ================================

@hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax")
def main(config):
    """
    Main function for MAPPO training execution with parameter sharing.
    
    Simple training script for testing and basic execution.
    For more comprehensive experiments, use the runner scripts.
    """
    config = OmegaConf.to_container(config)
    run_rng = jax.random.PRNGKey(config["SEED"])
    
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config),
            device=jax.devices()[config["DEVICE"]]
        )
        out = train_jit(run_rng, config["LR"], config["ENT_COEF"], config["CLIP_EPS"])
        
        print("MAPPO training with parameter sharing completed successfully!")


if __name__ == "__main__":
    main()

# """ 
# Based on the JaxMARL Implementation of MAPPO
# """

# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# from flax import struct
# from flax.linen.initializers import constant, orthogonal
# from flax.training.train_state import TrainState
# import optax
# import distrax
# import jaxmarl
# from jaxmarl.wrappers.baselines import get_space_dim, LogEnvState
# from jaxmarl.wrappers.baselines import LogWrapper
# import hydra
# from omegaconf import OmegaConf
# from typing import Sequence, NamedTuple, Any, Dict, Optional
# import functools

# class Actor(nn.Module):
#     config: Dict

#     @nn.compact
#     def __call__(self, x):
#         if self.config["network"]["activation"] == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh

#         obs, done, avail_actions = x

#         actor_mean = nn.Dense(
#             self.config["network"]["actor_hidden_dim"],
#             kernel_init=orthogonal(jnp.sqrt(2)),
#             bias_init=constant(0.0),
#         )(obs)
#         actor_mean = activation(actor_mean)
#         actor_mean = nn.Dense(
#             self.config["network"]["actor_hidden_dim"],
#             kernel_init=orthogonal(jnp.sqrt(2)),
#             bias_init=constant(0.0)
#         )(actor_mean)
#         actor_mean = activation(actor_mean)
#         actor_mean = nn.Dense(
#             self.config["ACT_DIM"],
#             kernel_init=orthogonal(0.01),
#             bias_init=constant(0.0)
#         )(actor_mean)
#         actor_log_std = self.param(
#             "log_std",
#             nn.initializers.zeros,
#             (self.config["ACT_DIM"],)
#         )
#         pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

#         return pi

# class Critic(nn.Module):
#     config: Dict

#     @nn.compact
#     def __call__(self, x):
#         if self.config["network"]["activation"] == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh

#         obs = x

#         critic = nn.Dense(
#             self.config["network"]["critic_hidden_dim"],
#             kernel_init=orthogonal(jnp.sqrt(2)),
#             bias_init=constant(0.0),
#         )(obs)
#         critic = activation(critic)
#         critic = nn.Dense(
#             self.config["network"]["critic_hidden_dim"],
#             kernel_init=orthogonal(jnp.sqrt(2)),
#             bias_init=constant(0.0),
#         )(critic)
#         critic = activation(critic)
#         critic = nn.Dense(
#             1,
#             kernel_init=orthogonal(1.0),
#             bias_init=constant(0.0)
#         )(critic)

#         return jnp.squeeze(critic, axis=-1)

# class Transition(NamedTuple):
#     done: jnp.ndarray
#     all_done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     log_prob: jnp.ndarray
#     obs: jnp.ndarray
#     global_obs: jnp.ndarray
#     info: jnp.ndarray
#     avail_actions: jnp.ndarray

# class ActorCriticTrainState(NamedTuple):
#     actor: TrainState
#     critic: TrainState

# class RunnerState(NamedTuple):
#     train_state: ActorCriticTrainState
#     env_state: LogEnvState
#     last_obs: Dict[str, jnp.ndarray]
#     last_done: jnp.ndarray
#     update_step: int
#     rng: jnp.ndarray

# class UpdateState(NamedTuple):
#     train_state: ActorCriticTrainState
#     traj_batch: Transition
#     advantages: jnp.ndarray
#     targets: jnp.ndarray
#     rng: jnp.ndarray

# class UpdateBatch(NamedTuple):
#     traj_batch: Transition
#     advantages: jnp.ndarray
#     targets: jnp.ndarray

# class EvalInfo(NamedTuple):
#     env_state: Optional[LogEnvState]
#     done: Optional[jnp.ndarray]
#     action: Optional[jnp.ndarray]
#     value: Optional[jnp.ndarray]
#     reward: Optional[jnp.ndarray]
#     log_prob: Optional[jnp.ndarray]
#     obs: Optional[jnp.ndarray]
#     info: Optional[jnp.ndarray]
#     avail_actions: Optional[jnp.ndarray]

# @struct.dataclass
# class EvalInfoLogConfig:
#     env_state: bool = True
#     done: bool = True
#     action: bool = True
#     value: bool = True
#     reward: bool = True
#     log_prob: bool = True
#     obs: bool = True
#     info: bool = True
#     avail_actions: bool = True

# def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
#     """Convert dict of arrays to batched array."""
#     return jnp.concatenate(tuple(qty[a] for a in agents))

# def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
#     """Convert batched array to dict of arrays."""
#     # N.B. assumes the leading dimension is the agent dimension
#     return dict(zip(agents, jnp.split(qty, len(agents))))

# def make_train(config, save_train_state=False):
#     env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#     config["NUM_UPDATES"] = (
#         config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
#     )
#     config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
#     config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
#     config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
#     env = LogWrapper(env, replace_info=True)

#     def linear_schedule(initial_lr):
#         def _linear_schedule(count):
#             frac = (
#                 1.0
#                 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
#                 / config["NUM_UPDATES"]
#             )
#             return initial_lr * frac
#         return _linear_schedule

#     def train(rng, lr, ent_coef, clip_eps):

#         # INIT NETWORK
#         actor_network = Actor(config=config)
#         critic_network = Critic(config=config)
#         rng, actor_network_rng, critic_network_rng = jax.random.split(rng, 3)
#         init_x_actor = (
#             jnp.zeros( # obs
#                 (1, config["OBS_DIM"])
#             ),
#             jnp.zeros( # done
#                 (1,)
#             ),
#             jnp.zeros( # avail_actions
#                 (1, config["ACT_DIM"])
#             ),
#         )
#         init_x_critic = jnp.zeros((1, config["GOBS_DIM"]))
#         actor_network_params = actor_network.init(actor_network_rng, init_x_actor)
#         critic_network_params = critic_network.init(critic_network_rng, init_x_critic)
#         if config["ANNEAL_LR"]:
#             actor_tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
#             )
#             critic_tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
#             )
#         else:
#             actor_tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(lr, eps=config["ADAM_EPS"])
#             )
#             critic_tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(lr, eps=config["ADAM_EPS"])
#             )
#         if config["SCALE_CLIP_EPS"]:
#             clip_eps /= env.num_agents
#         if config["RATIO_CLIP_EPS"]:
#             clip_eps_min = 1.0 - clip_eps
#             clip_eps_max = 1.0/(1.0 - clip_eps)
#         else:
#             clip_eps_min = 1.0 - clip_eps
#             clip_eps_max = 1.0 + clip_eps
#         actor_train_state = TrainState.create(
#             apply_fn=actor_network.apply,
#             params=actor_network_params,
#             tx=actor_tx,
#         )
#         critic_train_state = TrainState.create(
#             apply_fn=critic_network.apply,
#             params=critic_network_params,
#             tx=critic_tx,
#         )
#         train_state = ActorCriticTrainState(
#             actor=actor_train_state,
#             critic=critic_train_state,
#         )

#         # INIT ENV
#         rng, _rng = jax.random.split(rng)
#         reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
#         obsv, env_state = jax.vmap(env.reset)(reset_rng)
#         init_dones = jnp.zeros((env.num_agents*config["NUM_ENVS"],), dtype=bool)

#         # TRAIN LOOP
#         def _update_step(runner_state, unused):
#             # COLLECT TRAJECTORIES
#             def _env_step(runner_state, unused):
#                 rng = runner_state.rng
#                 obs_batch = batchify(runner_state.last_obs, env.agents)
#                 avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
#                 avail_actions = jax.lax.stop_gradient(
#                     batchify(avail_actions, env.agents)
#                 )
#                 actor_in = (
#                     obs_batch,
#                     runner_state.last_done,
#                     avail_actions
#                 )
#                 critic_in = runner_state.last_obs["global"]
#                 # SELECT ACTION
#                 pi = runner_state.train_state.actor.apply_fn(
#                     runner_state.train_state.actor.params,
#                     actor_in,
#                 )
#                 rng, act_rng = jax.random.split(rng)
#                 action, log_prob = pi.sample_and_log_prob(seed=act_rng)
#                 env_act = unbatchify(action, env.agents)

#                 # COMPUTE VALUE
#                 value = runner_state.train_state.critic.apply_fn(
#                     runner_state.train_state.critic.params,
#                     critic_in,
#                 )
#                 value = jnp.tile(value, env.num_agents)

#                 # STEP ENV
#                 rng, _rng = jax.random.split(rng)
#                 rng_step = jax.random.split(_rng, config["NUM_ENVS"])
#                 obsv, env_state, reward, done, info = jax.vmap(env.step)(
#                     rng_step, runner_state.env_state, env_act,
#                 )
#                 done_batch = batchify(done, env.agents)
#                 all_done = jnp.tile(done["__all__"], env.num_agents)
#                 info = jax.tree_util.tree_map(jnp.concatenate, info)
#                 transition = Transition(
#                     done=done_batch,
#                     all_done=all_done,
#                     action=action,
#                     value=value,
#                     reward=batchify(reward, env.agents),
#                     log_prob=log_prob,
#                     obs=obs_batch,
#                     global_obs=jnp.tile(runner_state.last_obs["global"], (env.num_agents,1)),
#                     info=info,
#                     avail_actions=avail_actions,
#                 )
#                 runner_state = RunnerState(
#                     train_state=runner_state.train_state,
#                     env_state=env_state,
#                     last_obs=obsv,
#                     last_done=done_batch,
#                     update_step=runner_state.update_step,
#                     rng=rng,
#                 )
#                 return runner_state, transition

#             runner_state, traj_batch = jax.lax.scan(
#                 _env_step, runner_state, None, config["NUM_STEPS"]
#             )

#             # CALCULATE ADVANTAGE
#             critic_in = runner_state.last_obs["global"]
#             last_val = runner_state.train_state.critic.apply_fn(
#                 runner_state.train_state.critic.params,
#                 critic_in,
#             )
#             last_val = jnp.tile(last_val, env.num_agents)

#             def _calculate_gae(traj_batch, last_val):
#                 def _get_advantages(gae_and_next_value, transition):
#                     gae, next_value = gae_and_next_value
#                     done, value, reward = (
#                         transition.all_done,
#                         transition.value,
#                         transition.reward,
#                     )
#                     delta = reward + config["GAMMA"] * next_value * (1 - done) - value
#                     gae = (
#                         delta
#                         + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
#                     )
#                     return (gae, value), gae

#                 _, advantages = jax.lax.scan(
#                     _get_advantages,
#                     (jnp.zeros_like(last_val), last_val),
#                     traj_batch,
#                     reverse=True,
#                     unroll=config["ADVANTAGE_UNROLL_DEPTH"],
#                 )
#                 return advantages, advantages + traj_batch.value

#             advantages, targets = _calculate_gae(traj_batch, last_val)

#             # UPDATE NETWORK
#             def _update_epoch(update_state, unused):
#                 def _update_minbatch(train_state, batch_info):
#                     def _actor_loss_fn(params, traj_batch, gae):
#                         # RERUN NETWORK
#                         actor_in = (
#                             traj_batch.obs,
#                             traj_batch.done,
#                             traj_batch.avail_actions,
#                         )
#                         pi = train_state.actor.apply_fn(params, actor_in)
#                         log_prob = pi.log_prob(traj_batch.action)

#                         # CALCULATE ACTOR LOSS
#                         logratio = log_prob - traj_batch.log_prob
#                         ratio = jnp.exp(logratio)
#                         gae = (
#                             (gae - gae.mean())
#                             / (gae.std() + 1e-8)
#                         )
#                         pg_loss1 = ratio * gae
#                         pg_loss2 = (
#                             jnp.clip(
#                                 ratio,
#                                 clip_eps_min,
#                                 clip_eps_max,
#                             )
#                             * gae
#                         )
#                         pg_loss = -jnp.minimum(pg_loss1, pg_loss2)
#                         pg_loss = pg_loss.mean()
#                         entropy = pi.entropy().mean()
#                         # debug metrics
#                         approx_kl = ((ratio - 1) - logratio).mean()
#                         clip_frac_min = jnp.mean(ratio < clip_eps_min)
#                         clip_frac_max = jnp.mean(ratio > clip_eps_max)
#                         # ---
#                         actor_loss = (
#                             pg_loss.sum()
#                             - ent_coef * entropy.sum()
#                         )
#                         return actor_loss, (
#                             pg_loss,
#                             entropy,
#                             approx_kl,
#                             clip_frac_min,
#                             clip_frac_max,
#                         )

#                     def _critic_loss_fn(critic_params, traj_batch, targets):
#                         critic_in = traj_batch.global_obs
#                         value = train_state.critic.apply_fn(
#                             critic_params,
#                             critic_in,
#                         )
#                         value_losses = jnp.square(value - targets)
#                         value_loss = 0.5 * value_losses.mean()
#                         critic_loss =  config["VF_COEF"] * value_loss
#                         return critic_loss, (value_loss,)

#                     actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
#                     actor_loss, actor_grads = actor_grad_fn(
#                         train_state.actor.params, batch_info.traj_batch, batch_info.advantages
#                     )
#                     critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
#                     critic_loss, critic_grads = critic_grad_fn(
#                         train_state.critic.params, batch_info.traj_batch, batch_info.targets
#                     )
#                     train_state = ActorCriticTrainState(
#                         actor = train_state.actor.apply_gradients(grads=actor_grads),
#                         critic = train_state.critic.apply_gradients(grads=critic_grads),
#                     )

#                     loss_info = {
#                         "total_loss": actor_loss[0] + critic_loss[0],
#                         "actor_loss": actor_loss[1][0],
#                         "critic_loss": critic_loss[1][0],
#                         "entropy": actor_loss[1][1],
#                         "approx_kl": actor_loss[1][2],
#                         "clip_frac_min": actor_loss[1][3],
#                         "clip_frac_max": actor_loss[1][4],
#                     }
#                     return train_state, loss_info

#                 rng = update_state.rng

#                 batch_size = config["NUM_STEPS"] * config["NUM_ENVS"] * env.num_agents
#                 minibatch_size = batch_size // config["NUM_MINIBATCHES"]
#                 assert (
#                     batch_size % minibatch_size == 0
#                 ), "unable to equally partition into minibatches"
#                 batch = UpdateBatch(
#                     traj_batch=update_state.traj_batch,
#                     advantages=update_state.advantages,
#                     targets=update_state.targets,
#                 )
#                 rng, _rng = jax.random.split(rng)
#                 permutation = jax.random.permutation(_rng, batch_size)
#                 # initial axes: (step, agent*env, ...)
#                 batch = jax.tree_util.tree_map(
#                     lambda x: x.reshape((batch_size, *x.shape[2:])),
#                     batch
#                 ) # reshape axes to (step*env*agent, ...)
#                 shuffled_batch = jax.tree_util.tree_map(
#                     lambda x: jnp.take(x, permutation, axis=0),
#                     batch
#                 ) # shuffle: maintains axes (step*env*agent, ...)
#                 minibatches = jax.tree_util.tree_map(
#                     lambda x: jnp.reshape(
#                         x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
#                     ),
#                     shuffled_batch
#                 ) # split into minibatches. axes (n_mini, minibatch_size, ...)
#                 train_state, loss_info = jax.lax.scan(
#                     _update_minbatch, update_state.train_state, minibatches
#                 )
#                 update_state = UpdateState(
#                     train_state=train_state,
#                     traj_batch=update_state.traj_batch,
#                     advantages=update_state.advantages,
#                     targets=update_state.targets,
#                     rng=rng,
#                 )
#                 return update_state, loss_info

#             runner_rng, update_rng = jax.random.split(runner_state.rng)
#             update_state = UpdateState(
#                 train_state=runner_state.train_state,
#                 traj_batch=traj_batch,
#                 advantages=advantages,
#                 targets=targets,
#                 rng=update_rng,
#             )
#             update_state, loss_info = jax.lax.scan(
#                 _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
#             )
#             update_step = runner_state.update_step + 1
#             metric = traj_batch.info
#             metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
#             loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
#             metric = {
#                 **metric,
#                 **loss_info,
#                 "update_step": update_step,
#                 "env_step": update_step * config["NUM_STEPS"] * config["NUM_ENVS"],
#             }
#             if save_train_state:
#                 metric.update({"train_state": update_state.train_state})
#             runner_state = RunnerState(
#                 train_state=update_state.train_state,
#                 env_state=runner_state.env_state,
#                 last_obs=runner_state.last_obs,
#                 last_done=runner_state.last_done,
#                 update_step=update_step,
#                 rng=runner_rng,
#             )
#             return runner_state, metric

#         rng, _rng = jax.random.split(rng)
#         runner_state = RunnerState(
#             train_state=train_state,
#             env_state=env_state,
#             last_obs=obsv,
#             last_done=init_dones,
#             update_step=0,
#             rng=_rng,
#         )
#         runner_state, metric = jax.lax.scan(
#             _update_step, runner_state, None, config["NUM_UPDATES"]
#         )
#         return {"runner_state": runner_state, "metrics": metric}

#     return train

# def make_evaluation(config):
#     env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#     config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
#     config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
#     config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
#     env = LogWrapper(env, replace_info=True)
#     max_steps = env.episode_length

#     def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig()):
#         rng_reset, rng_env = jax.random.split(rng)
#         rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
#         obsv, env_state = jax.vmap(env.reset)(rngs_reset)
#         init_dones = jnp.zeros((env.num_agents*config["NUM_EVAL_EPISODES"],), dtype=bool)

#         runner_state = RunnerState(
#             train_state=train_state,
#             env_state=env_state,
#             last_obs=obsv,
#             last_done=init_dones,
#             update_step=0,
#             rng=rng_env,
#         )

#         def _env_step(runner_state, unused):
#             rng = runner_state.rng
#             obs_batch = batchify(runner_state.last_obs, env.agents)
#             avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
#             avail_actions = jax.lax.stop_gradient(
#                 batchify(avail_actions, env.agents)
#             )
#             actor_in = (
#                 obs_batch,
#                 runner_state.last_done,
#                 avail_actions
#             )

#             # SELECT ACTION
#             pi = runner_state.train_state.apply_fn(
#                 runner_state.train_state.params,
#                 actor_in,
#             )
#             rng, act_rng = jax.random.split(rng)
#             action, log_prob = pi.sample_and_log_prob(seed=act_rng)
#             env_act = unbatchify(action, env.agents)

#             # COMPUTE VALUE
#             if config["env"]["compute_value"]:
#                 critic_in = runner_state.last_obs["global"]
#                 value = runner_state.train_state.critic.apply_fn(
#                     runner_state.train_state.critic.params,
#                     critic_in,
#                 )
#                 value = jnp.tile(value, env.num_agents)
#             else:
#                 value = None

#             # STEP ENV
#             rng, _rng = jax.random.split(rng)
#             rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
#             obsv, env_state, reward, done, info = jax.vmap(env.step)(
#                 rng_step, runner_state.env_state, env_act,
#             )
#             done_batch = batchify(done, env.agents)
#             info = jax.tree_util.tree_map(jnp.concatenate, info)
#             eval_info = EvalInfo(
#                 env_state=(env_state if log_eval_info.env_state else None),
#                 done=(done if log_eval_info.done else None),
#                 action=(action if log_eval_info.action else None),
#                 value=(value if log_eval_info.value else None),
#                 reward=(reward if log_eval_info.reward else None),
#                 log_prob=(log_prob if log_eval_info.log_prob else None),
#                 obs=(obs_batch if log_eval_info.obs else None),
#                 info=(info if log_eval_info.info else None),
#                 avail_actions=(avail_actions if log_eval_info.avail_actions else None),
#             )
#             runner_state = RunnerState(
#                 train_state=runner_state.train_state,
#                 env_state=env_state,
#                 last_obs=obsv,
#                 last_done=done_batch,
#                 update_step=runner_state.update_step,
#                 rng=rng,
#             )
#             return runner_state, eval_info

#         _, eval_info = jax.lax.scan(
#             _env_step, runner_state, None, max_steps
#         )

#         return eval_info
#     return env, run_evaluation


# @hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax")
# def main(config):
#     config = OmegaConf.to_container(config)
#     run_rng = jax.random.PRNGKey(config["SEED"])
#     with jax.disable_jit(config["DISABLE_JIT"]):
#         train_jit = jax.jit(
#             make_train(config),
#             device=jax.devices()[config["DEVICE"]]
#         )
#         out = train_jit(run_rng, config["LR"], config["ENT_COEF"], config["CLIP_EPS"])
#         breakpoint()


# if __name__ == "__main__":
#     main()
