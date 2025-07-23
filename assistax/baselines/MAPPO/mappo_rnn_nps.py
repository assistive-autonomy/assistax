"""
Multi-Agent Proximal Policy Optimization (MAPPO) with RNN and No Parameter Sharing

This implementation provides a clean, research-friendly MAPPO algorithm with:
- Individual actor networks for each agent (no parameter sharing)
- Centralized critic with global state information
- GRU-based recurrent networks for both actor and critic
- GAE (Generalized Advantage Estimation) for advantage computation
- PPO clipping for stable policy updates

Based on the JaxMARL Implementation of MAPPO.
"""

import functools
from typing import Sequence, NamedTuple, Any, Dict, Optional

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import hydra
from omegaconf import OmegaConf

import assistax
from assistax.wrappers.baselines import get_space_dim, LogEnvState, LogWrapper
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper


# ============================================================================
# NEURAL NETWORK ARCHITECTURES
# ============================================================================

class ScannedRNN(nn.Module):
    """
    Scanned RNN cell that processes sequences using jax.lax.scan.
    
    This wrapper enables efficient processing of variable-length sequences
    by scanning over the time dimension while handling episode resets.
    """
    
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """
        Process a single timestep of the RNN.
        
        Args:
            carry: Hidden state from previous timestep
            x: Tuple of (input, reset_flags)
            
        Returns:
            Tuple of (new_hidden_state, output)
        """
        rnn_state = carry
        ins, resets = x
        
        # Reset hidden state when episode ends
        rnn_state = jnp.where(
            jnp.expand_dims(resets, -1),  # Add feature dimension to reset flags
            self.initialize_carry(rnn_state.shape),
            rnn_state
        )
        
        # Apply GRU cell
        new_rnn_state, output = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, output

    @staticmethod
    def initialize_carry(hidden_shape):
        """Initialize hidden state for GRU cell."""
        hidden_size = hidden_shape[-1]
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), hidden_shape)


@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiActorRNN(nn.Module):
    """
    Multi-agent actor network with individual parameters for each agent.
    
    Each agent has its own set of parameters (no parameter sharing).
    Uses GRU for recurrent processing and outputs continuous action distributions.
    """
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        """
        Forward pass through actor network.
        
        Args:
            hstate: Hidden state from previous timestep
            x: Tuple of (observations, done_flags, available_actions)
            
        Returns:
            Tuple of (new_hidden_state, (action_mean, action_std))
        """
        # Select activation function
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        # Observation embedding
        embedding = nn.Dense(
            self.config["network"]["embedding_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        # RNN processing
        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)

        # Policy head (outputs action distribution parameters)
        actor_mean = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
        actor_mean = activation(actor_mean)
        
        # Final action layer
        actor_mean = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),  # Small init for stable policy
            bias_init=constant(0.0)
        )(actor_mean)
        
        # Learnable log standard deviation (shared across all actions)
        actor_log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.config["ACT_DIM"],)
        )
        
        pi = (actor_mean, jnp.exp(actor_log_std))
        return hstate, pi


class CriticRNN(nn.Module):
    """
    Centralized critic network that uses global state information.
    
    Processes global observations to estimate state values for advantage computation.
    Uses GRU for recurrent processing.
    """
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        """
        Forward pass through critic network.
        
        Args:
            hstate: Hidden state from previous timestep
            x: Tuple of (global_observations, done_flags)
            
        Returns:
            Tuple of (new_hidden_state, value_estimate)
        """
        # Select activation function
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done = x

        # Global observation embedding
        embedding = nn.Dense(
            self.config["network"]["embedding_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        # RNN processing
        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)
        
        # Value head
        critic = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        
        # Final value output
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        return hstate, jnp.squeeze(critic, axis=-1)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class Transition(NamedTuple):
    """Single transition data structure for PPO training."""
    done: jnp.ndarray              # Agent-specific done flags
    all_done: jnp.ndarray          # Environment-level done flag
    action: jnp.ndarray            # Actions taken
    value: jnp.ndarray             # Value estimates
    reward: jnp.ndarray            # Rewards received
    log_prob: jnp.ndarray          # Action log probabilities
    obs: jnp.ndarray               # Agent observations
    global_obs: jnp.ndarray        # Global observations
    info: jnp.ndarray              # Environment info
    avail_actions: jnp.ndarray     # Available actions mask


class ActorCriticTrainState(NamedTuple):
    """Training state for actor-critic networks."""
    actor: TrainState
    critic: TrainState


class ActorCriticHiddenState(NamedTuple):
    """Hidden states for actor-critic networks."""
    actor: jnp.ndarray
    critic: jnp.ndarray


class RunnerState(NamedTuple):
    """State maintained by the training runner."""
    train_state: ActorCriticTrainState
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    last_all_done: jnp.ndarray
    hstate: ActorCriticHiddenState
    update_step: int
    rng: jnp.ndarray


class UpdateState(NamedTuple):
    """State used during network updates."""
    train_state: ActorCriticTrainState
    init_hstate: ActorCriticHiddenState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray


class UpdateBatch(NamedTuple):
    """Batch data for network updates."""
    init_hstate: ActorCriticHiddenState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray


class EvalInfo(NamedTuple):
    """Information logged during evaluation."""
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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """
    Convert dictionary of agent-specific arrays to single batched array.
    
    Args:
        qty: Dictionary mapping agent names to arrays
        agents: Sequence of agent names defining the order
        
    Returns:
        Batched array with agent dimension first
    """
    return jnp.stack(tuple(qty[a] for a in agents))


def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """
    Convert batched array back to dictionary of agent-specific arrays.
    
    Args:
        qty: Batched array with agent dimension first
        agents: Sequence of agent names
        
    Returns:
        Dictionary mapping agent names to arrays
    """
    return dict(zip(agents, qty))


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def make_train(config, save_train_state=False, load_zoo=False):
    """
    Create the main training function for MAPPO.
    
    Args:
        config: Training configuration dictionary
        save_train_state: Whether to save training state in metrics
        load_zoo: Whether to load pre-trained agents from zoo
        
    Returns:
        training function
    """
    # Initialize environment
    if load_zoo:
        zoo = ZooManager(config["ZOO_PATH"])
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
    else:
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Calculate derived configuration values
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
    
    # Wrap environment for logging
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
        Main training function for MAPPO.
        
        Args:
            rng: Random number generator key
            lr: Learning rate
            ent_coef: Entropy coefficient
            clip_eps: PPO clipping parameter
            
        Returns:
            Dictionary containing final runner state and training metrics
        """
        
        # ====================================================================
        # INITIALIZE NETWORKS
        # ====================================================================
        
        actor_network = MultiActorRNN(config=config)
        critic_network = CriticRNN(config=config)
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        
        # Initialize dummy inputs for network parameter initialization
        init_x_actor = (
            jnp.zeros((env.num_agents, 1, config["NUM_ENVS"], config["OBS_DIM"])),  # obs
            jnp.zeros((env.num_agents, 1, config["NUM_ENVS"])),                     # done
            jnp.zeros((env.num_agents, 1, config["NUM_ENVS"], config["ACT_DIM"])),  # avail_actions
        )
        init_hstate_actor = jnp.zeros(
            (env.num_agents, config["NUM_ENVS"], config["network"]["gru_hidden_dim"])
        )
        init_x_critic = (
            jnp.zeros((1, config["NUM_ENVS"], config["GOBS_DIM"])),  # global_obs
            jnp.zeros((1, config["NUM_ENVS"])),                      # done
        )
        init_hstate_critic = jnp.zeros(
            (config["NUM_ENVS"], config["network"]["gru_hidden_dim"])
        )
        
        # Initialize network parameters
        actor_network_params = actor_network.init(actor_rng, init_hstate_actor, init_x_actor)
        critic_network_params = critic_network.init(critic_rng, init_hstate_critic, init_x_critic)
        
        # Setup optimizers
        if config["ANNEAL_LR"]:
            # Use linear learning rate schedule
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
        else:
            # Use constant learning rate
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr, eps=config["ADAM_EPS"])
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr, eps=config["ADAM_EPS"])
            )
        
        # Configure PPO clipping
        if config["SCALE_CLIP_EPS"]:
            clip_eps /= env.num_agents
        
        if config["RATIO_CLIP_EPS"]:
            # Asymmetric clipping for ratio
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0 / (1.0 - clip_eps)
        else:
            # Symmetric clipping for ratio
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0 + clip_eps
        
        # Create training states
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
        train_state = ActorCriticTrainState(
            actor=actor_train_state,
            critic=critic_train_state,
        )

        # ====================================================================
        # INITIALIZE ENVIRONMENT
        # ====================================================================
        
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_dones = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool)
        init_all_dones = jnp.zeros((1, config["NUM_ENVS"]), dtype=bool)

        # ====================================================================
        # MAIN TRAINING LOOP
        # ====================================================================
        
        def _update_step(runner_state, unused):
            """Single update step: collect trajectories and update networks."""
            
            # ================================================================
            # COLLECT TRAJECTORIES
            # ================================================================
            
            def _env_step(runner_state, unused):
                """Single environment step during trajectory collection."""
                rng = runner_state.rng
                
                # Prepare observations and available actions
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                
                # Prepare network inputs (add time dimension for RNN)
                actor_in = (
                    jnp.expand_dims(obs_batch, 1),
                    jnp.expand_dims(runner_state.last_done, 1),
                    jnp.expand_dims(avail_actions, 1),
                )
                critic_in = (
                    jnp.expand_dims(runner_state.last_obs["global"], 0),
                    jnp.expand_dims(runner_state.last_all_done.squeeze(0), 0),
                )

                # ============================================================
                # SELECT ACTIONS
                # ============================================================
                
                actor_hstate, (actor_mean, actor_std) = runner_state.train_state.actor.apply_fn(
                    runner_state.train_state.actor.params,
                    runner_state.hstate.actor, actor_in,
                )
                
                # Remove time dimension and prepare for sampling
                actor_mean = actor_mean.squeeze(1)
                actor_std = jnp.expand_dims(actor_std, axis=1)  # Add env batch dim
                
                # Sample actions from policy
                pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                env_act = unbatchify(action, env.agents)

                # ============================================================
                # COMPUTE VALUES
                # ============================================================
                
                critic_hstate, value = runner_state.train_state.critic.apply_fn(
                    runner_state.train_state.critic.params,
                    runner_state.hstate.critic, critic_in,
                )
                value = value.squeeze(0)  # Remove time dimension
                value = jnp.broadcast_to(value, (env.num_agents, *value.shape))

                # ============================================================
                # STEP ENVIRONMENT
                # ============================================================
                
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
                
                # Process environment outputs
                done_batch = batchify(done, env.agents)
                all_done = jnp.expand_dims(done["__all__"], 0)
                info = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), info)
                
                # Store transition
                transition = Transition(
                    done=done_batch,
                    all_done=all_done,
                    action=action,
                    value=value,
                    reward=batchify(reward, env.agents),
                    log_prob=log_prob,
                    obs=obs_batch,
                    global_obs=jnp.expand_dims(runner_state.last_obs["global"], axis=0),
                    info=info,
                    avail_actions=avail_actions,
                )
                
                # Update runner state
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    last_all_done=all_done,
                    hstate=ActorCriticHiddenState(actor=actor_hstate, critic=critic_hstate),
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                return runner_state, transition

            # Collect trajectory batch
            init_hstate = runner_state.hstate
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # ================================================================
            # COMPUTE ADVANTAGES USING GAE
            # ================================================================
            
            # Get final value estimate for bootstrap
            critic_in = (
                jnp.expand_dims(runner_state.last_obs["global"], 0),
                jnp.expand_dims(runner_state.last_all_done.squeeze(0), 0),
            )
            _, last_val = runner_state.train_state.critic.apply_fn(
                runner_state.train_state.critic.params,
                runner_state.hstate.critic, critic_in,
            )
            last_val = last_val.squeeze(0)  # Remove time dim
            last_val = jnp.broadcast_to(last_val, (env.num_agents, *last_val.shape))

            def _calculate_gae(traj_batch, last_val):
                """Calculate Generalized Advantage Estimation."""
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.all_done,
                        transition.value,
                        transition.reward,
                    )
                    # Temporal difference error
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    # GAE computation
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

            # ================================================================
            # UPDATE NETWORKS
            # ================================================================
            
            def _update_epoch(update_state, unused):
                """Single epoch of network updates."""
                
                def _update_minibatch(train_state, batch_info):
                    """Update networks on a single minibatch."""
                    
                    def _actor_loss_fn(actor_params, init_actor_hstate, traj_batch, gae):
                        """Compute actor loss (PPO objective)."""
                        # Re-run actor network
                        actor_in = (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                        )
                        _, (actor_mean, actor_std) = train_state.actor.apply_fn(
                            actor_params,
                            init_actor_hstate.squeeze(1),  # Remove step dim
                            actor_in,
                        )
                        actor_std = jnp.expand_dims(actor_std, axis=(1, 2))  # Add step & env dims
                        pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Compute PPO loss
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        
                        # Normalize advantages
                        gae = (
                            (gae - gae.mean(axis=(-2, -1), keepdims=True))
                            / (gae.std(axis=(-2, -1), keepdims=True) + 1e-8)
                        )
                        
                        # PPO clipped objective
                        pg_loss1 = ratio * gae
                        pg_loss2 = (
                            jnp.clip(ratio, clip_eps_min, clip_eps_max) * gae
                        )
                        pg_loss = -jnp.minimum(pg_loss1, pg_loss2)
                        pg_loss = pg_loss.mean(axis=(-2, -1))
                        
                        # Entropy bonus
                        entropy = pi.entropy().mean(axis=(-2, -1))
                        
                        # Debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean(axis=(-2, -1))
                        clip_frac_min = jnp.mean(ratio < clip_eps_min, axis=(-2, -1))
                        clip_frac_max = jnp.mean(ratio > clip_eps_max, axis=(-2, -1))
                        
                        # Total actor loss
                        actor_loss = pg_loss.sum() - ent_coef * entropy.sum()
                        
                        return actor_loss, (
                            pg_loss, entropy, approx_kl, clip_frac_min, clip_frac_max
                        )

                    def _critic_loss_fn(critic_params, init_critic_hstate, traj_batch, targets):
                        """Compute critic loss (value function MSE)."""
                        # Re-run critic network
                        critic_in = (
                            traj_batch.global_obs.squeeze(0),
                            traj_batch.all_done.squeeze(0),
                        )
                        _, value = train_state.critic.apply_fn(
                            critic_params,
                            init_critic_hstate.squeeze((0, 1)),  # Remove (ag, step) dims
                            critic_in,
                        )
                        
                        # Value function loss
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean()
                        critic_loss = config["VF_COEF"] * value_loss
                        
                        return critic_loss, (value_loss,)

                    # Compute gradients
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        train_state.actor.params,
                        batch_info.init_hstate.actor,
                        batch_info.traj_batch,
                        batch_info.advantages,
                    )
                    
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        train_state.critic.params,
                        batch_info.init_hstate.critic,
                        batch_info.traj_batch,
                        batch_info.targets,
                    )

                    # Apply gradients
                    train_state = ActorCriticTrainState(
                        actor=train_state.actor.apply_gradients(grads=actor_grads),
                        critic=train_state.critic.apply_gradients(grads=critic_grads),
                    )
                    
                    # Collect loss information
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

                # Prepare minibatches
                rng = update_state.rng
                batch_size = config["NUM_ENVS"]
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                assert (
                    batch_size % minibatch_size == 0
                ), "Batch size must be divisible by number of minibatches"
                
                # Create batch with added dimensions for RNN processing
                batch = UpdateBatch(
                    init_hstate=ActorCriticHiddenState(
                        actor=jnp.expand_dims(update_state.init_hstate.actor, 0),      # Add step dim
                        critic=jnp.expand_dims(update_state.init_hstate.critic, (0, 1)), # Add (step, ag) dims
                    ),
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                )
                
                # Shuffle and split into minibatches
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                
                # Reorganize batch: (step, agent, env, ...) -> (env, agent, step, ...)
                batch = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 2), batch)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])),
                    shuffled_batch
                )
                # Final reorganization: (n_mini, minibatch_size, agent, step, ...) -> (n_mini, agent, step, minibatch_size, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.moveaxis(x, 1, 3), minibatches
                )
                
                # Process minibatches
                train_state, loss_info = jax.lax.scan(
                    _update_minibatch, update_state.train_state, minibatches
                )
                
                # Update state
                update_state = UpdateState(
                    train_state=train_state,
                    init_hstate=update_state.init_hstate,
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    rng=rng,
                )
                return update_state, loss_info

            # Run multiple epochs of updates
            runner_rng, update_rng = jax.random.split(runner_state.rng)
            update_state = UpdateState(
                train_state=runner_state.train_state,
                init_hstate=init_hstate,
                traj_batch=traj_batch,
                advantages=advantages,
                targets=targets,
                rng=update_rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            
            # ================================================================
            # COLLECT METRICS AND UPDATE RUNNER STATE
            # ================================================================
            
            update_step = runner_state.update_step + 1
            
            # Process environment metrics
            metric = traj_batch.info
            metric = jax.tree_util.tree_map(lambda x: x.mean(axis=(0, 2)), metric)
            
            # Process loss metrics
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(axis=(0, 1)), loss_info)
            
            # Combine all metrics
            metric = {
                **metric,
                **loss_info,
                "update_step": update_step,
                "env_step": update_step * config["NUM_STEPS"] * config["NUM_ENVS"],
            }
            
            # Optionally save training state
            if save_train_state:
                metric.update({"train_state": update_state.train_state})
            
            # Update runner state
            runner_state = RunnerState(
                train_state=update_state.train_state,
                env_state=runner_state.env_state,
                last_obs=runner_state.last_obs,
                last_done=runner_state.last_done,
                last_all_done=runner_state.last_all_done,
                hstate=runner_state.hstate,
                update_step=update_step,
                rng=runner_rng,
            )
            return runner_state, metric

        # ====================================================================
        # INITIALIZE RUNNER AND START TRAINING
        # ====================================================================
        
        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            last_all_done=init_all_dones,
            hstate=ActorCriticHiddenState(
                actor=init_hstate_actor,
                critic=init_hstate_critic,
            ),
            update_step=0,
            rng=_rng,
        )
        
        # Run training loop
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        
        return {"runner_state": runner_state, "metrics": metric}

    return train


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def make_evaluation(config, load_zoo=False):
    """
    Create evaluation function for trained MAPPO agents.
    
    Args:
        config: Configuration dictionary
        load_zoo: Whether to load agents from zoo
        
    Returns:
        Tuple of (environment, evaluation_function)
    """
    # Initialize environment (same as training)
    if load_zoo:
        zoo = ZooManager(config["ZOO_PATH"])
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
    else:
        env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Set configuration parameters
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    config["GOBS_DIM"] = get_space_dim(env.observation_space("global"))
    env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length

    def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig()):
        """
        Run evaluation episodes with trained agents.
        
        Args:
            rng: Random number generator key
            train_state: Trained actor-critic state
            log_eval_info: Configuration for what to log
            
        Returns:
            Evaluation information across all episodes
        """
        # Initialize evaluation episodes
        rng_reset, rng_env = jax.random.split(rng)
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        obsv, env_state = jax.vmap(env.reset)(rngs_reset)
        
        # Initialize states
        init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"]), dtype=bool)
        init_all_dones = jnp.zeros((1, config["NUM_EVAL_EPISODES"]), dtype=bool)
        init_hstate_actor = jnp.zeros(
            (env.num_agents, config["NUM_EVAL_EPISODES"], config["network"]["gru_hidden_dim"])
        )
        init_hstate_critic = jnp.zeros(
            (config["NUM_EVAL_EPISODES"], config["network"]["gru_hidden_dim"])
        )
        
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            last_all_done=init_all_dones,
            hstate=ActorCriticHiddenState(
                actor=init_hstate_actor, 
                critic=init_hstate_critic
            ),
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
            
            # Prepare actor input
            actor_in = (
                jnp.expand_dims(obs_batch, 1),
                jnp.expand_dims(runner_state.last_done, 1),
                jnp.expand_dims(avail_actions, 1),
            )

            # Select actions
            actor_hstate, (actor_mean, actor_std) = runner_state.train_state.actor.apply_fn(
                runner_state.train_state.actor.params,
                runner_state.hstate.actor, actor_in,
            )
            
            actor_mean = actor_mean.squeeze(1)
            actor_std = jnp.expand_dims(actor_std, axis=1)
            pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
            rng, act_rng = jax.random.split(rng)
            action, log_prob = pi.sample_and_log_prob(seed=act_rng)
            env_act = unbatchify(action, env.agents)

            # Compute values if requested
            if config["eval"]["compute_value"]:
                critic_in = (
                    jnp.expand_dims(runner_state.last_obs["global"], 0),
                    jnp.expand_dims(runner_state.last_all_done.squeeze(0), 0),
                )
                critic_hstate, value = runner_state.train_state.critic.apply_fn(
                    runner_state.train_state.critic.params,
                    runner_state.hstate.critic, critic_in,
                )
                value = value.squeeze(0)
                value = jnp.broadcast_to(value, (env.num_agents, *value.shape))
            else:
                value = None

            # Step environment
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config["NUM_EVAL_EPISODES"])
            obsv, env_state, reward, done, info = jax.vmap(env.step)(
                rng_step, runner_state.env_state, env_act,
            )
            
            # Process outputs
            done_batch = batchify(done, env.agents)
            all_done = jnp.expand_dims(done["__all__"], 0)
            info = jax.tree_util.tree_map(lambda x: x.swapaxes(0, 1), info)
            
            # Log evaluation info based on configuration
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
            
            # Update runner state
            runner_state = RunnerState(
                train_state=runner_state.train_state,
                env_state=env_state,
                last_obs=obsv,
                last_done=done_batch,
                last_all_done=all_done,
                hstate=ActorCriticHiddenState(
                    actor=actor_hstate, 
                    critic=critic_hstate
                ),
                update_step=runner_state.update_step,
                rng=rng,
            )
            return runner_state, eval_info
            
        # Run evaluation
        _, eval_info = jax.lax.scan(_env_step, runner_state, None, max_steps)
        return eval_info
        
    return env, run_evaluation


