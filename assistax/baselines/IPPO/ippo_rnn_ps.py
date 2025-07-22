"""
Independent Proximal Policy Optimization (IPPO) Implementation with Recurrent Neural Networks and Parameter Sharing

This module implements IPPO for multi-agent reinforcement learning using JAX/Flax with recurrent neural networks
where all agents share the same network parameters. This combines the memory and temporal processing capabilities
of RNNs with the memory efficiency and sample efficiency benefits of parameter sharing.

Key Features:
- Shared GRU-based actor-critic network across all agents
- Memory and temporal processing through recurrent connections
- PPO with GAE for advantage estimation
- Hidden state management across episode boundaries
- Support for partially observable environments
- Reduced memory usage compared to independent RNN networks
- Hyperparameter sweeping with vmapped training

Differences from Non-Parameter Sharing RNN Version:
- Single ActorCriticRNN network instead of vmapped MultiActorCriticRNN
- Agents concatenated rather than stacked for batching
- Shared hidden state processing through same network weights
- Simpler network initialization (no agent dimension)
- Policy distribution created directly in network
- More efficient memory usage

Differences from Feedforward Parameter Sharing Version:
- ScannedRNN for temporal processing with GRU cells
- Hidden state initialization and management
- Sequential input/output handling with time dimensions
- Episode reset logic for hidden states

Based on the JaxMARL Implementation of PPO
"""

import functools
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import optax
import distrax
import assistax
from assistax.wrappers.baselines import  get_space_dim, LogEnvState, LogWrapper
from assistax.wrappers.aht import ZooManager, LoadAgentWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Optional


# ================================ RECURRENT NETWORK COMPONENTS ================================

class ScannedRNN(nn.Module):
    """
    Efficient recurrent neural network implementation using JAX scan.
    
    This class implements a GRU-based RNN that processes sequences efficiently
    using jax.lax.scan. It handles episode resets by reinitializing hidden states
    when done flags are encountered. With parameter sharing, all agents use the
    same RNN weights but maintain separate hidden states.
    
    The scan operation processes the entire sequence in parallel while maintaining
    the recurrent dependencies, making it much more efficient than a standard
    loop-based RNN implementation.
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
            carry: Current hidden state of the RNN
            x: Tuple of (inputs, reset_flags) for this timestep
            
        Returns:
            Tuple of (new_hidden_state, output)
        """
        rnn_state = carry
        ins, resets = x
        
        # Reset hidden state if episode ended (handles partial observability across episodes)
        rnn_state = jnp.where(
            jnp.expand_dims(resets, -1),  # Broadcast reset flag to hidden dimensions
            self.initialize_carry(rnn_state.shape),
            rnn_state
        )
        
        # Process through GRU cell
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_shape):
        """
        Initialize the RNN hidden state.
        
        Args:
            hidden_shape: Shape of the hidden state tensor
            
        Returns:
            Initialized hidden state (typically zeros)
        """
        hidden_size = hidden_shape[-1]
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), hidden_shape)


# ================================ NETWORK ARCHITECTURE ================================

class ActorCriticRNN(nn.Module):
    """
    Shared recurrent actor-critic network for all agents.
    
    All agents use the same RNN network parameters, processing observations
    through the same network weights while maintaining separate hidden states.
    This combines memory capabilities with parameter efficiency.
    
    Architecture:
    1. Observation embedding layer
    2. Shared GRU recurrent layer for temporal processing
    3. Separate actor and critic heads using shared weights
    
    Args:
        config: Configuration dictionary containing network hyperparameters
    """
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        """
        Forward pass through shared recurrent actor-critic network.
        
        Args:
            hstate: Current hidden state of the RNN (concatenated across all agents)
            x: Tuple of (observations, done_flags, available_actions)
               where observations are concatenated across all agents
               
        Returns:
            Tuple of (new_hidden_state, policy_distribution, state_values)
            - new_hidden_state: Updated RNN hidden state for all agents
            - policy_distribution: Diagonal Gaussian distribution for continuous actions
            - state_values: Critic value estimates for all agents
        """
        # Select activation function
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        # ===== OBSERVATION EMBEDDING =====
        # Project observations to embedding space before RNN processing
        embedding = nn.Dense(
            self.config["network"]["embedding_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        # ===== RECURRENT PROCESSING =====
        # Process embedded observations through shared RNN for temporal dependencies
        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)

        # ===== ACTOR NETWORK =====
        actor_mean = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
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
        
        # Create policy distribution directly (unlike non-parameter sharing version)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

        # ===== CRITIC NETWORK =====
        critic = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0)
        )(critic)

        return hstate, pi, jnp.squeeze(critic, axis=-1)


# ================================ DATA STRUCTURES ================================

class Transition(NamedTuple):
    """Single transition containing all information needed for PPO updates."""
    done: jnp.ndarray              # Episode termination flags
    action: jnp.ndarray            # Actions taken
    value: jnp.ndarray             # Critic value estimates
    reward: jnp.ndarray            # Environment rewards
    log_prob: jnp.ndarray          # Log probabilities of actions
    obs: jnp.ndarray               # Observations
    info: jnp.ndarray              # Environment info (metrics, etc.)
    avail_actions: jnp.ndarray     # Available actions mask


class RunnerState(NamedTuple):
    """State maintained throughout training/evaluation runs, including RNN hidden states."""
    train_state: TrainState                    # Flax training state
    env_state: LogEnvState                     # Environment state
    last_obs: Dict[str, jnp.ndarray]          # Most recent observations
    last_done: jnp.ndarray                     # Most recent done flags
    hstate: jnp.ndarray                        # RNN hidden states (concatenated across agents)
    update_step: int                           # Current update iteration
    rng: jnp.ndarray                          # Random number generator state


class UpdateState(NamedTuple):
    """State used during network parameter updates, including initial hidden states."""
    train_state: TrainState       # Flax training state
    init_hstate: jnp.ndarray      # Initial RNN hidden states for trajectory
    traj_batch: Transition        # Batch of trajectory data
    advantages: jnp.ndarray       # GAE advantages
    targets: jnp.ndarray          # Value function targets
    rng: jnp.ndarray             # Random number generator state


class UpdateBatch(NamedTuple):
    """Batch data structure for minibatch updates, including hidden states."""
    init_hstate: jnp.ndarray      # Initial RNN hidden states
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
    Create a training function for IPPO with RNN networks and parameter sharing.
    
    Args:
        config: Configuration dictionary with all hyperparameters
        save_train_state: Whether to save training state in metrics
        
    Returns:
        Compiled training function
    """
    # Environment setup
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    # Configuration calculations
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    env = LogWrapper(env, replace_info=True)

    print(f"Num updates: {config['NUM_UPDATES']}")


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
        Main training function for RNN-based IPPO with parameter sharing.
        
        Args:
            rng: Random number generator key
            lr: Learning rate
            ent_coef: Entropy coefficient
            clip_eps: PPO clipping parameter
            
        Returns:
            Dictionary containing final runner state and training metrics
        """
        # ===== NETWORK INITIALIZATION =====
        network = ActorCriticRNN(config=config)
        rng, network_rng = jax.random.split(rng)
        
        # Initialize network with dummy sequential input (concatenated agents)
        init_x = (
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], config["OBS_DIM"])),      # observations
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"])),                         # done flags
            jnp.zeros((1, env.num_agents * config["NUM_ENVS"], config["ACT_DIM"])),      # available actions
        )
        
        # Initialize RNN hidden states (concatenated across all agents and environments)
        init_hstate = jnp.zeros(
            (env.num_agents * config["NUM_ENVS"], config["network"]["gru_hidden_dim"])
        )
        
        network_params = network.init(network_rng, init_hstate, init_x)
        
        # Optimizer setup
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
            )
        else:
            tx = optax.chain(
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
        
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # ===== ENVIRONMENT INITIALIZATION =====
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        
        # For parameter sharing, done flags are concatenated (num_agents * num_envs,)
        init_dones = jnp.zeros((env.num_agents * config["NUM_ENVS"],), dtype=bool)

        # ===== MAIN TRAINING LOOP =====
        def _update_step(runner_state, unused):
            """Single update step: collect trajectories and update network."""
            
            # === TRAJECTORY COLLECTION ===
            def _env_step(runner_state, unused):
                """Single environment step during trajectory collection."""
                rng = runner_state.rng
                
                # Prepare network inputs with time dimension for RNN (concatenated across agents)
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                
                # Add time dimension for RNN processing (single timestep)
                ac_in = (
                    jnp.expand_dims(obs_batch, 0),           # (time=1, agent*env, obs_dim)
                    jnp.expand_dims(runner_state.last_done, 0),  # (time=1, agent*env)
                    jnp.expand_dims(avail_actions, 0),       # (time=1, agent*env, act_dim)
                )
                
                # Get policy and value predictions from shared RNN network
                hstate, pi, value = runner_state.train_state.apply_fn(
                    runner_state.train_state.params,
                    runner_state.hstate, ac_in,
                )
                
                # Sample actions from policy
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                
                # Remove time dimension for environment interaction
                action = action.squeeze(0)      # (agent*env, act_dim)
                log_prob = log_prob.squeeze(0)  # (agent*env,)
                value = value.squeeze(0)        # (agent*env,)
                
                env_act = unbatchify(action, env.agents)

                # Execute environment step
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )

                # Process outputs (concatenate instead of stack due to parameter sharing)
                done_batch = batchify(done, env.agents)
                info = jax.tree_util.tree_map(jnp.concatenate, info)
                
                transition = Transition(
                    done=done_batch,
                    action=action,
                    value=value,
                    reward=batchify(reward, env.agents),
                    log_prob=log_prob,
                    obs=obs_batch,
                    info=info,
                    avail_actions=avail_actions,
                )
                
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    hstate=hstate,  # Update with new RNN hidden state
                    update_step=runner_state.update_step,
                    rng=rng,
                )
                
                return runner_state, transition

            # Store initial hidden state for this trajectory
            init_hstate = runner_state.hstate
            
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # === ADVANTAGE CALCULATION ===
            # Get final value for bootstrapping
            last_obs_batch = batchify(runner_state.last_obs, env.agents)
            ac_in = (
                jnp.expand_dims(last_obs_batch, 0),  # Add time dimension
                jnp.expand_dims(runner_state.last_done, 0),
                jnp.ones((1, env.num_agents * config["NUM_ENVS"], config["ACT_DIM"]), dtype=jnp.uint8),
            )
            _, _, last_val = runner_state.train_state.apply_fn(
                runner_state.train_state.params,
                runner_state.hstate, ac_in,
            )
            last_val = last_val.squeeze(0)  # Remove time dimension

            def _calculate_gae(traj_batch, last_val):
                """Calculate Generalized Advantage Estimation."""
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
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
                    """Single minibatch parameter update."""
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        """PPO loss function for shared RNN network."""
                        # Re-evaluate policy and values through shared RNN
                        ac_in = (
                            traj_batch.obs,      # (step, agent*env, obs_dim)
                            traj_batch.done,     # (step, agent*env)
                            traj_batch.avail_actions,  # (step, agent*env, act_dim)
                        )
                        
                        _, pi, value = train_state.apply_fn(
                            params,
                            init_hstate.squeeze(0),  # Remove step dimension from initial state
                            ac_in
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean()

                        # Actor loss (PPO clipped objective)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        
                        # Normalize advantages across all agents and timesteps
                        gae = (
                            (gae - gae.mean())
                            / (gae.std() + 1e-8)
                        )
                        
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                clip_eps_min,
                                clip_eps_max,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        
                        entropy = pi.entropy().mean()
                        
                        # Debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac_min = jnp.mean(ratio < clip_eps_min)
                        clip_frac_max = jnp.mean(ratio > clip_eps_max)
                        
                        total_loss = (
                            loss_actor.sum()
                            + config["VF_COEF"] * value_loss.sum()
                            - ent_coef * entropy.sum()
                        )
                        
                        return total_loss, (
                            value_loss,
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_frac_min,
                            clip_frac_max,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params,
                        batch_info.init_hstate,
                        batch_info.traj_batch,
                        batch_info.advantages,
                        batch_info.targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    loss_info = {
                        "total_loss": total_loss[0],
                        "actor_loss": total_loss[1][1],
                        "critic_loss": total_loss[1][0],
                        "entropy": total_loss[1][2],
                        "approx_kl": total_loss[1][3],
                        "clip_frac_min": total_loss[1][4],
                        "clip_frac_max": total_loss[1][5],
                    }
                    return train_state, loss_info

                rng = update_state.rng

                # Prepare minibatches - different structure due to concatenated agents
                batch_size = config["NUM_ENVS"] * env.num_agents
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                assert (
                    batch_size % minibatch_size == 0
                ), "Batch size must be divisible by number of minibatches"
                
                batch = UpdateBatch(
                    init_hstate=jnp.expand_dims(update_state.init_hstate, 0),  # Add step dimension
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                )
                
                # Shuffle and create minibatches
                # Note: Simpler axis manipulation compared to non-parameter sharing due to concatenated structure
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                
                # Reshape for minibatch processing
                # Original: (step, agent*env, ...) -> (agent*env, step, ...)
                batch = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(0, 1),  # Swap step and agent*env dimensions
                    batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),  # Shuffle over agent*env dimension
                    batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    ),
                    shuffled_batch
                )
                # Final shape: (n_mini, minibatch_size, step, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(1, 2),  # Move step dimension back
                    minibatches
                )
                # Final shape: (n_mini, step, minibatch_size, ...)
                
                train_state, loss_info = jax.lax.scan(
                    _update_minbatch, update_state.train_state, minibatches
                )
                
                update_state = UpdateState(
                    train_state=train_state,
                    init_hstate=update_state.init_hstate,
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                    rng=rng,
                )
                return update_state, loss_info

            runner_rng, update_rng = jax.random.split(runner_state.rng)
            update_state = UpdateState(
                train_state=runner_state.train_state,
                init_hstate=init_hstate,  # Store initial hidden state for trajectory
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
                hstate=runner_state.hstate,  # Maintain RNN hidden state
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
            hstate=init_hstate,  # Initialize with zero hidden states
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
    Create an evaluation function for trained RNN-based IPPO agents with parameter sharing.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (environment, evaluation_function)
    """
    # Environment setup
    env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
    config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
    env = LogWrapper(env, replace_info=True)
    max_steps = env.episode_length

    def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig()):
        """
        Run evaluation episodes with trained shared RNN agent.
        
        Args:
            rng: Random number generator key
            train_state: Trained model state
            log_eval_info: Configuration for what to log
            
        Returns:
            Evaluation information from all episodes
        """
        rng_reset, rng_env = jax.random.split(rng)
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        obsv, env_state = jax.vmap(env.reset)(rngs_reset)
        
        # Concatenated done flags and hidden states for parameter sharing
        init_dones = jnp.zeros((env.num_agents * config["NUM_EVAL_EPISODES"],), dtype=bool)
        init_hstate = jnp.zeros(
            (env.num_agents * config["NUM_EVAL_EPISODES"], config["network"]["gru_hidden_dim"])
        )
        
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            hstate=init_hstate,
            update_step=0,
            rng=rng_env,
        )

        def _env_step(runner_state, unused):
            """Single environment step during evaluation."""
            rng = runner_state.rng
            
            # Prepare inputs and get actions from shared RNN network
            obs_batch = batchify(runner_state.last_obs, env.agents)
            avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
            avail_actions = jax.lax.stop_gradient(
                batchify(avail_actions, env.agents)
            )
            
            # Add time dimension for RNN processing
            ac_in = (
                jnp.expand_dims(obs_batch, 0),
                jnp.expand_dims(runner_state.last_done, 0),
                jnp.expand_dims(avail_actions, 0),
            )
            
            hstate, pi, value = runner_state.train_state.apply_fn(
                runner_state.train_state.params,
                runner_state.hstate, ac_in,
            )
            
            rng, act_rng = jax.random.split(rng)
            action, log_prob = pi.sample_and_log_prob(seed=act_rng)
            
            # Remove time dimension
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
            value = value.squeeze(0)
            env_act = unbatchify(action, env.agents)

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
                hstate=hstate,  # Update RNN hidden state
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

@hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_mabrax")
def main(config):
    """
    Main function for hyperparameter sweeping and training execution with RNN networks and parameter sharing.
    
    Supports swept hyperparameters including learning rate, entropy coefficient,
    and clipping epsilon. Results are saved as numpy arrays for analysis.
    """
    config_key = hash(config) % 2**62
    sweep_config = config.SWEEP
    config = OmegaConf.to_container(config)
    
    # Initialize random number generators
    rng = jax.random.PRNGKey(config["SEED"])
    hparam_rng, run_rng = jax.random.split(rng, 2)
    
    # Generate hyperparameter configurations
    NUM_HPARAM_CONFIGS = sweep_config.num_configs
    lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(hparam_rng, 3)

    # Learning rate sweep
    if sweep_config.get("lr", False):
        lrs = 10**jax.random.uniform(
            lr_rng,
            shape=(NUM_HPARAM_CONFIGS,),
            minval=sweep_config.lr.min,
            maxval=sweep_config.lr.max,
        )
        lr_axis = 0
    else:
        lrs = config["LR"]
        lr_axis = None

    # Entropy coefficient sweep
    if sweep_config.get("ent_coef", False):
        ent_coefs = 10**jax.random.uniform(
            ent_coef_rng,
            shape=(NUM_HPARAM_CONFIGS,),
            minval=sweep_config.ent_coef.min,
            maxval=sweep_config.ent_coef.max,
        )
        ent_coef_axis = 0
    else:
        ent_coefs = config["ENT_COEF"]
        ent_coef_axis = None

    # Clipping epsilon sweep
    if sweep_config.get("clip_eps", False):
        clip_epss = 10**jax.random.uniform(
            clip_eps_rng,
            shape=(NUM_HPARAM_CONFIGS,),
            minval=sweep_config.clip_eps.min,
            maxval=sweep_config.clip_eps.max,
        )
        clip_eps_axis = 0
    else:
        clip_epss = config["CLIP_EPS"]
        clip_eps_axis = None

    # Prepare training runs
    run_rngs = jax.random.split(run_rng, config["NUM_SEEDS"])
    
    # Execute training with vmapped hyperparameter sweep
    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config),
            device=jax.devices()[config["DEVICE"]]
        )
        out = jax.vmap(
            jax.vmap(
                train_jit,
                in_axes=(0, None, None, None),  # Vmap over seeds
            ),
            in_axes=(None, lr_axis, ent_coef_axis, clip_eps_axis)  # Vmap over hyperparameters
        )(run_rngs, lrs, ent_coefs, clip_epss)
    
    # Save results
    jnp.save(f"metrics_{config_key}.npy", out["metrics"], allow_pickle=True)
    jnp.save(f"hparams_{config_key}.npy", {
        "lr": lrs,
        "ent_coef": ent_coefs,
        "clip_eps": clip_epss,
        "ratio_clip_eps": config["RATIO_CLIP_EPS"],
        "num_steps": config["NUM_STEPS"],
        "num_envs": config["NUM_ENVS"],
        "update_epochs": config["UPDATE_EPOCHS"],
        "num_minibatches": config["NUM_MINIBATCHES"],
    })


if __name__ == "__main__":
    main()

# """ 
# Based on the PureJaxRL Implementation of PPO
# """

# import functools
# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import jax.numpy as jnp
# from flax import struct
# from flax.linen.initializers import constant, orthogonal
# from flax.training.train_state import TrainState
# import optax
# import distrax
# import assistax
# from assistax.wrappers.baselines import get_space_dim, LogEnvState
# from assistax.wrappers.baselines import LogWrapper
# import hydra
# from omegaconf import OmegaConf
# from typing import Sequence, NamedTuple, Any, Dict, Optional


# class ScannedRNN(nn.Module):
#     @functools.partial(
#         nn.scan,
#         variable_broadcast="params",
#         in_axes=0,
#         out_axes=0,
#         split_rngs={"params": False},
#     )
#     @nn.compact
#     def __call__(self, carry, x):
#         rnn_state = carry
#         ins, resets = x
#         rnn_state = jnp.where(
#             # assume resets comes in with shape (n_step,)
#             jnp.expand_dims(resets,-1),
#             self.initialize_carry(rnn_state.shape),
#             rnn_state
#         )
#         new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
#         return new_rnn_state, y

#     @staticmethod
#     def initialize_carry(hidden_shape):
#         hidden_size = hidden_shape[-1]
#         cell = nn.GRUCell(features=hidden_size)
#         return cell.initialize_carry(jax.random.PRNGKey(0), hidden_shape)

# class ActorCriticRNN(nn.Module):
#     config: Dict

#     @nn.compact
#     def __call__(self, hstate, x):
#         if self.config["network"]["activation"] == "relu":
#             activation = nn.relu
#         else:
#             activation = nn.tanh

#         obs, done, avail_actions = x

#         embedding = nn.Dense(
#             self.config["network"]["embedding_dim"],
#             kernel_init=orthogonal(jnp.sqrt(2)),
#             bias_init=constant(0.0),
#         )(obs)
#         embedding = activation(embedding)

#         rnn_in = (embedding, done)
#         hstate, embedding = ScannedRNN()(hstate, rnn_in)

#         actor_mean = nn.Dense(
#             self.config["network"]["gru_hidden_dim"],
#             kernel_init=orthogonal(jnp.sqrt(2)),
#             bias_init=constant(0.0)
#         )(embedding)
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

#         critic = nn.Dense(
#             self.config["network"]["gru_hidden_dim"],
#             kernel_init=orthogonal(jnp.sqrt(2)),
#             bias_init=constant(0.0),
#         )(embedding)
#         critic = activation(critic)
#         critic = nn.Dense(
#             1,
#             kernel_init=orthogonal(1.0),
#             bias_init=constant(0.0)
#         )(critic)

#         return hstate, pi, jnp.squeeze(critic, axis=-1)

# class Transition(NamedTuple):
#     done: jnp.ndarray
#     action: jnp.ndarray
#     value: jnp.ndarray
#     reward: jnp.ndarray
#     log_prob: jnp.ndarray
#     obs: jnp.ndarray
#     info: jnp.ndarray
#     avail_actions: jnp.ndarray

# class RunnerState(NamedTuple):
#     train_state: TrainState
#     env_state: LogEnvState
#     last_obs: Dict[str, jnp.ndarray]
#     last_done: jnp.ndarray
#     hstate: jnp.ndarray
#     update_step: int
#     rng: jnp.ndarray

# class UpdateState(NamedTuple):
#     train_state: TrainState
#     init_hstate: jnp.ndarray
#     traj_batch: Transition
#     advantages: jnp.ndarray
#     targets: jnp.ndarray
#     rng: jnp.ndarray

# class UpdateBatch(NamedTuple):
#     init_hstate: jnp.ndarray
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
#     env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#     config["NUM_UPDATES"] = (
#         config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
#     )
#     config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
#     config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
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
#         network = ActorCriticRNN(config=config)
#         rng, network_rng = jax.random.split(rng)
#         init_x = (
#             jnp.zeros( # obs
#                 (1, env.num_agents*config["NUM_ENVS"], config["OBS_DIM"])
#             ),
#             jnp.zeros( # done
#                 (1, env.num_agents*config["NUM_ENVS"])
#             ),
#             jnp.zeros( # avail_actions
#                 (1, env.num_agents*config["NUM_ENVS"], config["ACT_DIM"])
#             ),
#         )
#         init_hstate = jnp.zeros(
#             (env.num_agents*config["NUM_ENVS"], config["network"]["gru_hidden_dim"])
#         )
#         network_params = network.init(network_rng, init_hstate, init_x)
#         if config["ANNEAL_LR"]:
#             tx = optax.chain(
#                 optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
#                 optax.adam(learning_rate=linear_schedule(lr), eps=config["ADAM_EPS"]),
#             )
#         else:
#             tx = optax.chain(
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
#         train_state = TrainState.create(
#             apply_fn=network.apply,
#             params=network_params,
#             tx=tx,
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
#                 ac_in = (
#                     # add time dimension to pass to RNN
#                     jnp.expand_dims(obs_batch, 0),
#                     jnp.expand_dims(runner_state.last_done, 0),
#                     jnp.expand_dims(avail_actions, 0),
#                 )
#                 # SELECT ACTION
#                 hstate, pi, value = runner_state.train_state.apply_fn(
#                     runner_state.train_state.params,
#                     runner_state.hstate, ac_in,
#                 )
#                 rng, act_rng = jax.random.split(rng)
#                 action, log_prob = pi.sample_and_log_prob(seed=act_rng)
#                 # remove time dimension
#                 action = action.squeeze(0)
#                 log_prob = log_prob.squeeze(0)
#                 value = value.squeeze(0)
#                 env_act = unbatchify(action, env.agents)

#                 # STEP ENV
#                 rng, _rng = jax.random.split(rng)
#                 rng_step = jax.random.split(_rng, config["NUM_ENVS"])
#                 obsv, env_state, reward, done, info = jax.vmap(env.step)(
#                     rng_step, runner_state.env_state, env_act,
#                 )
#                 done_batch = batchify(done, env.agents)
#                 info = jax.tree_util.tree_map(jnp.concatenate, info)
#                 transition = Transition(
#                     done=done_batch,
#                     action=action,
#                     value=value,
#                     reward=batchify(reward, env.agents),
#                     log_prob=log_prob,
#                     obs=obs_batch,
#                     info=info,
#                     avail_actions=avail_actions,
#                 )
#                 runner_state = RunnerState(
#                     train_state=runner_state.train_state,
#                     env_state=env_state,
#                     last_obs=obsv,
#                     last_done=done_batch,
#                     hstate=hstate,
#                     update_step=runner_state.update_step,
#                     rng=rng,
#                 )
#                 return runner_state, transition

#             init_hstate = runner_state.hstate
#             runner_state, traj_batch = jax.lax.scan(
#                 _env_step, runner_state, None, config["NUM_STEPS"]
#             )

#             # CALCULATE ADVANTAGE
#             last_obs_batch = batchify(runner_state.last_obs, env.agents)
#             ac_in = (
#                 # add time dimension to pass to RNN
#                 jnp.expand_dims(last_obs_batch, 0),
#                 jnp.expand_dims(runner_state.last_done, 0),
#                 jnp.ones((1, env.num_agents*config["NUM_ENVS"], config["ACT_DIM"]), dtype=jnp.uint8),
#             )
#             _, _, last_val = runner_state.train_state.apply_fn(
#                 runner_state.train_state.params,
#                 runner_state.hstate, ac_in,
#             )
#             last_val = last_val.squeeze(0)

#             def _calculate_gae(traj_batch, last_val):
#                 def _get_advantages(gae_and_next_value, transition):
#                     gae, next_value = gae_and_next_value
#                     done, value, reward = (
#                         transition.done,
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
#                     init_hstate, traj_batch, advantages, targets = batch_info

#                     def _loss_fn(params, init_hstate, traj_batch, gae, targets):
#                         # RERUN NETWORK
#                         ac_in = (
#                             traj_batch.obs,
#                             traj_batch.done,
#                             traj_batch.avail_actions,
#                         )
#                         _, pi, value = train_state.apply_fn(
#                             params,
#                             init_hstate.squeeze(0), # remove step dim
#                             ac_in
#                         )
#                         log_prob = pi.log_prob(traj_batch.action)

#                         # CALCULATE VALUE LOSS
#                         value_losses = jnp.square(value - targets)
#                         value_loss = 0.5 * value_losses.mean()

#                         # CALCULATE ACTOR LOSS
#                         logratio = log_prob - traj_batch.log_prob
#                         ratio = jnp.exp(logratio)
#                         gae = (
#                             (gae - gae.mean())
#                             / (gae.std() + 1e-8)
#                         )
#                         loss_actor1 = ratio * gae
#                         loss_actor2 = (
#                             jnp.clip(
#                                 ratio,
#                                 clip_eps_min,
#                                 clip_eps_max,
#                             )
#                             * gae
#                         )
#                         loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
#                         loss_actor = loss_actor.mean()
#                         entropy = pi.entropy().mean()
#                         # debug metrics
#                         approx_kl = ((ratio - 1) - logratio).mean()
#                         clip_frac_min = jnp.mean(ratio < clip_eps_min)
#                         clip_frac_max = jnp.mean(ratio > clip_eps_max)
#                         # ---
#                         total_loss = (
#                             loss_actor.sum()
#                             + config["VF_COEF"] * value_loss.sum()
#                             - ent_coef * entropy.sum()
#                         )
#                         return total_loss, (
#                             value_loss,
#                             loss_actor,
#                             entropy,
#                             approx_kl,
#                             clip_frac_min,
#                             clip_frac_max,
#                         )

#                     grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
#                     total_loss, grads = grad_fn(
#                         train_state.params,
#                         batch_info.init_hstate,
#                         batch_info.traj_batch,
#                         batch_info.advantages,
#                         batch_info.targets
#                     )
#                     train_state = train_state.apply_gradients(grads=grads)
#                     loss_info = {
#                         "total_loss": total_loss[0],
#                         "actor_loss": total_loss[1][1],
#                         "critic_loss": total_loss[1][0],
#                         "entropy": total_loss[1][2],
#                         "approx_kl": total_loss[1][3],
#                         "clip_frac_min": total_loss[1][4],
#                         "clip_frac_max": total_loss[1][5],
#                     }
#                     return train_state, loss_info

#                 rng = update_state.rng

#                 batch_size = config["NUM_ENVS"] * env.num_agents
#                 minibatch_size = batch_size // config["NUM_MINIBATCHES"]
#                 assert (
#                     batch_size % minibatch_size == 0
#                 ), "unable to equally partition into minibatches"
#                 batch = UpdateBatch(
#                     init_hstate=jnp.expand_dims(update_state.init_hstate, 0), # add step dimension
#                     traj_batch=update_state.traj_batch,
#                     advantages=update_state.advantages,
#                     targets=update_state.targets,
#                 )
#                 rng, _rng = jax.random.split(rng)
#                 permutation = jax.random.permutation(_rng, batch_size)
#                 # initial axes: (step, agent*env, ...)
#                 batch = jax.tree_util.tree_map(
#                     lambda x: x.swapaxes(0,1),
#                     batch
#                 ) # swap axes to (agent*env, step, ...)
#                 shuffled_batch = jax.tree_util.tree_map(
#                     lambda x: jnp.take(x, permutation, axis=0),
#                     batch
#                 ) # shuffle: maintains axes (agent*env, step ...)
#                 minibatches = jax.tree_util.tree_map(
#                     lambda x: jnp.reshape(
#                         x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
#                     ),
#                     shuffled_batch
#                 ) # split into minibatches. axes (n_mini, minibatch_size, step, ...)
#                 minibatches = jax.tree_util.tree_map(
#                     lambda x: x.swapaxes(1,2),
#                     minibatches
#                 ) # swap axes to (n_mini, step, minibatch_size, ...)
#                 train_state, loss_info = jax.lax.scan(
#                     _update_minbatch, update_state.train_state, minibatches
#                 )
#                 update_state = UpdateState(
#                     train_state=train_state,
#                     init_hstate=update_state.init_hstate,
#                     traj_batch=update_state.traj_batch,
#                     advantages=update_state.advantages,
#                     targets=update_state.targets,
#                     rng=rng,
#                 )
#                 return update_state, loss_info

#             runner_rng, update_rng = jax.random.split(runner_state.rng)
#             update_state = UpdateState(
#                 train_state=runner_state.train_state,
#                 init_hstate=init_hstate,
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
#                 hstate=runner_state.hstate,
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
#             hstate=init_hstate,
#             update_step=0,
#             rng=_rng,
#         )
#         runner_state, metric = jax.lax.scan(
#             _update_step, runner_state, None, config["NUM_UPDATES"]
#         )
#         return {"runner_state": runner_state, "metrics": metric}

#     return train

# def make_evaluation(config):
#     env = assistax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
#     config["OBS_DIM"] = get_space_dim(env.observation_space(env.agents[0]))
#     config["ACT_DIM"] = get_space_dim(env.action_space(env.agents[0]))
#     env = LogWrapper(env, replace_info=True)
#     max_steps = env.episode_length

#     def run_evaluation(rng, train_state, log_eval_info=EvalInfoLogConfig()):
#         rng_reset, rng_env = jax.random.split(rng)
#         rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
#         obsv, env_state = jax.vmap(env.reset)(rngs_reset)
#         init_dones = jnp.zeros((env.num_agents*config["NUM_EVAL_EPISODES"],), dtype=bool)
#         init_hstate = jnp.zeros(
#             (env.num_agents*config["NUM_EVAL_EPISODES"], config["network"]["gru_hidden_dim"])
#         )
#         runner_state = RunnerState(
#             train_state=train_state,
#             env_state=env_state,
#             last_obs=obsv,
#             last_done=init_dones,
#             hstate=init_hstate,
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
#             ac_in = (
#                 # add time dimension to pass to RNN
#                 jnp.expand_dims(obs_batch, 0),
#                 jnp.expand_dims(runner_state.last_done, 0),
#                 jnp.expand_dims(avail_actions, 0),
#             )
#             # SELECT ACTION
#             hstate, pi, value = runner_state.train_state.apply_fn(
#                 runner_state.train_state.params,
#                 runner_state.hstate, ac_in,
#             )
#             rng, act_rng = jax.random.split(rng)
#             action, log_prob = pi.sample_and_log_prob(seed=act_rng)
#             # remove time dimension
#             action = action.squeeze(0)
#             log_prob = log_prob.squeeze(0)
#             value = value.squeeze(0)
#             env_act = unbatchify(action, env.agents)

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
#                 hstate=hstate,
#                 update_step=runner_state.update_step,
#                 rng=rng,
#             )
#             return runner_state, eval_info

#         _, eval_info = jax.lax.scan(
#             _env_step, runner_state, None, max_steps
#         )

#         return eval_info
#     return env, run_evaluation

# @hydra.main(version_base=None, config_path="config", config_name="ippo_rnn_mabrax")
# def main(config):
#     config_key = hash(config) % 2**62
#     sweep_config = config.SWEEP
#     config = OmegaConf.to_container(config)
#     rng = jax.random.PRNGKey(config["SEED"])
#     hparam_rng, run_rng = jax.random.split(rng, 2)
#     # generate hyperparams
#     NUM_HPARAM_CONFIGS = sweep_config.num_configs
#     lr_rng, ent_coef_rng, clip_eps_rng = jax.random.split(hparam_rng, 3)

#     if sweep_config.get("lr", False):
#         lrs = 10**jax.random.uniform(
#             lr_rng,
#             shape=(NUM_HPARAM_CONFIGS,),
#             minval=sweep_config.lr.min,
#             maxval=sweep_config.lr.max,
#         )
#         lr_axis = 0
#     else:
#         lrs = config["LR"]
#         lr_axis = None

#     if sweep_config.get("ent_coef", False):
#         ent_coefs = 10**jax.random.uniform(
#             ent_coef_rng,
#             shape=(NUM_HPARAM_CONFIGS,),
#             minval=sweep_config.ent_coef.min,
#             maxval=sweep_config.ent_coef.max,
#         )
#         ent_coef_axis = 0
#     else:
#         ent_coefs = config["ENT_COEF"]
#         ent_coef_axis = None

#     if sweep_config.get("clip_eps", False):
#         clip_epss = 10**jax.random.uniform(
#             clip_eps_rng,
#             shape=(NUM_HPARAM_CONFIGS,),
#             minval=sweep_config.clip_eps.min,
#             maxval=sweep_config.clip_eps.max,
#         )
#         clip_eps_axis = 0
#     else:
#         clip_epss = config["CLIP_EPS"]
#         clip_eps_axis = None

#     run_rngs = jax.random.split(run_rng, config["NUM_SEEDS"])
#     with jax.disable_jit(config["DISABLE_JIT"]):
#         train_jit = jax.jit(
#             make_train(config),
#             device=jax.devices()[config["DEVICE"]]
#         )
#         out = jax.vmap(
#             jax.vmap(
#                 train_jit,
#                 in_axes=(0, None, None, None),
#             ),
#             in_axes=(None, lr_axis, ent_coef_axis, clip_eps_axis)
#         )(run_rngs, lrs, ent_coefs, clip_epss)
#     jnp.save(f"metrics_{config_key}.npy", out["metrics"], allow_pickle=True)
#     jnp.save(f"hparams_{config_key}.npy", {
#         "lr": lrs,
#         "ent_coef": ent_coefs,
#         "clip_eps": clip_epss,
#         "ratio_clip_eps": config["RATIO_CLIP_EPS"],
#         "num_steps": config["NUM_STEPS"],
#         "num_envs": config["NUM_ENVS"],
#         "update_epochs": config["UPDATE_EPOCHS"],
#         "num_minibatches": config["NUM_MINIBATCHES"],
#         }
#     )


# if __name__ == "__main__":
#     main()
