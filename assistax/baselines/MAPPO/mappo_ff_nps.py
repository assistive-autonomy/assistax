""" 
Based on the PureJaxRL Implementation of PPO
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
from jaxmarl.wrappers.baselines import LogWrapper, LogCrossplayWrapper
from jaxmarl.wrappers.aht_all import ZooManager, LoadAgentWrapper, LoadEvalAgentWrapper
import hydra
from omegaconf import OmegaConf
from typing import Sequence, NamedTuple, Any, Dict, Optional
from datetime import datetime

import functools

def _tree_shape(pytree):
    return jax.tree.map(lambda x: x.shape, pytree)

@functools.partial(
    nn.vmap,
    in_axes=0, out_axes=0,
    variable_axes={"params": 0},
    split_rngs={"params": True},
    axis_name="agents",
)
class MultiActor(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

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
        actor_mean = nn.Dense(
            self.config["ACT_DIM"],
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0)
        )(actor_mean)
        actor_log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.config["ACT_DIM"],)
        )
        pi = (actor_mean, jnp.exp(actor_log_std))

        return pi

class Critic(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs = x

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

class Transition(NamedTuple):
    done: jnp.ndarray
    all_done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    global_obs: jnp.ndarray
    info: jnp.ndarray
    avail_actions: jnp.ndarray

class ActorCriticTrainState(NamedTuple):
    actor: TrainState
    critic: TrainState

class RunnerState(NamedTuple):
    train_state: ActorCriticTrainState
    env_state: LogEnvState
    last_obs: Dict[str, jnp.ndarray]
    last_done: jnp.ndarray
    update_step: int
    rng: jnp.ndarray
    ag_idx: Optional[int] = None # hopefully this doesn't break something in other code

class UpdateState(NamedTuple):
    train_state: ActorCriticTrainState
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray
    rng: jnp.ndarray

class UpdateBatch(NamedTuple):
    traj_batch: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray

class EvalInfo(NamedTuple):
    env_state: Optional[LogEnvState]
    done: Optional[jnp.ndarray]
    action: Optional[jnp.ndarray]
    value: Optional[jnp.ndarray]
    reward: Optional[jnp.ndarray]
    log_prob: Optional[jnp.ndarray]
    obs: Optional[jnp.ndarray]
    info: Optional[jnp.ndarray]
    avail_actions: Optional[jnp.ndarray]
    ag_idx: Optional[jnp.ndarray]
    idx_mapping: Optional[Dict[int, str]] = None

@struct.dataclass
class EvalInfoLogConfig:
    env_state: bool = True
    done: bool = True
    action: bool = True
    value: bool = True
    reward: bool = True
    log_prob: bool = True
    obs: bool = True
    info: bool = True
    avail_actions: bool = True

def batchify(qty: Dict[str, jnp.ndarray], agents: Sequence[str]) -> jnp.ndarray:
    """Convert dict of arrays to batched array."""
    return jnp.stack(tuple(qty[a] for a in agents))

def unbatchify(qty: jnp.ndarray, agents: Sequence[str]) -> Dict[str, jnp.ndarray]:
    """Convert batched array to dict of arrays."""
    # N.B. assumes the leading dimension is the agent dimension
    return dict(zip(agents, qty))

def make_train(config, save_train_state=False, load_zoo=False):
     
    if load_zoo:
        zoo = ZooManager(config["ZOO_PATH"])
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        env = LoadAgentWrapper.load_from_zoo(env, zoo, load_zoo)
    else:
        env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    
    config["OBS_DIM"] = int(get_space_dim(env.observation_space(env.agents[0])))
    config["ACT_DIM"] = int(get_space_dim(env.action_space(env.agents[0])))
    config["GOBS_DIM"] = int(get_space_dim(env.observation_space("global")))
    env = LogWrapper(env, replace_info=True)

    def linear_schedule(initial_lr):
        def _linear_schedule(count):
            frac = (
                1.0
                - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                / config["NUM_UPDATES"]
            )
            return initial_lr * frac
        return _linear_schedule

    def train(rng, lr, ent_coef, clip_eps):

        # INIT NETWORK
        actor_network = MultiActor(config=config)
        critic_network = Critic(config=config)
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)
        init_x_actor = (
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
        init_x_critic = jnp.zeros((1, config["GOBS_DIM"]))
        actor_network_params = actor_network.init(actor_rng, init_x_actor)
        critic_network_params = critic_network.init(critic_rng, init_x_critic)
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
        if config["SCALE_CLIP_EPS"]:
            clip_eps /= env.num_agents
        if config["RATIO_CLIP_EPS"]:
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0/(1.0 - clip_eps)
        else:
            clip_eps_min = 1.0 - clip_eps
            clip_eps_max = 1.0 + clip_eps
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

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)
        init_dones = jnp.zeros((env.num_agents, config["NUM_ENVS"]), dtype=bool)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                rng = runner_state.rng
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                actor_in = (
                    obs_batch,
                    runner_state.last_done,
                    avail_actions
                )
                critic_in = runner_state.last_obs["global"]
                # SELECT ACTION
                actor_mean, actor_std = runner_state.train_state.actor.apply_fn(
                    runner_state.train_state.actor.params,
                    actor_in,
                )
                actor_std = jnp.expand_dims(actor_std, axis=1)
                pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                env_act = unbatchify(action, env.agents)

                # COMPUTE VALUE
                value = runner_state.train_state.critic.apply_fn(
                    runner_state.train_state.critic.params,
                    critic_in,
                )
                value = jnp.broadcast_to(value, (env.num_agents, *value.shape))

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, runner_state.env_state, env_act,
                )
                done_batch = batchify(done, env.agents)
                all_done = done["__all__"]
                all_done = jnp.broadcast_to(all_done, (env.num_agents, *all_done.shape))
                info = jax.tree_util.tree_map(lambda x: x.swapaxes(0,1), info)
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

            # CALCULATE ADVANTAGE
            critic_in = runner_state.last_obs["global"]
            last_val = runner_state.train_state.critic.apply_fn(
                runner_state.train_state.critic.params,
                critic_in,
            )
            last_val = jnp.broadcast_to(last_val, (env.num_agents, *last_val.shape))

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.all_done,
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        actor_in = (
                            traj_batch.obs,
                            traj_batch.done,
                            traj_batch.avail_actions,
                        )
                        actor_mean, actor_std = train_state.actor.apply_fn(
                            actor_params,
                            actor_in,
                        )
                        actor_std = jnp.expand_dims(actor_std, axis=1)
                        pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (
                            (gae - gae.mean(axis=-1, keepdims=True))
                            / (gae.std(axis=-1, keepdims=True) + 1e-8)
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
                        pg_loss = pg_loss.mean(axis=-1)
                        entropy = pi.entropy().mean(axis=-1)
                        # debug metrics
                        approx_kl = ((ratio - 1) - logratio).mean(axis=-1)
                        clip_frac_min = jnp.mean(ratio < clip_eps_min, axis=-1)
                        clip_frac_max = jnp.mean(ratio > clip_eps_max, axis=-1)
                        # ---
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
                        critic_in = traj_batch.global_obs
                        value = train_state.critic.apply_fn(
                            critic_params,
                            critic_in,
                        )
                        value_losses = jnp.square(value - targets)
                        value_loss = 0.5 * value_losses.mean()
                        critic_loss =  config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss,)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        train_state.actor.params, batch_info.traj_batch, batch_info.advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        train_state.critic.params, batch_info.traj_batch, batch_info.targets
                    )
                    train_state = ActorCriticTrainState(
                        actor = train_state.actor.apply_gradients(grads=actor_grads),
                        critic = train_state.critic.apply_gradients(grads=critic_grads),
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

                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                minibatch_size = batch_size // config["NUM_MINIBATCHES"]
                assert (
                    batch_size % minibatch_size == 0
                ), "unable to equally partition into minibatches"
                batch = UpdateBatch(
                    traj_batch=update_state.traj_batch,
                    advantages=update_state.advantages,
                    targets=update_state.targets,
                )
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(1,2),
                    batch
                ) # swap axes to (step, env, agent, ...)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size, *x.shape[2:]), order="F"),
                    batch
                ) # reshape axes to (step*env, agent, ...)
                # order="F" preserves the agent and ... dimension
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0),
                    batch
                ) # shuffle: maintains axes (step*env, agent, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config["NUM_MINIBATCHES"], -1, *x.shape[1:])
                    ),
                    shuffled_batch
                ) # split into minibatches. axes (n_mini, minibatch_size, agent, ...)
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.swapaxes(1,2),
                    minibatches
                ) # swap axes to (n_mini, agent, minibatch_size, ...)
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
            update_step = runner_state.update_step + 1
            metric = traj_batch.info
            metric = jax.tree_util.tree_map(lambda x: x.mean(axis=(0,2)), metric)
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(axis=(0,1)), loss_info)
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

        rng, _rng = jax.random.split(rng)
        runner_state = RunnerState(
            train_state=train_state,
            env_state=env_state,
            last_obs=obsv,
            last_done=init_dones,
            update_step=0,
            rng=_rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train

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
    config["GOBS_DIM"] = int(get_space_dim(env.observation_space("global")))
 
    if crossplay:
        env = LogCrossplayWrapper(env, replace_info=True, crossplay_info=crossplay)
        now = datetime.now()
        mapping_name = f"{config['ENV_NAME']}_{now.strftime('%Y-%m-%d_%H-%M-%S')}_idx_mapping.npy"
        jnp.save(mapping_name, env.idx_mapping)
    else:
        env = LogWrapper(env, replace_info=True, crossplay_info=crossplay)
    max_steps = env.episode_length

    def run_evaluation(rngs, train_state, log_eval_info=EvalInfoLogConfig()): # removed num_episodes=1 as this wasn't used
        
        if crossplay:
            rng_reset, rng_env = jax.random.split(rngs[0])
        else:
            rng_reset, rng_env = jax.random.split(rngs)# use first rng for init 
        
        rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
        init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)
        if crossplay:
            init_obsv, init_env_state = jax.vmap(env.reset, in_axes=(0, None))(rngs_reset, None) # init ag_idx with None
            init_runner_state = RunnerState(
                train_state=train_state,
                env_state=init_env_state,
                last_obs=init_obsv,
                last_done=init_dones,
                update_step=0,
                rng=rng_env,
                ag_idx=init_env_state.env_state.ag_idx # Init with None to start running epiodes
            )

        else:
            init_obsv, init_env_state = jax.vmap(env.reset)(rngs_reset)
            init_runner_state = RunnerState(
                train_state=train_state,
                env_state=init_env_state,
                last_obs=init_obsv,
                last_done=init_dones,
                update_step=0,
                rng=rng_env,
                # ag_idx=env_state.env_state.ag_idx['human'] # This is the wrong spot to be incrementing
            )
        def _run_episode(runner_state, episode_rng):

            rng_reset, rng_env = jax.random.split(episode_rng)
            rngs_reset = jax.random.split(rng_reset, config["NUM_EVAL_EPISODES"])
            init_dones = jnp.zeros((env.num_agents, config["NUM_EVAL_EPISODES"],), dtype=bool)
            if crossplay:
                obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(rngs_reset, runner_state.ag_idx) # I think this would skip the 0 index so I probably also want to init this with None

                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=init_dones,
                    update_step=runner_state.update_step,
                    rng=rng_env,
                    ag_idx=env_state.env_state.ag_idx # This is dict {'human': ag_idx} will be of dimension len(parallel envs)
                )

            else:
                obsv, env_state = jax.vmap(env.reset)(rngs_reset)
  
                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=init_dones,
                    update_step=runner_state.update_step,
                    rng=rng_env,
                    # ag_idx=env_state.env_state.ag_idx['human'] # This is the wrong spot to be incrementing
                )


            def _env_step(runner_state, unused):
                rng = runner_state.rng
                obs_batch = batchify(runner_state.last_obs, env.agents)
                avail_actions = jax.vmap(env.get_avail_actions)(runner_state.env_state.env_state)
                avail_actions = jax.lax.stop_gradient(
                    batchify(avail_actions, env.agents)
                )
                actor_in = (
                    obs_batch,
                    runner_state.last_done,
                    avail_actions
                )

                # SELECT ACTION

                actor_mean, actor_std = runner_state.train_state.apply_fn(
                    runner_state.train_state.params, # changed from runner_state.train_state.actor.params for crossplay but this breaks for normal training
                    actor_in,
                    )  

                actor_std = jnp.expand_dims(actor_std, axis=1)
                pi = distrax.MultivariateNormalDiag(actor_mean, actor_std)
                rng, act_rng = jax.random.split(rng)
                action, log_prob = pi.sample_and_log_prob(seed=act_rng)
                env_act = unbatchify(action, env.agents)

                # COMPUTE VALUE
                if config["eval"]["compute_value"]:
                    critic_in = runner_state.last_obs["global"]
                    value = runner_state.train_state.critic.apply_fn(
                        runner_state.train_state.critic.params,
                        critic_in,
                    )
                    value = jnp.broadcast_to(value, (env.num_agents, *value.shape))
                else:
                    value = None

                # STEP ENV
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
                    value=(value if log_eval_info.value else None),
                    reward=(reward if log_eval_info.reward else None),
                    log_prob=(log_prob if log_eval_info.log_prob else None),
                    obs=(obs_batch if log_eval_info.obs else None),
                    info=(info if log_eval_info.info else None),
                    avail_actions=(avail_actions if log_eval_info.avail_actions else None),
                    ag_idx=(runner_state.ag_idx if crossplay else None),
                )

                runner_state = RunnerState(
                    train_state=runner_state.train_state,
                    env_state=env_state,
                    last_obs=obsv,
                    last_done=done_batch,
                    update_step=runner_state.update_step,
                    rng=rng,
                    ag_idx=runner_state.ag_idx,
                )
                return runner_state, eval_info


            runner_state, episode_eval_info = jax.lax.scan(
                _env_step, runner_state, None, max_steps
            )
 
            return runner_state, episode_eval_info
        
        # if rngs.ndim == 1:
        #     rngs = jnp.expand_dims(rngs, 0)
        
        if crossplay:
            runner_state, all_episode_eval_infos = jax.lax.scan(
                _run_episode, init_runner_state, rngs
        )
        else:
            runner_state, all_episode_eval_infos = _run_episode(init_runner_state, rngs)
        

        # _, all_episode_eval_infos = jax.lax.scan(
        #     lambda carry, rng: (carry, _run_episode(rng)),
        #     None,
        #     rngs # consider renaming rng for more obvious name
        # )
        return all_episode_eval_infos

    return env, run_evaluation

@hydra.main(version_base=None, config_path="config", config_name="mappo_mabrax")
def main(config):
    config = OmegaConf.to_container(config)
    rng = jax.random.PRNGKey(config["SEED"])
    hparam_rng, run_rng = jax.random.split(rng, 2)


    with jax.disable_jit(config["DISABLE_JIT"]):
        train_jit = jax.jit(
            make_train(config),
            device=jax.devices()[config["DEVICE"]]
        )
        out = train_jit(run_rng, config["LR"], config["ENT_COEF"], config["CLIP_EPS"])


if __name__ == "__main__":
    main()
