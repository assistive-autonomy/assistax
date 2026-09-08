import os
import os.path as osp
import warnings
import uuid
import pandas as pd
import jax
import jax.numpy as jnp
import chex
import distrax
import flax.linen as nn
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict, unflatten_dict
import safetensors.flax
from assistax.envs.multi_agent_env import State, MultiAgentEnv
from assistax.wrappers.baselines import JaxMARLWrapper
from gymnax.environments import spaces
from assistax.wrappers.training import compute_preference_reward
from typing import Sequence, NamedTuple, Any, Dict, Optional, Callable, Tuple, List
import functools
from omegaconf import OmegaConf
from flax.traverse_util import flatten_dict

# Shared pytree helpers (dependency-free module, avoids a utils <-> aht cycle).
from assistax.baselines.tree_utils import _tree_take, _stack_tree, _tree_shape

# debug utility

def get_param_keys(zoo_state) -> list[str]:
    flat_params = flatten_dict(zoo_state.params, sep='/')
    return sorted(flat_params.keys())


def _default_pref_config() -> Dict[str, float]:
    """Return zero-weight preference config (produces zero reward)."""
    return {
        "w_speed": 0.0, "w_force": 0.0, "w_touch": 0.0,
        "speed_range_min": 0.0, "speed_range_max": 1.0,
        "force_range_min": 0.0, "force_range_max": 1.0,
        "reward_budget": 0.0, "overall_weight": 0.0, "touch_threshold": 0.1,
    }


def _extract_pref_config(zoo: "ZooManager", agent_uuid: str) -> Dict[str, float]:
    """Extract preference reward parameters from a zoo agent's config.

    Returns zero weights if the agent has no preference_rewards section,
    so ``compute_preference_reward`` will return 0.
    """
    try:
        config = zoo._load_config(agent_uuid)
    except Exception:
        return _default_pref_config()

    pref = config.get("ENV_KWARGS", {}).get("preference_rewards", None)
    if pref is None:
        return _default_pref_config()

    pw = pref.get("preference_weights", {})
    pr = pref.get("preference_ranges", {})
    speed_range = pr.get("speed_range", [0.0, 1.0])
    force_range = pr.get("force_range", [0.0, 1.0])

    return {
        "w_speed": float(pw.get("speed_preference", 0.0)),
        "w_force": float(pw.get("force_preference", 0.0)),
        "w_touch": float(pw.get("touch_penalty", 0.0)),
        "speed_range_min": float(speed_range[0]),
        "speed_range_max": float(speed_range[1]),
        "force_range_min": float(force_range[0]),
        "force_range_max": float(force_range[1]),
        "reward_budget": float(pref.get("reward_budget", 1.5)),
        "overall_weight": float(pref.get("overall_weight", 1.0)),
        "touch_threshold": float(pref.get("touch_threshold", 0.1)),
    }


def _stack_pref_configs(pref_config_list: List[Dict[str, float]]) -> Dict[str, jnp.ndarray]:
    """Stack per-agent pref config dicts into a dict of JAX arrays, each shaped ``(pop_size,)``."""
    return {
        key: jnp.array([cfg[key] for cfg in pref_config_list])
        for key in pref_config_list[0]
    }


def _zoo_agents_expect_pref_obs(
    env: MultiAgentEnv,
    zoo: "ZooManager",
    load_agents_uuids: Dict,
) -> bool:
    """Check whether loaded zoo agents were trained with +7 pref observations."""
    base_obs_dim = env.observation_spaces[env.agents[0]].shape[0]
    for algo_dict in load_agents_uuids.values():
        for agent, uuids in algo_dict.items():
            first_uuid = uuids[0] if isinstance(uuids, list) else uuids
            cfg = zoo._load_config(first_uuid)
            if cfg.get("OBS_DIM", base_obs_dim) > base_obs_dim:
                return True
    return False


@struct.dataclass
class ActorCriticOutput:
    pi: Optional[chex.Array | Tuple[chex.Array,chex.Array]] = None
    V: Optional[chex.Array] = None
    hstate: Optional[chex.Array] = None

class AgentIdxState(NamedTuple):
    ag_idx = None

# Define Network architectures.
# TODO would be nice to import these instead of copying them here.
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            # assume resets comes in with shape (n_step,)
            jnp.expand_dims(resets,-1),
            self.initialize_carry(rnn_state.shape),
            rnn_state
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(hidden_shape):
        hidden_size = hidden_shape[-1]
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), hidden_shape)


class IPPOActorCritic(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
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

        return ActorCriticOutput(
            pi=pi,
            V=jnp.squeeze(critic, axis=-1),
            hstate=None,
        )


class IPPOActorCriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        embedding = nn.Dense(
            self.config["network"]["embedding_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)

        actor_mean = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
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

        return ActorCriticOutput(
            pi=pi,
            V=jnp.squeeze(critic, axis=-1),
            hstate=hstate,
        )


class MAPPOActor(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
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

        return ActorCriticOutput(
            pi=pi,
            V=None,
            hstate=None,
        )


class MAPPOActorRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hstate, x):
        if self.config["network"]["activation"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        obs, done, avail_actions = x

        embedding = nn.Dense(
            self.config["network"]["embedding_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, done)
        hstate, embedding = ScannedRNN()(hstate, rnn_in)

        actor_mean = nn.Dense(
            self.config["network"]["gru_hidden_dim"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0)
        )(embedding)
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
        return ActorCriticOutput(
            pi=pi,
            V=None,
            hstate=hstate,
        )

class SACActor(nn.Module):
    config: Dict
    
    @nn.compact
    def __call__(self, hstate, x):
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
        pi = actor_mean, jnp.exp(actor_log_std) # could try softplus instead or just return log_std and then do the transformation after 

        return ActorCriticOutput(
            pi=pi,
            V=None,
            hstate=None,
        )


@struct.dataclass
class ZooState:
    agent_uuid: str
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict
    hstate_reset_fn: Callable = struct.field(pytree_node=False, default=lambda x: None)


@struct.dataclass
class LoadNetworkState:
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Dict
    hstate_reset_fn: Callable = struct.field(pytree_node=False, default=lambda x: None)
    pop_size: int = 1
    uuids: Optional[List[str]] = None


@struct.dataclass
class LoadAgentState:
    _state: State
    ag_idx: Dict[str, chex.Array]
    load_agent_actions: Dict[str, chex.Array]
    hstate: Optional[Dict[str, chex.Array]] = None
    prev_contact_force: Optional[jnp.ndarray] = None
    all_partner_actions: Optional[Dict[str, chex.Array]] = None

    def __getattr__(self, name: str):
        return getattr(self._state, name)


class ZooManager:
    """Class for managing access to the agent zoo."""

    def __init__(self, zoo_path: str):
        self.zoo_path = zoo_path
        self.index_path = osp.join(zoo_path, "index.csv")
        self.index_cols = [
            "agent_uuid",
            "scenario",
            "scenario_agent_id",
            "algorithm",
            "is_rnn",
            "rnn_dim",
            "team_uuid",
            "w_speed",
            "w_force",
            "w_touch",
        ]

        self._init_zoo(zoo_path)
        self.index = pd.read_csv(self.index_path)
        # Migrate old zoo index files that lack newer columns
        missing_cols = [c for c in self.index_cols if c not in self.index.columns]
        if missing_cols:
            for col in missing_cols:
                self.index[col] = ""
            self.index.to_csv(self.index_path, index=False)

    def load_agent(self, agent_uuid: str) -> ZooState:
        """Load an agent from the zoo given an agent UUID."""
        apply_fn, hstate_reset_fn = self._load_architecture(agent_uuid)
        
        return ZooState(
            agent_uuid=agent_uuid,
            apply_fn=apply_fn,
            hstate_reset_fn=hstate_reset_fn,
            params=self._load_safetensors(agent_uuid),
        )

    def _load_safetensors(self, agent_uuid: str) -> Dict:
        return unflatten_dict(
            safetensors.flax.load_file(osp.join(self.zoo_path, "params", agent_uuid+".safetensors")),
            sep='/'
        )

    def _save_safetensors(self, agent_uuid, param_dict):
        safetensors.flax.save_file(
            flatten_dict(param_dict, sep='/'),
            osp.join(self.zoo_path, "params", agent_uuid+".safetensors")
        )

    def _load_config(self, agent_uuid: str) -> Dict:
        config_path = osp.join(self.zoo_path, "config", agent_uuid+".yaml")
        return OmegaConf.to_container(
            OmegaConf.load(config_path),
            resolve=True
        )

    def _save_config(self, agent_uuid: str, config):
        if hasattr(config["OBS_DIM"], "item"):
            config["OBS_DIM"] = config["OBS_DIM"].item()
        if hasattr(config["ACT_DIM"], "item"):
            config["ACT_DIM"] = config["ACT_DIM"].item()
        if ("GOBS_DIM" in config) and hasattr(config["GOBS_DIM"], "item"):
            config["GOBS_DIM"] = config["GOBS_DIM"].item()

        OmegaConf.save(
            config,
            osp.join(self.zoo_path, "config", agent_uuid+".yaml")
        )


    def _load_architecture(self, agent_uuid: str) -> Tuple[Callable, Callable]:
        agent_config = self._load_config(agent_uuid)
        is_recurrent = agent_config["network"]["recurrent"]
        alg = agent_config["ALGORITHM"]
        recurrent_dim_size = agent_config["network"].get("gru_hidden_dim")

        def _no_rnn_hstate_reset_fn(key):
            return None

        def _zero_rnn_hstate_reset_fn(key):
            return jnp.zeros((recurrent_dim_size,))

        if is_recurrent:
            if alg == "IPPO":
                apply_fn = IPPOActorCriticRNN(config=agent_config).apply
            elif alg == "MAPPO":
                apply_fn = MAPPOActorRNN(config=agent_config).apply
            else:
                raise Exception(f"Unknown Algorithm {alg}")
            hstate_reset_fn = _zero_rnn_hstate_reset_fn
        else:
            if alg == "IPPO":
                apply_fn = IPPOActorCritic(config=agent_config).apply
            elif alg == "MAPPO":
                apply_fn = MAPPOActor(config=agent_config).apply
            elif alg == "MASAC":
                apply_fn = SACActor(config=agent_config).apply
            else:
                raise Exception(f"Unknown Algorithm {alg}")
            hstate_reset_fn = _no_rnn_hstate_reset_fn
        return apply_fn, hstate_reset_fn

    def _init_zoo(self, zoo_path):
        """Initialises the zoo directory."""
        os.makedirs(zoo_path, exist_ok=True)
        os.makedirs(osp.join(zoo_path, "config"), exist_ok=True)
        os.makedirs(osp.join(zoo_path, "params"), exist_ok=True)
        if not osp.exists(self.index_path):
            with open(self.index_path, "w", encoding="utf-8") as f:
                f.write(','.join(self.index_cols) + '\n')

    def _write_index(self, index_dict):
        """Writes the details of the current agent to the index file."""
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(','.join(str(index_dict[col]) for col in self.index_cols) + '\n')

    def save_agent(self, config, param_dict, scenario_agent_id, team_uuid="", preference_weights=None):
        """Saves the current agent to the zoo."""
        agent_uuid = str(uuid.uuid4())
        self._save_safetensors(agent_uuid, param_dict)
        self._save_config(agent_uuid, config)
        pw = preference_weights or {}
        self._write_index({
            "agent_uuid": agent_uuid,
            "scenario": config["ENV_NAME"],
            "scenario_agent_id": scenario_agent_id,
            "algorithm": config["ALGORITHM"],
            "is_rnn": config["network"]["recurrent"],
            "rnn_dim": config["network"].get("gru_hidden_dim", 0),
            "team_uuid": team_uuid,
            "w_speed": pw.get("w_speed", ""),
            "w_force": pw.get("w_force", ""),
            "w_touch": pw.get("w_touch", ""),
        })


class LoadAgentWrapper(JaxMARLWrapper):
    def __init__(
        self,
        env: MultiAgentEnv,
        load_agents: Dict[str, LoadNetworkState],
        pref_configs: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
        agents_expect_pref_obs: bool = True,
    ):
        super().__init__(env)

        self.loaded_agents = ['human']
        self.loaded_params = load_agents
        self.pref_configs = pref_configs
        self.agents = [
            agent
            for agent in self._env.agents
            if agent not in self.loaded_agents
        ]
        self.num_agents = len(self.agents)
        self.num_loaded_agents = len(self.loaded_agents)

        # Expand observation spaces if preference configs are present
        self._num_pref_obs = 7 if (self.pref_configs is not None and agents_expect_pref_obs) else 0
        if self._num_pref_obs > 0:
            self._env.observation_spaces = {
                agent: spaces.Box(
                    -jnp.inf, jnp.inf,
                    shape=(space.shape[0] + self._num_pref_obs,),
                )
                for agent, space in self._env.observation_spaces.items()
            }

    @classmethod
    def load_from_zoo(
        cls,
        env: MultiAgentEnv,
        zoo: ZooManager | str,
        load_agents_uuids: Dict[str, str | list[str]],
    ):
        """Loads agents from a zoo using ZooManager and groups them by algorithm.
        
        This implementation groups agents into three categories: "IPPO", "MAPPO", and "MASAC".
        It uses the agent configuration (via zoo._load_config) to determine the algorithm.
        """
        if isinstance(zoo, str):
            zoo = ZooManager(zoo_path=zoo)

        load_agents: Dict[str, Dict[str, LoadNetworkState]] = {}
        pref_config_lists: Dict[str, List[Dict[str, float]]] = {}
        for algorithm, agents_dict in load_agents_uuids.items():
            if algorithm not in load_agents:
                load_agents[algorithm] = {}
            for agent, agent_uuids in agents_dict.items():
                if agent not in pref_config_lists:
                    pref_config_lists[agent] = []
                if isinstance(agent_uuids, str):
                    # Single agent case.
                    zoo_state = zoo.load_agent(agent_uuids)
                    pref_config_lists[agent].append(_extract_pref_config(zoo, agent_uuids))
                    load_agents[algorithm][agent] = LoadNetworkState(
                        apply_fn=jax.vmap(zoo_state.apply_fn, in_axes=(0, None, None)),
                        hstate_reset_fn=zoo_state.hstate_reset_fn,
                        params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), zoo_state.params),
                        pop_size=1,
                    )
                else:
                    # Multiple agents: load each zoo_state.
                    zoo_states = [zoo.load_agent(agent_uuid) for agent_uuid in agent_uuids]
                    for agent_uuid in agent_uuids:
                        pref_config_lists[agent].append(_extract_pref_config(zoo, agent_uuid))

                    # group the zoo states by their parameter shapes
                    shape_groups = {}
                    for agent_uuid, zs in zip(agent_uuids, zoo_states):

                        flat_shapes, _ = jax.tree_util.tree_flatten(_tree_shape(zs.params))
                        shape_key = tuple(flat_shapes)
                        shape_groups.setdefault(shape_key, []).append(agent_uuid)

                    # if more than one group exists there is a shape mismatch raise error and return which uuids are wrong to help fix
                    if len(shape_groups) > 1:
                        raise ValueError(
                            f"Mismatching parameter shapes for agent '{agent}' under algorithm '{algorithm}'.\n"
                            f"Groups by shape signature (each key is a tuple of shapes): {shape_groups}"
                        )


                    load_agents[algorithm][agent] = LoadNetworkState(
                        apply_fn=jax.vmap(zoo_states[0].apply_fn, in_axes=(0, None, None)), #TODO and NOTE! this throughs an error when wrong zoo path is provided which is not very insightful we should have a better error
                        hstate_reset_fn=zoo_states[0].hstate_reset_fn,
                        params=_stack_tree([zs.params for zs in zoo_states]),
                        pop_size=len(zoo_states),
                    )

        stacked_pref_configs = {
            agent: _stack_pref_configs(configs)
            for agent, configs in pref_config_lists.items()
            if configs
        }
        expect_pref_obs = _zoo_agents_expect_pref_obs(env, zoo, load_agents_uuids) if stacked_pref_configs else False
        return cls(env, load_agents, pref_configs=stacked_pref_configs or None, agents_expect_pref_obs=expect_pref_obs)

    def take_internal_action(
        self,
        key: chex.PRNGKey,
        obs: Dict[str, chex.Array],
        dones: Dict[str, bool],
        avail_actions: Dict[str, chex.Array],
        hstate: Dict[str, Dict[str, chex.Array]],
    ) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array]]:
        """
        Compute the action taken by each of the loaded agents and the new RNN hidden state.

        Here, the loaded parameters are nested by algorithm and then by agent (e.g. { "IPPO": { "human": ... },
        "MAPPO": { "human": ... }, ... } ).  This function computes actions for each (algorithm, agent)
        pair sequentially. Finally, for each agent (e.g. "human"), the actions from all algorithms are concatenated
        along the first (population) dimension. The resulting dictionary will have the shape:

            { 'human': (total_n_agents, action_size) }
        """
        # Temporary containers to accumulate the actions and new hidden states per agent.
        temp_actions = {}   # keys: agent, values: list of action arrays
        hstates = {}   # keys: agent, values: list of hstate arrays

        # Iterate over each algorithm group.
        for algorithm, agents_dict in self.loaded_params.items():
            # For each agent within this algorithm.
            temp_hstates = {}
            for agent, train_state in agents_dict.items():
                # Split the key for each call.
                key, subkey = jax.random.split(key)
                
                # Retrieve the observation, done flag, and available actions using the agent key.
                # (We assume that these dictionaries are keyed by the agent name, not algorithm.)
                network_out = train_state.apply_fn(
                    train_state.params,
                    hstate[algorithm][agent],
                    (obs[agent], dones[agent], avail_actions[agent])
                )
                
                # Create a distribution from the network output and sample an action.
                pi = distrax.MultivariateNormalDiag(*network_out.pi)
                action = pi.sample(seed=subkey)
                
                # Accumulate actions and new hidden states.
                if agent not in temp_actions:
                    temp_actions[agent] = []
                temp_actions[agent].append(action)
                
                
                temp_hstates[agent] = network_out.hstate
            
            hstates[algorithm] = temp_hstates
        
        final_actions = {agent: jnp.concatenate(actions, axis=0) for agent, actions in temp_actions.items()} # turn into {'human': (total_n_agents, action_size)}

        return final_actions, hstates

    def reset_internal_hstates(self, key: chex.PRNGKey) -> Dict[str, chex.Array]:
        """Reset the hstate for each of the loaded agents."""
        hstates = {}
        for algorithm, agents_dict in self.loaded_params.items():
            hstates[algorithm] = {
                agent: train_state.hstate_reset_fn(_key)
                for _key, (agent, train_state) in zip(
                    jax.random.split(key, self.num_loaded_agents), agents_dict.items()
                    )
                }
            
        return hstates
    
    def reset_agent_index(self, key: chex.PRNGKey) -> Dict[str, chex.Array]:
        """
        Reset the agent population ID for each loaded agent.
        
        self.loaded_params has shape:
        {
            "IPPO": { "human": train_state, ... },
            "MAPPO": { "human": train_state, ... },
            "MASAC": { "human": train_state, ... }
        }
        
        For each agent (e.g. "human") we want to combine the population sizes from all algorithms and sample
        a single random index in [0, total_population).
        
        The final returned dictionary is flat and looks like:
        { "human": index, ... }
        """
        # First, accumulate the total population size for each agent across algorithms.
        combined_pop = {}  # e.g. { "human": total_population, ... }
        for algo, agents in self.loaded_params.items():
            for agent, train_state in agents.items():
                pop = train_state.pop_size 
                if agent in combined_pop:
                    combined_pop[agent] += pop
                else:
                    combined_pop[agent] = pop

        agent_keys = jax.random.split(key, len(combined_pop))

        indices = {}
        for (agent, total_pop), subkey in zip(combined_pop.items(), agent_keys):
            indices[agent] = jax.random.randint(subkey, shape=(), minval=0, maxval=total_pop)

        return indices

    def _get_pref_obs_vector(self, ag_idx: Dict[str, chex.Array]) -> jnp.ndarray:
        """Build the 7-value preference obs vector for the current partner."""
        idx = ag_idx["human"]
        pref = self.pref_configs["human"]
        return jnp.array([
            pref["w_speed"][idx], pref["w_force"][idx], pref["w_touch"][idx],
            pref["speed_range_min"][idx], pref["speed_range_max"][idx],
            pref["force_range_min"][idx], pref["force_range_max"][idx],
        ])

    def _append_pref_to_obs(
        self, obs: Dict[str, jnp.ndarray], ag_idx: Dict[str, chex.Array],
    ) -> Dict[str, jnp.ndarray]:
        """Append the current partner's pref vector to each agent's observation."""
        pref_vec = self._get_pref_obs_vector(ag_idx)
        return {agent: jnp.concatenate([o, pref_vec]) for agent, o in obs.items()}

    def _compute_partner_pref_reward(
        self,
        states_st: State,
        ag_idx: Dict[str, chex.Array],
        prev_contact_force: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute preference reward for the current partner (single env, no inner vmap)."""
        human_idx = ag_idx["human"]
        pref = self.pref_configs["human"]

        total_pref_reward, updated_cf, pref_components = compute_preference_reward(
            speed=states_st.info["ee_speed"],
            force=states_st.info["ee_force"],
            prev_contact_force=prev_contact_force,
            w_speed=pref["w_speed"][human_idx],
            w_force=pref["w_force"][human_idx],
            w_touch=pref["w_touch"][human_idx],
            speed_range_min=pref["speed_range_min"][human_idx],
            speed_range_max=pref["speed_range_max"][human_idx],
            force_range_min=pref["force_range_min"][human_idx],
            force_range_max=pref["force_range_max"][human_idx],
            reward_budget=pref["reward_budget"][human_idx],
            overall_weight=pref["overall_weight"][human_idx],
            touch_threshold=pref["touch_threshold"][human_idx],
        )
        return total_pref_reward, updated_cf, pref_components

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], LoadAgentState]:
        """Resets the environment and initialises the loaded agent state."""
        key_env, key_hstate, key_action, key_ag_idx = jax.random.split(key, 4)
        obs, state = self._env.reset(key_env)
        dones = {agent: False for agent in self.loaded_agents}
        avail_actions = self._env.get_avail_actions(state)
        hstate = self.reset_internal_hstates(key_hstate)

        ag_idx = self.reset_agent_index(key_ag_idx)

        # Append preference obs before take_internal_action so zoo networks get correct input dim
        if self._num_pref_obs > 0:
            obs = self._append_pref_to_obs(obs, ag_idx)

        all_partner_actions, hstate = self.take_internal_action(
            key_action, obs, dones, avail_actions, hstate
        )
        load_agent_actions = jax.tree.map(lambda i, a: a[i], ag_idx, all_partner_actions)

        init_prev_cf = jnp.array(0.0) if self.pref_configs is not None else None

        if self.pref_configs is not None:
            state = state.replace(metrics={
                **state.metrics,
                "speed_pref_reward": jnp.zeros(()),
                "force_pref_reward": jnp.zeros(()),
                "touch_penalty_reward": jnp.zeros(()),
                "total_pref_reward": jnp.zeros(()),
                "pref_raw_speed": jnp.zeros(()),
                "pref_raw_force": jnp.zeros(()),
            })

        state = LoadAgentState(
            _state=state,
            load_agent_actions=load_agent_actions,
            hstate=hstate,
            ag_idx=ag_idx,
            prev_contact_force=init_prev_cf,
            all_partner_actions=all_partner_actions,
        )
        return obs, state

    def step(
        self,
        key: chex.PRNGKey,
        state: LoadAgentState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[LoadAgentState]=None,
    ):
        """Performs step transitions in the environment."""

        key_step, key_reset, key_action, key_ag_idx = jax.random.split(key, 4)

        # read in the loaded agent actions from the state
        actions = {**state.load_agent_actions, **actions}

        obs_st, states_st, rewards, dones, infos = self._env.step_env(
            key_step, state._state, actions
        )

        # Augment rewards with the current partner's preference reward
        if self.pref_configs is not None:
            pref_reward, new_cf, pref_components = self._compute_partner_pref_reward(
                states_st, state.ag_idx, state.prev_contact_force
            )
            rewards = {agent: rewards[agent] + pref_reward for agent in rewards}
            states_st = states_st.replace(metrics={**states_st.metrics, **pref_components})
            new_prev_cf = jnp.where(dones["__all__"], 0.0, new_cf)
        else:
            new_prev_cf = state.prev_contact_force

        if reset_state is None:
            obs_re, states_re = self._env.reset(key_reset)
            ag_idx_re = self.reset_agent_index(key_ag_idx)
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)
            ag_idx_re = reset_state.ag_idx

        # Pad reset metrics with zero-valued pref keys so pytree structures match
        if self.pref_configs is not None:
            zero_pref = {k: jnp.zeros_like(v) for k, v in pref_components.items()}
            states_re = states_re.replace(metrics={**states_re.metrics, **zero_pref})

        # Auto-reset environment based on termination
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st,
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        ag_idx = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), ag_idx_re, state.ag_idx
        )

        # Append preference obs to each agent's observation
        if self._num_pref_obs > 0:
            obs = self._append_pref_to_obs(obs, ag_idx)

        # Take the next action with the loaded agents
        avail_actions = self._env.get_avail_actions(state)

        all_partner_actions, load_agent_hstate = self.take_internal_action(
            key_action, obs, dones, avail_actions, state.hstate,
        )
        load_agent_actions = jax.tree.map(lambda i, a: a[i], ag_idx, all_partner_actions)
        states = LoadAgentState(
            _state=states,
            load_agent_actions=load_agent_actions,
            hstate=load_agent_hstate,
            ag_idx=ag_idx,
            prev_contact_force=new_prev_cf,
            all_partner_actions=all_partner_actions,
        )

        return obs, states, rewards, dones, infos


def extract_uuids_from_eval_results(env_wrapper, eval_results):
    """
    Extract agent UUIDs from evaluation results.
    Handles the tiled agent indices pattern.
    """
    uuid_info = {}
    
    if hasattr(eval_results, 'info') and eval_results.info is not None:
        if 'agent_indices' in eval_results.info:
            agent_indices = eval_results.info['agent_indices']
            
            for agent_type in env_wrapper.loaded_agents:
                if agent_type in agent_indices:
                    indices = agent_indices[agent_type]
                    
                    # Since we tiled the indices to [64, 2], we only need the first column
                    # as both columns contain the same values
                    if hasattr(indices, 'shape') and len(indices.shape) > 1:
                        # Just take the first column to get original indices
                        indices = indices[:, 0]
                    
                    # Convert to Python list if it's a JAX array
                    if hasattr(indices, 'tolist'):
                        indices = indices.tolist()
                    
                    # Ensure indices is a list (handle the single index case)
                    if not isinstance(indices, list):
                        indices = [indices]
                    
                    # Get UUIDs for each index
                    uuid_info[agent_type] = [
                        env_wrapper.get_uuid(agent_type, idx)
                        for idx in indices
                    ]
    
    return uuid_info

class LoadEvalAgentWrapper(JaxMARLWrapper):
    def __init__(
        self,
        env: MultiAgentEnv,
        load_agents: Dict[str, LoadNetworkState],
        pref_configs: Optional[Dict[str, Dict[str, jnp.ndarray]]] = None,
        agents_expect_pref_obs: bool = True,
    ):
        super().__init__(env)
        self.loaded_agents = ['human'] # also currently hard coded this works for assistax but not other JaxMARL envs
        self.loaded_params = load_agents
        self.pref_configs = pref_configs
        self.agents = [
            agent for agent in self._env.agents if agent not in self.loaded_agents
        ] # might need to change this to avoid breaking eval
        # self.agents = self._env.agents
        self.num_agents = len(self.agents)
        self.num_loaded_agents = len(self.loaded_agents)
        self.total_pop_size = sum([train_state.pop_size for agents_dict in self.loaded_params.values() for train_state in agents_dict.values()])
        self.idx_mapping = self._create_uuid_mapping()
        # self.idxs = self._init_idxs()
        # self.current_idx = {agent_type: 0 for agent_type in self.loaded_agents}
        # self.idx_mask = {agent_type: jax.nn.one_hot(0, self.total_pop_size, dtype=int) for agent_type in self.loaded_agents}

        # Expand observation spaces if preference configs are present
        self._num_pref_obs = 7 if (self.pref_configs is not None and agents_expect_pref_obs) else 0
        if self._num_pref_obs > 0:
            self._env.observation_spaces = {
                agent: spaces.Box(
                    -jnp.inf, jnp.inf,
                    shape=(space.shape[0] + self._num_pref_obs,),
                )
                for agent, space in self._env.observation_spaces.items()
            }

    
    def _create_uuid_mapping(self):
        """Create a dictionary for UUID lookups"""
        mapping = {}
        for agent_type in self.loaded_agents:
            agent_mapping = {}
            index = 0
            
            for algo, agents_dict in self.loaded_params.items():
                if agent_type in agents_dict:
                    train_state = agents_dict[agent_type]
                    if hasattr(train_state, 'uuids') and train_state.uuids is not None:
                        for i, uuid in enumerate(train_state.uuids):
                            agent_mapping[index + i] = uuid
                    else:
                        # If UUIDs aren't available, use placeholders
                        for i in range(train_state.pop_size):
                            agent_mapping[index + i] = f"{algo}_{agent_type}_{i}"
                    
                    index += train_state.pop_size
            
            mapping[agent_type] = agent_mapping
        
        return mapping
    
    @classmethod
    def load_from_zoo(
        cls,
        env: MultiAgentEnv,
        zoo: ZooManager | str,
        load_agents_uuids: Dict[str, str | list[str]],
    ):
        """Loads agents from a zoo using ZooManager and groups them by algorithm."""
        
        if isinstance(zoo, str):
            zoo = ZooManager(zoo_path=zoo)

        load_agents: Dict[str, Dict[str, LoadNetworkState]] = {}
        pref_config_lists: Dict[str, List[Dict[str, float]]] = {}

        for algorithm, agents_dict in load_agents_uuids.items():
            if algorithm not in load_agents:
                load_agents[algorithm] = {}

            for agent, agent_uuids in agents_dict.items():
                if agent not in pref_config_lists:
                    pref_config_lists[agent] = []

                if isinstance(agent_uuids, str):
                    try:
                        zoo_state = zoo.load_agent(agent_uuids)
                    except FileNotFoundError:
                        warnings.warn(
                            f"Agent file for UUID {agent_uuids} not found; skipping agent '{agent}' under algorithm '{algorithm}'."
                        )
                        continue
                    pref_config_lists[agent].append(_extract_pref_config(zoo, agent_uuids))
                    load_agents[algorithm][agent] = LoadNetworkState(
                        apply_fn=jax.vmap(zoo_state.apply_fn, in_axes=(0, None, None)),
                        hstate_reset_fn=zoo_state.hstate_reset_fn,
                        params=jax.tree.map(lambda x: jnp.expand_dims(x, 0), zoo_state.params),
                        pop_size=1,
                        uuids=[agent_uuids],
                    )
                else:
                    zoo_states = []
                    successful_agent_uuids = []
                    for agent_uuid in agent_uuids:
                        try:
                            state = zoo.load_agent(agent_uuid)
                            zoo_states.append(state)
                            successful_agent_uuids.append(agent_uuid)
                        except FileNotFoundError:
                            warnings.warn(f"Agent file for UUID {agent_uuid} not found; skipping.")
                    if not zoo_states:
                        warnings.warn(
                            f"No valid agents loaded for '{agent}' under algorithm '{algorithm}'."
                        )
                        continue

                    for agent_uuid in successful_agent_uuids:
                        pref_config_lists[agent].append(_extract_pref_config(zoo, agent_uuid))

                    # Group the zoo states by their parameter shapes.
                    shape_groups = {}
                    for agent_uuid, zs in zip(successful_agent_uuids, zoo_states):
                        flat_shapes, _ = jax.tree_util.tree_flatten(_tree_shape(zs.params))
                        shape_key = tuple(flat_shapes)
                        shape_groups.setdefault(shape_key, []).append(agent_uuid)

                    if len(shape_groups) > 1:
                        raise ValueError(
                            f"Mismatching parameter shapes for agent '{agent}' under algorithm '{algorithm}'.\n"
                            f"Groups by shape signature (each key is a tuple of shapes): {shape_groups}"
                        )

                    load_agents[algorithm][agent] = LoadNetworkState(
                        apply_fn=jax.vmap(zoo_states[0].apply_fn, in_axes=(0, None, None)),
                        hstate_reset_fn=zoo_states[0].hstate_reset_fn,
                        params=_stack_tree([zs.params for zs in zoo_states]),
                        pop_size=len(zoo_states),
                        uuids=successful_agent_uuids,
                    )

        stacked_pref_configs = {
            agent: _stack_pref_configs(configs)
            for agent, configs in pref_config_lists.items()
            if configs
        }
        expect_pref_obs = _zoo_agents_expect_pref_obs(env, zoo, load_agents_uuids) if stacked_pref_configs else False
        return cls(env, load_agents, pref_configs=stacked_pref_configs or None, agents_expect_pref_obs=expect_pref_obs)

    def take_internal_action(
        self,
        key: chex.PRNGKey,
        obs: Dict[str, chex.Array],
        dones: Dict[str, bool],
        avail_actions: Dict[str, chex.Array],
        hstate: Dict[str, Dict[str, chex.Array]],
    ) -> Tuple[Dict[str, chex.Array], Dict[str, chex.Array]]:
        """
        Compute the action taken by each of the loaded agents and update the corresponding hidden state.
        Actions from each algorithm are concatenated along the population dimension.
        """
        temp_actions = {}  # keys: agent, values: list of action arrays
        hstates = {}       # keys: algorithm, values: {agent: new hstate}

        # Iterate over each algorithm group.
        for algorithm, agents_dict in self.loaded_params.items():
            temp_hstates = {}
            for agent, train_state in agents_dict.items():
                key, subkey = jax.random.split(key)
                network_out = train_state.apply_fn(
                    train_state.params,
                    hstate[algorithm][agent],
                    (obs[agent], dones[agent], avail_actions[agent])
                )
                pi = distrax.MultivariateNormalDiag(*network_out.pi)
                action = pi.sample(seed=subkey)
                
                if agent not in temp_actions:
                    temp_actions[agent] = []
                temp_actions[agent].append(action)
                temp_hstates[agent] = network_out.hstate
            
            hstates[algorithm] = temp_hstates
        
        # Concatenate actions for each agent across all algorithm groups.
        final_actions = {
            agent: jnp.concatenate(actions, axis=0) for agent, actions in temp_actions.items()
        }

        return final_actions, hstates

    def reset_internal_hstates(self, key: chex.PRNGKey) -> Dict[str, chex.Array]:
        """Reset the hidden states for each of the loaded agents."""
        hstates = {}
        for algorithm, agents_dict in self.loaded_params.items():
            hstates[algorithm] = {
                agent: train_state.hstate_reset_fn(_key)
                for _key, (agent, train_state) in zip(
                    jax.random.split(key, self.num_loaded_agents), agents_dict.items()
                )
            }
        return hstates

    def _get_pref_obs_vector(self, ag_idx: Dict[str, chex.Array]) -> jnp.ndarray:
        """Build the 7-value preference obs vector for the current partner."""
        idx = ag_idx["human"]
        pref = self.pref_configs["human"]
        return jnp.array([
            pref["w_speed"][idx], pref["w_force"][idx], pref["w_touch"][idx],
            pref["speed_range_min"][idx], pref["speed_range_max"][idx],
            pref["force_range_min"][idx], pref["force_range_max"][idx],
        ])

    def _append_pref_to_obs(
        self, obs: Dict[str, jnp.ndarray], ag_idx: Dict[str, chex.Array],
    ) -> Dict[str, jnp.ndarray]:
        """Append the current partner's pref vector to each agent's observation."""
        pref_vec = self._get_pref_obs_vector(ag_idx)
        return {agent: jnp.concatenate([o, pref_vec]) for agent, o in obs.items()}

    def _compute_partner_pref_reward(
        self,
        states_st: State,
        ag_idx: Dict[str, chex.Array],
        prev_contact_force: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
        """Compute preference reward for the current partner (single env, no inner vmap)."""
        human_idx = ag_idx["human"]
        pref = self.pref_configs["human"]

        total_pref_reward, updated_cf, pref_components = compute_preference_reward(
            speed=states_st.info["ee_speed"],
            force=states_st.info["ee_force"],
            prev_contact_force=prev_contact_force,
            w_speed=pref["w_speed"][human_idx],
            w_force=pref["w_force"][human_idx],
            w_touch=pref["w_touch"][human_idx],
            speed_range_min=pref["speed_range_min"][human_idx],
            speed_range_max=pref["speed_range_max"][human_idx],
            force_range_min=pref["force_range_min"][human_idx],
            force_range_max=pref["force_range_max"][human_idx],
            reward_budget=pref["reward_budget"][human_idx],
            overall_weight=pref["overall_weight"][human_idx],
            touch_threshold=pref["touch_threshold"][human_idx],
        )
        return total_pref_reward, updated_cf, pref_components

    def reset_agent_index( # probably actually don't even need this anymore
        self, current_idx: Dict[str, chex.Array]
    ) -> Dict[str, int]:
        """
        Instead of sampling a random index for each loaded agent, cycle through all agents.

        We use multiply the arange index array by the onehot current index to get the index.
        After this we roll the onehot mask forward by +1
        """

        ag_index = {}
        for agent_type in self.loaded_agents:
            if current_idx is None:
                ag_index[agent_type] = -1
            else:
                ag_index[agent_type] = (current_idx[agent_type] + 1) % self.total_pop_size

        return ag_index

    def reset(self, key: chex.PRNGKey, current_idx: Optional[Dict[str, chex.Array]]) -> Tuple[Dict[str, chex.Array], LoadAgentState]:
        """
        Reset the environment and initialize the loaded agent state.
        Instead of randomly selecting a loaded agent, we initialize the agent index to 0.
        """
        key_env, key_hstate, key_action = jax.random.split(key, 3)
        obs, state = self._env.reset(key_env)
        dones = {agent: False for agent in self.loaded_agents}
        avail_actions = self._env.get_avail_actions(state)
        hstate = self.reset_internal_hstates(key_hstate)

        # Initialize indices deterministically (starting at 0).
        current_idx = self.reset_agent_index(current_idx)

        current_idx = self._preprocess_current_idx(current_idx) # ensure its not int
        # Ensure each index is a scalar
        current_idx = jax.tree.map(self._ensure_scalar_idx, current_idx)

        # Append preference obs before take_internal_action so zoo networks get correct input dim
        if self._num_pref_obs > 0:
            obs = self._append_pref_to_obs(obs, current_idx)

        load_agent_actions, hstate = self.take_internal_action(
            key_action, obs, dones, avail_actions, hstate
        )

        # Use indices to select actions
        load_agent_actions = jax.tree.map(lambda i, a: a[i], current_idx, load_agent_actions)

        init_prev_cf = jnp.array(0.0) if self.pref_configs is not None else None

        if self.pref_configs is not None:
            state = state.replace(metrics={
                **state.metrics,
                "speed_pref_reward": jnp.zeros(()),
                "force_pref_reward": jnp.zeros(()),
                "touch_penalty_reward": jnp.zeros(()),
                "total_pref_reward": jnp.zeros(()),
                "pref_raw_speed": jnp.zeros(()),
                "pref_raw_force": jnp.zeros(()),
            })

        state = LoadAgentState(
            _state=state,
            load_agent_actions=load_agent_actions,
            hstate=hstate,
            ag_idx=current_idx,
            prev_contact_force=init_prev_cf,
        )
        return obs, state


    def step(
        self,
        key: chex.PRNGKey,
        state: LoadAgentState,
        actions: Dict[str, chex.Array],
        reset_state: Optional[LoadAgentState] = None,
    ):
        key_step, key_reset, key_action = jax.random.split(key, 3)

        actions = {**state.load_agent_actions, **actions}

        obs_st, states_st, rewards, dones, infos = self._env.step_env(
            key_step, state._state, actions
        )

        # Augment rewards with the current partner's preference reward
        if self.pref_configs is not None:
            pref_reward, new_cf, pref_components = self._compute_partner_pref_reward(
                states_st, state.ag_idx, state.prev_contact_force
            )
            rewards = {agent: rewards[agent] + pref_reward for agent in rewards}
            states_st = states_st.replace(metrics={**states_st.metrics, **pref_components})
            new_prev_cf = jnp.where(dones["__all__"], 0.0, new_cf)
        else:
            new_prev_cf = state.prev_contact_force

        if reset_state is None:
            obs_re, states_re = self._env.reset(key_reset) # TODO: Below is very hacky either get rid entirely or
            ag_idx_re = self.reset_agent_index(state.ag_idx) # This makes it more robust but as we don't have early termination we probs dont need this
        else:
            states_re = reset_state
            obs_re = self.get_obs(states_re)
            ag_idx_re = reset_state.ag_idx

        # Pad reset metrics with zero-valued pref keys so pytree structures match
        if self.pref_configs is not None:
            zero_pref = {k: jnp.zeros_like(v) for k, v in pref_components.items()}
            states_re = states_re.replace(metrics={**states_re.metrics, **zero_pref})

        # Auto-reset environment based on termination.
        states = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st,
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(dones["__all__"], x, y), obs_re, obs_st
        )
        ag_idx = jax.tree.map(lambda x, y: jax.lax.select(dones["__all__"], x, y), ag_idx_re, state.ag_idx # get rid of this grimm_stuff
        )

        # Append preference obs to each agent's observation
        if self._num_pref_obs > 0:
            obs = self._append_pref_to_obs(obs, ag_idx)

        avail_actions = self._env.get_avail_actions(state)
        load_agent_actions, load_agent_hstate = self.take_internal_action(
            key_action, obs, dones, avail_actions, state.hstate,
        )

        load_agent_actions = jax.tree.map(lambda i, a: a[i], ag_idx, load_agent_actions)

        states = LoadAgentState(
            _state=states,
            load_agent_actions=load_agent_actions,
            hstate=load_agent_hstate,
            ag_idx=ag_idx,
            prev_contact_force=new_prev_cf,
        )

        return obs, states, rewards, dones, infos
    
    def get_uuid(self, agent_type, index):
        """
        Get the UUID for a specific agent index.
        This method is intended for use OUTSIDE of JAX-traced code.
        """
        if hasattr(index, 'item'):
            index = index.item()  # Convert JAX array to Python int
        
        if agent_type in self.idx_mapping:
            agent_mapping = self.idx_mapping[agent_type]
            if index in agent_mapping:
                return agent_mapping[index]
        
        return "unknown"
    
    def _ensure_scalar_idx(self, idx_array: chex.Array) -> chex.Array:
        """
        Ensures the index is a scalar (ndim=0).
        If it's an array, checks that all values are identical and returns the first value.
        
        Args:
            idx_array: An array of indices
            
        Returns:
            A scalar index value
        """
        # If already a scalar, return as is
        if idx_array.ndim == 0:
            return idx_array
        
        # Check if all values are equal to the first element
        is_uniform = jnp.all(idx_array == idx_array[0])
        
        # Use JAX's conditional to handle this in a JIT-compatible way
        # If array is uniform, return the first element, otherwise use a predefined value
        result = jax.lax.cond(
            is_uniform,
            lambda _: idx_array[0],  # Return first element if all are the same
            lambda _: jnp.array(-9999),  # Return 0 as fallback (you may want to customize this)
            operand=None
        )
        
        return result
    
    def _preprocess_current_idx(self, current_idx):
        """
        Preprocess current_idx to handle Python int and convert to proper dict format.
        """        
        # If single integer, use same value for all agent types
        if isinstance(current_idx, int):
            return {agent_type: jnp.array(current_idx) for agent_type in self.loaded_agents}
        
        # If already a dict, convert any Python ints to JAX arrays
        processed = {}
        for agent_type, idx in current_idx.items():
            if isinstance(idx, int):
                processed[agent_type] = jnp.array(idx)
            else:
                processed[agent_type] = idx
        
        return processed
