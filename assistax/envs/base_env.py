# Copyright 2024 The JAXMARL Authors, under the Apache License, Version 2.0.
# Copyright 2025 The Assistax Authors.

from typing import Dict, Literal, Optional, Tuple
import chex
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from gymnax.environments import spaces
# from brax import envs
from assistax import envs 
import jax
import jax.numpy as jnp
from functools import partial

# from .mappings import _agent_action_mapping, _agent_observation_mapping

from typing import Dict, List, Tuple, Union
import jax.numpy as jnp

# TODO: programatically generate these mappings from the kinematic trees
#       and add an observation distance parameter to the environment

# Defining the agents action and observation mappings

_agent_action_mapping = {
    "scratchitch": {
        "robot": jnp.array([17, 18, 19, 20, 21, 22, 23]),
        "human": jnp.array([11, 12, 13]),
    },
    "bedbathing": {
        "robot": jnp.array([17, 18, 19, 20, 21, 22, 23]),
        "human": jnp.array([11, 12, 13]),
    },
    "armmanipulation": {
        "robot": jnp.array([17, 18, 19, 20, 21, 22, 23]),
        "human": jnp.array([11, 12, 13]),
    },
    "pushcoop": {
        "robot1": jnp.array([0, 1, 2, 3, 4, 5, 6]),
        "robot2": jnp.array([7, 8, 9, 10, 11, 12, 13]),
    },
}


def listerize(ranges: List[Union[int, Tuple[int, int]]]) -> List[int]:
    return [
        i
        for r in ranges
        for i in (range(r[0], r[1] + 1) if isinstance(r, tuple) else [r])
    ]


ranges: Dict[str, Dict[str, List[Union[int, Tuple[int, int]]]]] = {
    "scratchitch": {
        # "robot": [(0,80)], # Old obs
        # "human": [(81,161)] # Old obs
        "robot": [(0,28)], # New obs
        "human": [(29,68)], # New obs
        "global": [(0,68)],
    },
    "bedbathing": {
        "robot": [(0,24)],
        "human": [(25,60)],
        "global": [(0,60)],
    },
    "armmanipulation": {
        "robot": [(0,28)],
        "human": [(28,67)],
        "global": [(0,67)],
    },
    "pushcoop": {
        "robot1": [(0, 26)],
        "robot2": [(27, 53)],
        "global": [(0, 53)],
    },
}

_agent_observation_mapping = {
    k: {k_: jnp.array(listerize(v_)) for k_, v_ in v.items()} for k, v in ranges.items()
}

# The base class for multi-agent Brax environments taken and adjusted from JAXMARL.

class MABraxEnv(MultiAgentEnv):
    def __init__(
        self,
        env_name: str,
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True,
        homogenisation_method: Optional[Literal["max", "concat"]] = None,
        backend: str = "positional",
        **kwargs
    ):
        """Multi-Agent Brax environment.

        Args:
            env_name: Name of the environment to be used.
            episode_length: Length of an episode. Defaults to 1000.
            action_repeat: How many repeated actions to take per environment
                step. Defaults to 1.
            auto_reset: Whether to automatically reset the environment when
                an episode ends. Defaults to True.
            homogenisation_method: Method to homogenise observations and actions
                across agents. If None, no homogenisation is performed, and
                observations and actions are returned as is. If "max", observations
                and actions are homogenised by taking the maximum dimension across
                all agents and zero-padding the rest. In this case, the index of the
                agent is prepended to the observation as a one-hot vector. If "concat",
                observations and actions are homogenised by masking the dimensions of
                the other agents with zeros in the full observation and action vectors.
                Defaults to None.
        """
        base_env_name = env_name.split("_")[0]
        
        env = envs.create(
            base_env_name, episode_length, action_repeat, auto_reset, backend=backend, **kwargs
        )
        self.env = env
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.auto_reset = auto_reset
        self.homogenisation_method = homogenisation_method
        self.het_reward = kwargs['het_reward'] # adding this
        self.agent_obs_mapping = _agent_observation_mapping[env_name]
        self.agent_action_mapping = _agent_action_mapping[env_name]
        self.agents = list(self.agent_action_mapping.keys())
        self.num_agents = len(self.agents)
        self.max_agent_obs_size = max(
            o.size 
            for a,o in self.agent_obs_mapping.items()
            if not a == "global"
        )
        obs_sizes = {
            agent: (
                # TODO move the global obs out of here and treat manually. It's wrong atm, and shouldn't have the extra num_agents indicator
                self.num_agents + (
                    self.max_agent_obs_size if homogenisation_method == "max"
                    else self.env.observation_size if homogenisation_method == "concat"
                    else obs.size
                )
            )
            for agent, obs in self.agent_obs_mapping.items()
            if not agent == "global"
        }
        obs_sizes["global"] = self.agent_obs_mapping["global"].size
        act_sizes = {
            agent: max([a.size for a in self.agent_action_mapping.values()])
            if homogenisation_method == "max"
            else self.env.action_size
            if homogenisation_method == "concat"
            else act.size
            for agent, act in self.agent_action_mapping.items()
        }

        self.observation_spaces = {
            agent: spaces.Box(
                -jnp.inf,
                jnp.inf,
                shape=(obs_size,),
            )
            for agent, obs_size in obs_sizes.items()
        }
        self.action_spaces = {
            agent: spaces.Box(
                -1.0,
                1.0,
                shape=(act_sizes[agent],),
            )
            for agent in self.agents
        }

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], envs.State]:
        state = self.env.reset(key)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: envs.State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[
        Dict[str, chex.Array], envs.State, Dict[str, float], Dict[str, bool], Dict
    ]:
        global_action = self.map_agents_to_global_action(actions)
        next_state = self.env.step(key, state, global_action)  # type: ignore
        observations = self.get_obs(next_state)
        # rewards = {agent: next_state.reward for agent in self.agents}
        # rewards["__all__"] = next_state.reward
        rewards = {agent: next_state.reward[i] if self.het_reward else next_state.reward for i, agent in enumerate(self.agents)}
        rewards["__all__"] = jnp.mean(next_state.reward) if self.het_reward else next_state.reward # added this for het rewards but maybe it would be best
        dones = {agent: next_state.done.astype(jnp.bool_) for agent in self.agents}
        dones["__all__"] = next_state.done.astype(jnp.bool_)
        return (
            observations,
            next_state,  # type: ignore
            rewards,
            dones,
            next_state.info,
        )

    @partial(jax.jit, static_argnums=(0,))
    def get_avail_actions(self, state: envs.State) -> Dict[str, chex.Array]:
        return {a: jnp.ones(self.action_spaces[a].shape, dtype=jnp.uint8) for a in self.agents}

    def get_obs(self, state: envs.State) -> Dict[str, chex.Array]:
        """Extracts agent observations from the global state.

        Args:
            state: Global state of the environment.

        Returns:
            A dictionary of observations for each agent.
        """
        return self.map_global_obs_to_agents(state.obs)

    def map_agents_to_global_action(
        self, agent_actions: Dict[str, jnp.ndarray]
    ) -> jnp.ndarray:
        global_action = jnp.zeros(self.env.action_size)
        for agent_name, action_indices in self.agent_action_mapping.items():
            if self.homogenisation_method == "max":
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name][: action_indices.size]
                )
            elif self.homogenisation_method == "concat":
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name][action_indices]
                )
            else:
                global_action = global_action.at[action_indices].set(
                    agent_actions[agent_name]
                )
        return global_action

    def map_global_obs_to_agents(
        self, global_obs: jnp.ndarray
    ) -> Dict[str, jnp.ndarray]:
        """Maps the global observation vector to the individual agent observations.

        Args:
            global_obs: The global observation vector.

        Returns:
            A dictionary mapping agent names to their observations. The mapping method
            is determined by the homogenisation_method parameter.
        """
        agent_obs = {
            "global": global_obs[self.agent_obs_mapping["global"]]
        }
        for agent_idx, agent_name in enumerate(self.agents):
            obs_indices = self.agent_obs_mapping[agent_name]
            if self.homogenisation_method == "max":
                # Vector with the agent idx one-hot encoded as the first num_agents
                # elements and then the agent's own observations (zero padded to
                # the size of the largest agent observation vector)
                agent_obs[agent_name] = (
                    jnp.zeros(
                        self.num_agents + self.max_agent_obs_size
                    )
                    .at[agent_idx]
                    .set(1)
                    .at[agent_idx + 1 : agent_idx + 1 + obs_indices.size]
                    .set(global_obs[obs_indices])
                )
            elif self.homogenisation_method == "concat":
                # Zero vector except for the agent's own observations
                # (size of the global observation vector)
                agent_obs[agent_name] = (
                    jnp.zeros(global_obs.shape)
                    .at[obs_indices]
                    .set(global_obs[obs_indices])
                )
            else:
                # Just agent's own observations
                agent_obs[agent_name] = global_obs[obs_indices]
        return agent_obs

    @property
    def sys(self):
        return self.env.sys



class ScratchItch(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("scratchitch", **kwargs)

class BedBathing(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("bedbathing", **kwargs)

class ArmManipulation(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("armmanipulation", **kwargs)

class PushCoop(MABraxEnv):
    def __init__(self, **kwargs):
        super().__init__("pushcoop", **kwargs)