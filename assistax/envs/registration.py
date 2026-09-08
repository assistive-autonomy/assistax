# Copyright 2024 The JAXMARL Authors, under the Apache License, Version 2.0.
# Copyright 2025 The Assistax Authors.

"""Make method for Assisatax"""

from assistax.envs.base_env import (
    ScratchItch,
    BedBathing,
    ArmManipulation,
    PushCoop,
    CooperativeHandover,
    Feeding,
    TeethBrushing,
)

def make(env_id: str, **env_kwargs):
    """Create an assistax multi-agent environment by name.

    This is the main entry point for using the environments. It returns a
    JaxMARL-style multi-agent environment with a functional (JAX) API::

        import jax, jax.numpy as jnp, assistax
        env = assistax.make("scratchitch")
        obs, state = env.reset(jax.random.PRNGKey(0))
        actions = {a: jnp.zeros(env.action_spaces[a].shape) for a in env.agents}
        obs, state, rewards, dones, info = env.step(jax.random.PRNGKey(1), state, actions)

    ``obs``, ``rewards`` and ``dones`` are dicts keyed by agent name (``rewards``
    and ``dones`` also contain an ``"__all__"`` entry). Actions are ``Box(-1, 1)``
    per actuator.

    Args:
        env_id: One of ``registered_envs`` (e.g. ``"scratchitch"``, ``"feeding"``).
        **env_kwargs: Passed to the environment constructor / ``assistax.envs.create``,
            e.g. ``backend`` ("mjx"), ``episode_length``, ``het_reward``,
            ``disability``, ``preference_rewards``, ``sparse_rewards``.

    Returns:
        A multi-agent environment exposing ``reset``/``step``, ``agents``,
        ``observation_spaces`` and ``action_spaces``.

    Raises:
        ValueError: If ``env_id`` is not in ``registered_envs``.
    """
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")
    if env_id == "scratchitch":
        env = ScratchItch(**env_kwargs)
    elif env_id == "bedbathing":
        env = BedBathing(**env_kwargs)
    elif env_id == "armmanipulation":
        env = ArmManipulation(**env_kwargs)
    elif env_id == "pushcoop":
        env = PushCoop(**env_kwargs)    
    elif env_id == "handover":
        env = CooperativeHandover(**env_kwargs)
    elif env_id == "feeding":
        env = Feeding(**env_kwargs)
    elif env_id == "teethbrushing":
        env = TeethBrushing(**env_kwargs)

    return env
   
registered_envs = [
    "scratchitch",
    "bedbathing",
    "armmanipulation",
    "pushcoop",
    "handover",
    "feeding",
    "teethbrushing",
]