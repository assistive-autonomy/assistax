# Copyright 2024 The JAXMARL Authors, under the Apache License, Version 2.0.
# Copyright 2025 The Assistax Authors.

"""Make method for Assisatax"""

from assistax.envs.base_env import (
    ScratchItch,
    BedBathing,
    ArmManipulation,
    PushCoop,
)

def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
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

    return env
   
registered_envs = [
    "scratchitch",
    "bedbathing",
    "armmanipulation",
    "pushcoop",
]