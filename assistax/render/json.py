# Copyright 2024 The Brax Authors.
#Copyright 2025 The Assistax Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import
"""Saves a system config and trajectory as json."""

import json
from typing import List, Text, Tuple

from brax.base import State, System
from etils import epath
import jax
import jax.numpy as jp
import mujoco
import numpy as np


# State attributes needed for the visualizer.
_STATE_ATTR = ["x", "contact"]

# Fields that get json encoded.
_ENCODE_FIELDS = [
    "contact",
    "opt",
    "timestep",
    "face",
    "size",
    "link_idx",
    "link_names",
    "name",
    "dist",
    "pos",
    "rgba",
    "rot",
    "states",
    "transform",
    "vert",
    "x",
    "xd",
]

_GEOM_TYPE_NAMES = {
    0: "Plane",
    1: "HeightMap",
    2: "Sphere",
    3: "Capsule",
    5: "Cylinder",
    6: "Box",
    7: "Mesh",
}


def _to_dict(obj):
    """Converts python object to a json encodeable object."""
    if isinstance(obj, list) or isinstance(obj, tuple):
        return [_to_dict(s) for s in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items() if k in _ENCODE_FIELDS}
    if isinstance(obj, jax.Array):
        return _to_dict(obj.tolist())
    if hasattr(obj, "__dict__"):
        d = dict(obj.__dict__)
        d["name"] = obj.__class__.__name__
        return _to_dict(d)
    if isinstance(obj, np.ndarray):
        return _to_dict(obj.tolist())
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return str(obj)
    return obj


def _compress_contact(states: State) -> State:
    """Reduces the number of contacts based on penetration > 0."""
    if states.contact is None or states.contact.pos.shape[0] == 0:
        return states

    contact_mask = states.contact.dist < 0
    n_contact = contact_mask.sum(axis=1).max()

    def pad(arr, n):
        r = jp.zeros(n)
        if len(arr.shape) > 1:
            r = jp.zeros((n, *arr.shape[1:]))
        r = r.at[: arr.shape[0], ...].set(arr)
        return r

    def compress(contact, i):
        return jax.tree.map(
            lambda x: pad(x[contact_mask[i]], n_contact), contact.take(i)
        )

    c = [compress(states.contact, i) for i in range(states.x.pos.shape[0])]
    return states.replace(contact=jax.tree.map(lambda *x: jp.stack(x), *c))


def _get_mesh(mj: mujoco.MjModel, i: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gets mesh from mj at index i."""
    last = (i + 1) >= mj.nmesh
    face_start = mj.mesh_faceadr[i]
    face_end = mj.mesh_faceadr[i + 1] if not last else mj.mesh_face.shape[0]
    face = mj.mesh_face[face_start:face_end]

    vert_start = mj.mesh_vertadr[i]
    vert_end = mj.mesh_vertadr[i + 1] if not last else mj.mesh_vert.shape[0]
    vert = mj.mesh_vert[vert_start:vert_end]

    return vert, face


def dumps(sys: System, states: List[State]) -> Text:
    """Creates a json string of the system config.

    Args:
      sys: brax System object
      states: list of brax system states

    Returns:
      string containing json dump of system and states

    Raises:
      RuntimeError: if states have invalid shape
    """
    if any((len(s.x.pos.shape), len(s.x.rot.shape)) != (2, 2) for s in states):
        pos_shape = max(len(s.x.pos.shape) for s in states)
        rot_shape = max(len(s.x.rot.shape) for s in states)
        raise RuntimeError(
            "Expected state.x position and rotation to have 2 shape dimensions but "
            f"received len(pos.shape)={pos_shape} and len(rot.shape)={rot_shape}"
        )

    d = _to_dict(sys)

    # TODO: move the manipulations below to javascript

    # fill in empty link names
    link_names = [n or f"link {i}" for i, n in enumerate(sys.link_names)]
    link_names += ["world"]

    # unpack geoms into a dict for the visualizer
    link_geoms = {}
    for id_ in range(sys.ngeom):
        link_idx = sys.geom_bodyid[id_] - 1

        rgba = sys.geom_rgba[id_]
        if (rgba == jp.array([0.5, 0.5, 0.5, 1.0])).all(): # changed for rendering
            # convert the default mjcf color to brax default color
            rgba = np.array([0.4, 0.33, 0.26, 1.0])

        geom = {
            "name": _GEOM_TYPE_NAMES[sys.geom_type[id_]],
            "link_idx": link_idx,
            "pos": sys.geom_pos[id_],
            "rot": sys.geom_quat[id_],
            "rgba": rgba,
            "size": sys.geom_size[id_],
        }

        if geom["name"] == "Mesh":
            vert, face = _get_mesh(sys.mj_model, sys.geom_dataid[id_])
            geom["vert"] = vert
            geom["face"] = face

        link_geoms.setdefault(link_names[link_idx], []).append(_to_dict(geom))

    d["geoms"] = link_geoms

    # stack states for the viewer
    states = jax.tree.map(lambda *x: jp.stack(x), *states)
    states = _compress_contact(states)

    states = _to_dict(states)
    d["states"] = {k: states[k] for k in _STATE_ATTR}

    return json.dumps(d)


def save(path: str, sys: System, states: List[State]):
    """Saves a system config and trajectory as json."""
    with epath.Path(path).open("w") as fout:
        fout.write(dumps(sys, states))
