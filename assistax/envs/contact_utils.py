"""Map MJX contact-slot ids to geom names, so hardcoded contact ids can be checked by name.

MJX gives each model a fixed contact layout: slot ``i`` always holds the same geom pair, whether or not the
geoms touch. The envs read contact forces by slot id, so an id silently changes meaning when collision pairs
are added or removed in a scene XML, or when a MuJoCo/MJX upgrade changes the layout.
``tests/test_contacts.py`` pins every id; use this module to find the right ones.

Command line::

    python -m assistax.envs.contact_utils feeding                         # every contact id of the env -> geom pair
    python -m assistax.envs.contact_utils feeding --pair head spoon_bowl  # all slots for a geom pair
"""
import argparse

import jax
import mujoco
import numpy as np
from brax.envs.base import Env
from mujoco import mjx


def contact_geom_pairs(model: mujoco.MjModel) -> np.ndarray:
    """Geom-id pair held by every MJX contact slot, shape ``(ncon, 2)``; fixed for a model and MuJoCo version."""
    mx = mjx.put_model(model)
    data = jax.jit(mjx.forward)(mx, mjx.make_data(mx))
    contact = data._impl.contact if hasattr(data, "_impl") else data.contact
    return np.asarray(contact.geom)


def find_contact_ids(model: mujoco.MjModel, pairs: np.ndarray, geom_a: str, geom_b: str) -> list[int]:
    """All contact slots between two named geoms, in either order."""
    a, b = model.geom(geom_a).id, model.geom(geom_b).id
    hit = ((pairs[:, 0] == a) & (pairs[:, 1] == b)) | ((pairs[:, 0] == b) & (pairs[:, 1] == a))
    return [int(i) for i in np.flatnonzero(hit)]


def env_contact_ids(env: Env) -> dict[str, list[int]]:
    """Every contact-slot attribute of an env: integer (or integer-list) attributes with "contact" in the name."""
    out = {}
    for name, value in vars(env.unwrapped).items():
        if "contact" not in name.lower() or isinstance(value, bool):
            continue
        values = np.ravel(np.asarray(value, dtype=object)) if isinstance(value, (list, tuple, np.ndarray)) else [value]
        if len(values) and all(isinstance(v, (int, np.integer)) and not isinstance(v, bool) for v in values):
            out[name] = [int(v) for v in values]
    return out


def main() -> None:
    """Print the geom pair behind each of an env's contact ids, or all slots for one geom pair."""
    from assistax import envs  # local import: keeps this module importable without the env registry

    parser = argparse.ArgumentParser(description="Map MJX contact-slot ids to geom names.")
    parser.add_argument("env", help="env name, e.g. feeding")
    parser.add_argument("--pair", nargs=2, metavar=("GEOM_A", "GEOM_B"), help="list all slots for this geom pair")
    args = parser.parse_args()
    env = envs.create(args.env, episode_length=None, auto_reset=False)
    m = env.sys.mj_model
    pairs = contact_geom_pairs(m)
    print(f"{args.env}: {len(pairs)} contact slots (mujoco {mujoco.__version__})")
    if args.pair:
        print(f"slots for {tuple(args.pair)}: {find_contact_ids(m, pairs, *args.pair)}")
        return
    for attr, slots in env_contact_ids(env).items():
        for slot in slots:
            print(f"  {attr}: slot {slot} -> ({m.geom(int(pairs[slot, 0])).name}, {m.geom(int(pairs[slot, 1])).name})")


if __name__ == "__main__":
    main()
