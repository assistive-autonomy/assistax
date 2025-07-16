import numpy as np
import jax.numpy as jnp
import pickle

mesh_idx = 0
keys = ["vertices", "vertex_normals", "faces", "face_normals", "face_angles"]
keys = ["vertex_normals"]

numpy_arrays = {}
jax_arrays = {}


for mesh_idx in range(57):
    for k in keys:
        with open(
            f"/disk/scratch1/tmcinroe/data_assistax/mesh_{mesh_idx}_{k}_numpy.data",
            "rb",
        ) as f:
            numpy_data = pickle.load(f)

        with open(
            f"/disk/scratch1/tmcinroe/data_assistax/mesh_{mesh_idx}_{k}_jax.data", "rb"
        ) as f:
            jax_data = pickle.load(f)

        try:
            if not np.allclose(numpy_data, jax_data):
                print(f"{mesh_idx}: {k}")
                print(f"\tnp: {numpy_data.shape}, jnp: {jax_data.shape}")
                print(f"\tnp: {numpy_data.sum()}")
                print(f"\tjnp: {jax_data.sum()}")
        except:
            print(f"{mesh_idx}: {k}")
            print(f"\tcould not check")
            print(f"\tnp: {numpy_data.shape}, jnp: {jax_data.shape}")
            # print(f"\tnp: {numpy_data}")
            # print(f"\tjnp: {jax_data}")
