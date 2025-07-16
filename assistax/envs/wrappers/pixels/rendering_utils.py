"""
Utilities for rendering pixel observations directly on an XLA device
"""

from typing import Optional, Dict, NamedTuple, Iterable, Any

from jax._src.api import vmap
from brax import base, math
from mujoco.mjx._src.types import GeomType
import brax
import flax
import jax.numpy as jnp
import jax
import numpy as onp
from functools import partial


from .renderer import CameraParameters as Camera
from .renderer import LightParameters as Light
from .renderer import Model as RendererMesh
from .renderer import ModelObject as Instance
from .renderer import ShadowParameters as Shadow
from .renderer import (
    Renderer,
    UpAxis,
    create_capsule,
    create_cube,
    transpose_for_display,
)

# import trimesh
from .trimesh_jax import (
    Trimesh,
    compute_face_normals_and_triangles,
    compute_face_angles,
    compute_vertex_normals,
)


from .geom_primitives import Capsule, Box, Sphere, Plane, Convex, Mesh


def quat_from_3x3(m: jax.Array) -> jax.Array:
    """Converts 3x3 rotation matrix to quaternion."""
    w = jnp.sqrt(1 + m[0, 0] + m[1, 1] + m[2, 2]) / 2.0
    x = (m[2][1] - m[1][2]) / (w * 4)
    y = (m[0][2] - m[2][0]) / (w * 4)
    z = (m[1][0] - m[0][1]) / (w * 4)
    return jnp.array([w, x, y, z])


@flax.struct.dataclass
class PixelState(base.Base):
    pipeline_state: Optional[base.State]
    obs: jnp.ndarray
    pixels: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    #key: Optional[jnp.ndarray]
    metrics: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)
    info: Dict[str, jnp.ndarray] = flax.struct.field(default_factory=dict)


@partial(jax.jit, static_argnames="hw")
def render_pixels(sys: brax.System, pipeline_states: brax.State, hw: int):
    # (1) grab the cameras and the view targets. The camera object contains its own view target
    # the extra bit we grab with _get_targets() is only used to render shadows. Maybe we can remove?
    # print(f'Within render pixels')

    # The "current_frame" arg is meant to work for the "video distractor" case
    batched_camera = _get_cameras(pipeline_states, hw, hw)
    # print(f'after batched_camera')

    print(f"got the batched_camera object....")
    batched_target = _get_targets(pipeline_states)

    print("got the batched_target...")
    # print(f'after batched_target')
    objs = _build_objects(sys, pipeline_states)
    print("finally built those god damn objects...")
    # print(f'after _build_objects')
    images = _render(objs, pipeline_states, batched_camera, batched_target, hw)
    print(f"done with _render()")
    print(f"images: {images.shape} // {images.dtype}")
    # print(f'after _render')
    # return None
    return images


# Perhaps we do not want to jit this function as it will only be run once. This should
# save on compile-time overhead
def build_objects_for_cache(sys: brax.System, n_envs: int):
    objs = _build_objects(sys, jnp.zeros((n_envs, 1)))

    # we now have a list of Obj(), but they are not tracedarrays
    jax_objs = []
    for obj in objs:
        #print(f"bool: {obj.instance.double_sided}")
        #print(obj.instance.double_sided.shape)
        #print(len(obj.instance.double_sided.shape) > 0)
        obj = Obj(link_idx=obj.link_idx, off=obj.off, rot=obj.rot, instance=obj.instance._replace(double_sided=jnp.array(obj.instance.double_sided).reshape(1,)))
        #obj = obj.replace(instance=obj.instance._replace(double_sided=jnp.array(obj.instance.double_sided).reshape(1,)))
        #obj.instance = obj.instance._replace(double_sided = jnp.array(obj.instance.double_sided).reshape(1,))
        jax_objs.append(jax.tree_map(lambda x: jnp.array(x), obj))
        #print(f"bool: {obj.instance.double_sided}")
        #print(obj.instance.double_sided.shape)
        #qqq
        # try:
        #    print(
        #        f"{jax_objs[-1].instance.model.verts.shape} // {jax_objs[-1].instance.model.faces.shape}"
        #    )
        # except:
        #    print("This one has no verts...")
        #    print(jax_objs[-1])
        #    qqq
    
    vmappable_objs = Obj(
        #instance=jax.tree_map(lambda *x: jnp.concatenate([jnp.expand_dims(_x, 0) for _x in x], axis=0), *zip([i.instance for i in jax_objs]))[0],
        rot=jnp.concatenate([x.rot[None] for x in jax_objs], axis=0),
        off=jnp.concatenate([x.off[None] for x in jax_objs], axis=0),
        link_idx=jnp.concatenate(
            [jnp.array(x.link_idx)[None] for x in jax_objs], axis=0
        ),
    )
    #print(f"new instances: {vmappable_objs.instance.transform.shape}")
    #qqq

    return jax_objs, vmappable_objs


@partial(jax.jit, static_argnames="hw")
def render_pixels_with_cached_objs(
    pipeline_states: brax.State,
    cached_objs: Iterable[Any],
    cached_vmappable_objs: Iterable[Any],
    hw: int,
):
    # Here, cached_objs has .instances attribute while vmappable does not
    batched_camera = _get_cameras(pipeline_states, hw, hw)
    batched_target = _get_targets(pipeline_states)
    
    images = _render_cached(
        cached_objs,
        cached_vmappable_objs,
        pipeline_states,
        batched_camera,
        batched_target,
        hw,
    )
    return images


def get_camera(
    state: brax.State,
    width: int,
    height: int,
) -> Camera:
    """Gets camera object."""
    eye, up = _eye(state), _up()

    hfov = HFOV  # orig: 58.0 -- higher == more zoomed out
    vfov = hfov * height / width

    # Position of the camera && target need to be updated
    # _eye determines the position
    # get_target updates the "lookat" target
    target = get_target(state)

    camera = Camera(
        viewWidth=width,
        viewHeight=height,
        position=eye,
        target=target,
        up=up,
        hfov=hfov,
        vfov=vfov,
    )

    return camera


# _get_cameras = jax.jit(get_camera)


# The groundplane used in the environment. This component is important for the locomoation
# tasks, as the agent needs to understand the contact between the plane and the rigid body
def grid(grid_size: int, color) -> jnp.ndarray:
    # These need to be numpy arrays because jax arrays are immutable
    grid = onp.zeros((grid_size, grid_size, 3), dtype=onp.single)
    grid[:, :] = onp.array(color) / 255.0
    grid[0] = onp.zeros((grid_size, 3), dtype=onp.single)
    # to reverse texture along y direction
    grid[:, -1] = onp.zeros((grid_size, 3), dtype=onp.single)
    return jnp.asarray(grid)


hw = 128
_GROUND: jnp.ndarray = grid(hw, [200, 200, 200])

CAMERA_TARGET = 0
CAM_EYE = 0
# Alter the offset to alter the position of the camera relative to CAM_EYE geom
CAM_OFF = jnp.array([1.7, 1.7, 1.5])
CAM_UP = jnp.array([0.0, 1.0, 0.2])
# The purpose of CAM_Z is to have a constant location along the z-axis for the camera
# This is useful if the goems may move up/down (e.g., locomotion envs)
CAM_Z = 3.5
HFOV = 40.0


def _eye(state: brax.State) -> jnp.ndarray:
    """
    Determines the camera location for a Brax system as a position relative to a given geom in the brax System.
    This is computed as an  "offset" from the location of the geom.
    Args:
      sys:
      state:

    Returns:

    """
    """"""
    print("inside _eye()...")
    # there are sys.mj_model.nbody - 1 (ignores the plane?)
    print(f"state: {state.x.pos.shape}")
    print(f"geom: {state.geom_xpos.shape}")
    # [x, y, z]
    cam_eye = state.x.pos[CAM_EYE, :] + CAM_OFF
    return cam_eye

    # if env_name == 'reacher':
    #  # The geom we are attaching the camera to does not move, so we don't need anything special
    #  return state.x.pos[CAM_EYE, :] + CAM_OFF
    # elif env_name in ['halfcheetah', 'ant', 'walker2d', 'pusher', 'swimmer', 'hopper'] or 'humanoid' in env_name:
    #  # All the geoms the camera can attach to are going to be moving.
    #  # We want the camera to be steady. This can be done with idx 2 of <x,y,z>
    #  cam_eye = state.x.pos[CAM_EYE, :] + CAM_OFF
    #  return cam_eye.at[2].set(CAM_Z)
    # elif env_name == 'inverted_pendulum':
    #  return CAM_OFF


def _up() -> jnp.ndarray:
    """Determines the up orientation of the camera."""
    # [0,1,1] [1,1,1] weird angle isometric
    # return jnp.array([0., 0., 0.])
    # return jnp.array([0., 0., 1.])
    return CAM_UP


def get_target(state: brax.State) -> jnp.ndarray:
    """Gets target of camera. I.e., the center of the camera's viewport"""
    # This function is within the vmap context of _get_cameras
    return jnp.array(
        [state.x.pos[CAMERA_TARGET, 0], state.x.pos[CAMERA_TARGET, 1], 0.6]
    )
    # if env_name in ["reacher", "ant", "pusher", "swimmer"]:
    #    return jnp.array(
    #        [state.x.pos[CAMERA_TARGET, 0], state.x.pos[CAMERA_TARGET, 1], 0]
    #    )
    # elif env_name in ["halfcheetah", "walker2d", "hopper"]:
    #    return jnp.array(
    #        [state.x.pos[CAMERA_TARGET, 0], state.x.pos[CAMERA_TARGET, 1], 0.6]
    #    )
    # elif "humanoid" in env_name:
    #    return jnp.array(
    #        [state.x.pos[CAMERA_TARGET, 0], state.x.pos[CAMERA_TARGET, 1], 1.1]
    #    )
    # elif env_name == "inverted_pendulum":
    #    return jnp.array([0.0, 0.0, 0.0])


# TODO; make every grom primitive congruent. That way we can vmap over the objects instead of looping over
#  them. This could grant us a large speedup.
class Obj(NamedTuple):
    """An object to be rendered in the scene.

    Assume the system is unchanged throughout the rendering.

    col is accessed from the batched geoms `sys.geoms`, representing one geom.
    """

    """An instance to be rendered in the scene, defined by jaxrenderer."""
    link_idx: int
    """col.link_idx if col.link_idx is not None else -1"""
    off: jnp.ndarray
    """col.transform.rot"""
    rot: jnp.ndarray
    """col.transform.rot"""
    instance: Optional[Instance] = None


@partial(jax.vmap, in_axes=(None, 0, None, None, None, None))
# @partial(jax.jit, static_argnames=("geom_id", "geom_num", "body_id"))
@partial(jax.jit, static_argnames=("geom_id", "geom_num"))
def _vmap_build(
    sys: brax.System,
    pipeline_states,
    specular_map: jnp.ndarray,
    tex: jnp.ndarray,
    geom_id: int,
    geom_num: int,
):
    """ """
    # Plane
    if geom_id == 0:
        model = create_cube(
            half_extents=jnp.array([1000.0, 1000.0, 0.0001]),
            texture_scaling=jnp.array(8192.0),
            diffuse_map=_GROUND,
            specular_map=specular_map,
        )
    # HFIELD
    elif geom_id == 1:
        raise NotImplementedError("Have not impl HFIELD yet.")
    # Sphere
    elif geom_id == 2:
        # sphere geom_rbound = geom_size[0]
        print(f"======== SPHERE =========")
        print(f"geom_rbound: {sys.mj_model.geom_rbound[geom_num]}")
        print(f"geom_size: {sys.mj_model.geom_size[geom_num]}")
        print("==========================")
        # is sys.mj_model.geom_rbound == sys.mj_model.geom_size[0]?
        radius = sys.mj_model.geom_rbound[geom_num]
        model = create_capsule(
            radius=radius,
            half_height=jnp.array(0.0),
            up_axis=UpAxis.Z,
            diffuse_map=tex,
            specular_map=specular_map,
        )
    # Capsule
    elif geom_id == 3:
        # capsule geom_rbound = geom_size[0] + geom_size[1]
        # capsule geom_size[0] and geom_size[1] is not always same
        # capsule geom_size[2] is always 0
        print(f"======== CAPSULE =========")
        print(f"geom_rbound: {sys.mj_model.geom_rbound[geom_num]}")
        print(f"geom_size: {sys.mj_model.geom_size[geom_num]}")
        print("==========================")

        # geom_rbound is the radius of the "bounding sphere". This means the half_height
        # should just be the radius, right? If so, then what is radius?
        bs_radius = sys.mj_model.geom_rbound[geom_num]
        # length = sys.mj_model.geom_l

        # When using "fromto", only need to provide a single number for size: the
        # radius of the object. in panda.xml, 0.04 or 0.04
        # sys.mj_model.geom_size[geom_num] = [0.07 0.07 0.  ]

        # The "half_height" determines the length of the cylinder between the two
        # half-spheres. I.g., half_height=0 is just a sphere with a given radius
        # geom_size[0] is radius
        # geom_size[1] * 2 is height in THREE.CylinderGeometry
        model = create_capsule(
            radius=sys.mj_model.geom_size[geom_num][0],
            half_height=sys.mj_model.geom_size[geom_num][1],
            up_axis=UpAxis.Z,
            diffuse_map=tex,
            specular_map=specular_map,
        )
    elif geom_id == 4:
        raise NotImplementedError("Have not implemented ELLIPSOID yet.")
    elif geom_id == 5:
        raise NotImplementedError("Have no implemented CYLINDER yet.")
    elif geom_id == 6:
        model = create_cube(
            half_extents=sys.geom_size[geom_num] * 0.0,
            diffuse_map=tex,
            texture_scaling=jnp.array(16.0),
            specular_map=specular_map,
        )
    elif geom_id == 7:
        # nmeshvert: 399342
        # geom_dataid: (106,)
        # mesh_vertadr: (57,)
        # Get vertices:
        # (1) get the idx at which this particular mesh's idx begins and ends
        # (2) use (1) to query mj_model.mesh_vert[begin:end]

        # we can use "geom_dataid: id of geom's mesh/hfield" to determine the mesh idx
        # as well as if the mesh is the last mesh. geom_dataid: (n_geoms,)
        mesh_idx = sys.mj_model.geom_dataid[geom_num]
        last_mesh = (mesh_idx + 1) >= sys.mj_model.nmesh
        vert_idx_start = sys.mj_model.mesh_vertadr[mesh_idx]
        vert_idx_end = (
            sys.mj_model.mesh_vertadr[mesh_idx + 1]
            if not last_mesh
            else sys.mj_model.mesh_vert.shape[0]
        )
        # mesh_vert (399342, 3)
        vertices = sys.mesh_vert[vert_idx_start:vert_idx_end]

        # Get faces
        face_idx_start = sys.mj_model.mesh_faceadr[mesh_idx]
        face_idx_end = (
            sys.mj_model.mesh_faceadr[mesh_idx + 1]
            if not last_mesh
            else sys.mj_model.mesh_face.shape[0]
        )
        faces = sys.mj_model.mesh_face[face_idx_start:face_idx_end]

        # print(f"vertices: {vertices.shape}")
        # print(f"faces: {faces.shape}")
        # print(f"BEFORE: {tex.shape}")
        material_id = sys.mj_model.geom_matid[geom_num]
        tex = sys.mat_rgba[material_id][:3].reshape((1, 1, 3))
        # print(f"AFTER: {tex.shape}")

        # I am now beginning to think that we do not need **any** of the dataclasses...
        # We cannot jit the creation of this dataclass, which I think is slowing down
        # the runtime (bottlenecking it, effectively).
        face_normals, _triangles = compute_face_normals_and_triangles(vertices, faces)
        face_angles = compute_face_angles(_triangles)
        vertex_normals = compute_vertex_normals(
            vertices, faces, face_normals, face_angles
        )

        # tm = Trimesh.create(vertices, faces)
        # print("made the object.")
        ## We need the vertex_normals, which is computed with geometry.weighted_vertex_normals()
        ## The above fn takes vertex_count (len(trimesh.vertices)),
        ## face_normals (needs computing)
        ## face_angles (needs computing)

        ## Face normals:
        ## (a) triangles = vertices[faces]
        ## (b) triangles_cross = triangles.cross(triangles)

        # face_normals, _triangles = tm.compute_face_normals_and_triangles()
        # face_angles = tm.compute_face_angles(_triangles)
        # vertex_normals = tm.compute_vertex_normals(face_normals, face_angles)
        model = RendererMesh.create(
            verts=vertices,
            norms=vertex_normals,
            uvs=jnp.zeros((vertices.shape[0], 2), dtype=int),
            faces=faces,
            diffuse_map=tex,
        )
    else:
        raise NotImplementedError(f"Geom of ID {geom_id} not implemented nor known.")

    # Lets give this a funny shout... Trying to copy how they do it in the .js file
    # https://github.com/google/brax/blob/main/brax/visualizer/js/system.js#L223
    # x *does not include the ground plane*! This would explain the n_bodies - 1 (29, 3)
    # https://github.com/google/brax/blob/c87dcfc5094afffb149f98e48903fb39c2b7f7af/brax/mjx/pipeline.py#L75C17-L75C34

    # x.rot, x.pos = (29, 3), (29, 4)
    # print(
    #    f".x.rot: {pipeline_states.x.rot.shape} // {pipeline_states.x.rot[body_id - 1].shape}"
    # )
    # print(
    #    f".x.pos: {pipeline_states.x.pos.shape} // {pipeline_states.x.pos[body_id - 1].shape}"
    # )

    ## (106, 3), (106, 4)
    # print(f"sys.geom_pos: {sys.mj_model.geom_pos.shape}")
    # print(f"sys.geom_quat: {sys.mj_model.geom_quat.shape}")

    # rot_raw = pipeline_states.x.rot[body_id]
    # rot = jnp.array([rot_raw[1], rot_raw[2], rot_raw[3], rot_raw[0]])
    # off = pipeline_states.x.pos[body_id]
    # rot = quat_from_3x3(math.inv_3x3(pipeline_states.geom_xmat[geom_num]))

    # The groundplane's information is **not** within pipeline_states.x
    # if geom_id != 990:
    # rot = quat_from_3x3(pipeline_states.geom_xmat[geom_num])
    # off = pipeline_states.geom_xpos[geom_num]
    # copying this thing:
    # https://github.com/google/brax/blob/main/brax/io/json.py#L129
    rot = sys.geom_quat[geom_num]
    off = sys.geom_pos[geom_num]
    # print(f"rot: {rot.shape}")
    # print(f"off: {off.shape}")

    # Then there's this idea...
    # https://github.com/google/brax/blob/c87dcfc5094afffb149f98e48903fb39c2b7f7af/brax/contact.py#L43
    # else:
    #    # off = pipeline_states.geom_xpos[geom_num]
    #    # rot = math.ang_to_quat(pipeline_states.xd.ang[body_id - 1])

    #    def local_to_global(pos1, quat1, pos2, quat2):
    #        pos = pos1 + math.rotate(pos2, quat1)
    #        mat = math.quat_to_3x3(math.quat_mul(quat1, quat2))
    #        return pos, mat

    #    x = pipeline_states.x.concatenate(base.Transform.zero((1,)))
    #    pos, mat = local_to_global(
    #        x.pos[body_id - 1],
    #        x.rot[body_id - 1],
    #        sys.mj_model.geom_pos[geom_num],
    #        sys.mj_model.geom_quat[geom_num],
    #    )
    #    print(f"pos: {pos.shape}")
    #    print(f"3x3: {mat.shape}")

    #    off = pos
    #    rot = quat_from_3x3(mat)
    return model, rot, off


@jax.jit
def _build_objects(sys: brax.System, pipeline_states: brax.State) -> list[Obj]:
    """
    Converts a brax System to a list of Obj.

    Args:
      sys:

    Returns:

    """

    objs: list[Obj] = []

    def take_i(obj, i):
        return jax.tree_map(lambda x: jnp.take(x, i, axis=0), obj)

    # Loop through each geom type (sys.mj_model.geom_type) in the list of the
    # environment's geoms. Within each step in the loop, loop over each of the
    # batched envs, create the N geoms (N = num of parallel envs) in a list
    # final outer list is len of ngeom, and len of each inner list is len of
    # N.

    # print(f"pipeline_states: {type(pipeline_states)}")
    # print(f"n envs: {pipeline_states.geom_xpos.shape}")
    # print(f"links: {type(pipeline_states.contact)}")
    # print(
    #    f"idxs: {pipeline_states.contact.link_idx[0].shape} // {pipeline_states.contact.link_idx[1].shape}"
    # )

    # print(
    #    f"type: {type(pipeline_states.contact.link_idx[0])} // {type(pipeline_states.contact.link_idx[1])}"
    # )

    # print(f"geom_bodyid: {sys.geom_bodyid}")
    # print(f"dof_parentid: {sys.dof_parentid}")
    # qqq

    ## Can we use the idx from sys.geom_bodyid to query sys.dof_parentid?

    # link_idxs = [x for x in pipeline_states.contact.link_idx[1]]
    # print(f"link_idxs: {len(link_idxs)}")

    # print(f"dof_parentid: {sys.dof_parentid} // {sys.dof_parentid.shape}")
    print(f"bbox: {sys.geom_size} // {sys.geom_size.shape}")
    # qqq
    # cuboid.verts
    for idx, geom_id in enumerate(sys.mj_model.geom_type):
        print(f"geom_id: {geom_id}")
        tex = sys.mj_model.geom_rgba[idx, :3].reshape((1, 1, 3))
        # reference: https://github.com/erwincoumans/tinyrenderer/blob/89e8adafb35ecf5134e7b17b71b0f825939dc6d9/model.cpp#L215
        specular_map = jax.lax.full(tex.shape[:2], 2.0)

        # Can we use the idx from sys.geom_bodyid to query sys.body_parentid?
        # link_idx = sys.body_parentid[sys.geom_bodyid[idx] - 1]

        # copying this thing:
        # https://github.com/google/brax/blob/main/brax/io/json.py#L129
        link_idx = sys.geom_bodyid[idx] - 1

        # TODO: temporary for dev. remove when done
        #if geom_id in [0, 1, 2, 3, 4, 5, 6, 7]:  # [0, 1, 2, 3]:
        if geom_id in [0, 1, 2, 3, 4, 5, 6]:
            model, rot, off = _vmap_build(
                sys,
                pipeline_states,
                specular_map,
                tex,
                geom_id,
                idx,
                # sys.geom_bodyid[idx],
            )

            outs = [
                (
                    Instance(model=jax.tree_map(lambda x: x[i], model)),
                    jax.tree_map(lambda x: x[i], rot),
                    jax.tree_map(lambda x: x[i], off),
                )
                for i in range(model.verts.shape[0])
            ]

            print(f"outs: {type(outs)} // {len(outs)}")
            outs = [
                Obj(instance=instance, link_idx=link_idx, rot=rot, off=off)
                for (instance, rot, off) in outs
            ]
        else:
            outs = []
        # Plane
        # if geom_id == 0:
        #    # geom_xpos is (106, 3) -- what we want
        #    # print(
        #    #    f"geom_xpos: {pipeline_states.geom_xpos} // {pipeline_states.geom_xpos.shape}"
        #    # )

        #    # print(f"geom_xmat: {pipeline_states.geom_xmat.shape}")

        #    # print(f"rot: {pipeline_states.x.rot.shape}")

        #    # TODO: find (106, 4) for rot!!! (sys.geom_quat?)
        #    # rot: (4,) quaternion rotation the coordinate frame

        #    model, rot, off = _vmap_build(
        #        sys, pipeline_states, specular_map, tex, geom_id, idx
        #    )
        #    # print(f"model: {type(model)} // {model.verts.shape}")
        #    # print(f"rot: {rot.shape}")
        #    # print(f"off: {off.shape}")
        #    # qqq
        #    outs = [
        #        (
        #            Instance(model=jax.tree_map(lambda x: x[i], model)),
        #            jax.tree_map(lambda x: x[i], rot),
        #            jax.tree_map(lambda x: x[i], off),
        #        )
        #        for i in range(model.verts.shape[0])
        #    ]

        #    print(f"outs: {type(outs)} // {len(outs)}")
        #    outs = [
        #        Obj(instance=instance, link_idx=link_idx, rot=rot, off=off)
        #        for (instance, rot, off) in outs
        #    ]
        ## HFIELD
        # elif geom_id == 1:
        #    raise NotImplementedError("Geom of type HFIELD not implemented yet.")
        ## SPHERE
        # elif geom_id == 2:
        #
        # else:
        #    outs = []

        objs.extend(outs)

    return objs


@partial(jax.vmap, in_axes=(0, None))
def _inner_with_state_vmap(vmappable_objs: Iterable[Any], x: brax.Transform):
    pos = x.pos[vmappable_objs.link_idx] + math.rotate(
        vmappable_objs.off, x.rot[vmappable_objs.link_idx]
    )
    rot = math.quat_mul(x.rot[vmappable_objs.link_idx], vmappable_objs.rot)
    return pos, rot


def _with_state_vmap(
    objs: Iterable[Obj], vmappable_objs: Iterable[Any], x: brax.Transform
) -> list[Instance]:
    """For this process, we only need positon and orientation!"""
    print(f"IN _WITH_STATE_VMAP(): {vmappable_objs.rot.shape}")
    print(f"... {x.pos.shape} // {x.rot.shape}")
    poss, rots = _inner_with_state_vmap(vmappable_objs, x)
    print(f"... {poss.shape} // {rots.shape}")
    new_instances = [objs[i].instance.replace_with_position(poss[i]).replace_with_orientation(rots[i]) for i in range(poss.shape[0])]
    return new_instances


def _with_state(objs: Iterable[Obj], x: brax.Transform) -> list[Instance]:
    """x must have at least 1 element. This can be ensured by calling
    `x.concatenate(base.Transform.zero((1,)))`. x is `state.x`.

    This function does not modify any inputs, rather, it produces a new list of
    `Instance`s.
    """
    # if (len(x.pos.shape), len(x.rot.shape)) != (2, 2):
    #   raise RuntimeError('unexpected shape in state')
    # TODO: CAN WE JUST IGNORE THIS? DOES MJX HAVE US ALREADY G2G?
    instances: list[Instance] = []
    print(f"Objs: {objs[0]}")
    #qqq

    for obj in objs:
        # instances.append(obj.instance)
        i = obj.link_idx
        print(
            f"x.pos: {x.pos.shape} // {x.rot.shape} // {obj.off.shape}, {obj.rot.shape}"
        )
        # rotate((3,), (4,))
        # obj.off is local position offset rel. to body
        # x.rot is gotten from xquat in mjx.System: https://github.com/google/brax/blob/main/brax/mjx/pipeline.py#L75
        pos = x.pos[i] + math.rotate(obj.off, x.rot[i])
        # obj.rot is: local orientation offset of geom rel. to body
        rot = math.quat_mul(x.rot[i], obj.rot)
        instance = obj.instance
        print(f"ORIGINAL INSTANCE: {instance}")
        qqq
        instance = instance.replace_with_position(pos)
        instance = instance.replace_with_orientation(rot)
        instances.append(instance)
    return instances


_get_instances = jax.jit(
    jax.vmap(
        lambda objs, state: _with_state(
            objs, state.x.concatenate(base.Transform.zero((1,)))
        ),
        in_axes=(None, 0),
    )
)

_get_instances_vmap = jax.jit(
    jax.vmap(
        lambda objs, vmappable_objs, state: _with_state_vmap(
            objs, vmappable_objs, state.x.concatenate(base.Transform.zero((1,)))
        ),
        in_axes=(None, None, 0),
    )
)


@jax.default_matmul_precision("float32")
def render_instances(
    instances: list[Instance],
    width: int,
    height: int,
    camera: Camera,
    light: Optional[Light] = None,
    shadow: Optional[Shadow] = None,
    camera_target: Optional[jnp.ndarray] = None,
    enable_shadow: bool = False,
) -> jnp.ndarray:
    """Renders an RGB array of sequence of instances.

    Rendered result is not transposed with `transpose_for_display`; it is in
    floating numbers in [0, 1], not `uint8` in [0, 255].
    """
    if light is None:
        direction = jnp.array([0.57735, -0.57735, 0.57735])
        light = Light(
            direction=direction,
            ambient=0.8,
            diffuse=0.8,
            specular=0.6,
        )

    img = Renderer.get_camera_image(
        objects=instances,
        light=light,
        camera=camera,
        width=width,
        height=height,
        shadow_param=shadow,
    )
    arr = jax.lax.clamp(0.0, img, 1.0)
    return arr


_inner_inner_render = jax.jit(
    render_instances,
    static_argnames=("width", "height", "enable_shadow"),
    inline=True,
)


# @partial(jax.jit, static_argnames="hw")
def inner_render(instances, camera, target, hw) -> jnp.ndarray:
    img = _inner_inner_render(
        instances=instances,
        width=hw,
        height=hw,
        camera=camera,
        camera_target=target,
    )
    return img


# _inner_render = jax.jit(
#    jax.vmap(inner_render, in_axes=(0, 0, 0, None)), static_argnames="hw"
# )
_inner_render = jax.vmap(inner_render, in_axes=(0, 0, 0, None))


def render(
    objs, states: brax.State, batched_camera, batched_target, hw: int
) -> jnp.ndarray:
    batched_instances = _get_instances(objs, states)
    print(f"original batched_instances: {batched_instances}")
    print(f"type: {type(batched_instances)}")
    print(f"num objs: {len(batched_instances)}")
    print(f"done _get_instances()")
    qqq
    print(f"hw: {hw}")
    images = _inner_render(batched_instances, batched_camera, batched_target, hw)
    print("done with _inner_render()")
    return images


def render_cached(objs, vmappable_objs, states, batched_camera, batched_target, hw):
    batched_instances = _get_instances_vmap(objs, vmappable_objs, states)
    print(f"done _get_instances()")
    print(f"original batched_instances: {batched_instances}")
    print(f"type: {type(batched_instances)}")
    print(f"num objs: {len(batched_instances)}")
    print(f"hw: {hw}")
    images = _inner_render(batched_instances, batched_camera, batched_target, hw)
    print("done with _inner_render()")
    return images


# vmap'ing the camera init over the environment axis
_get_cameras = jax.jit(
    jax.vmap(
        lambda state, width, height: get_camera(state, width, height),
        in_axes=(0, None, None),
    )
)
_get_targets = jax.jit(jax.vmap(get_target))
_render = jax.jit(render, static_argnames="hw")
_render_cached = jax.jit(render_cached, static_argnames="hw")
