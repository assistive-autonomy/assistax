"""
Some geometric primitives yanked from the v1 version of Brax and frozen in time
"""

from typing import Optional
from flax import struct
import jax
import jax.numpy as jp
from brax import math
from brax.base import Base, Transform


@struct.dataclass
class Geometry(Base):
    """A surface or spatial volume with a shape and material properties.

    Attributes:
      link_idx: Link index to which this Geometry is attached
      transform: transform for the geometry frame relative to the link frame, or
        relative to the world frame in the case of unparented geometry
      friction: resistance encountered when sliding against another geometry
      elasticity: bounce/restitution encountered when hitting another geometry
      solver_params: (7,) solver parameters (reference, impedance)
    """

    link_idx: Optional[jax.Array]
    transform: Transform
    friction: jax.Array
    elasticity: jax.Array
    solver_params: jax.Array


@struct.dataclass
class Sphere(Geometry):
    """A sphere.

    Attributes:
      radius: radius of the sphere
      rgba: (4,) the rgba to display in the renderer
    """

    radius: jax.Array
    rgba: Optional[jax.Array] = None


@struct.dataclass
class Capsule(Geometry):
    """A capsule.

    Attributes:
      radius: radius of the capsule end
      length: distance between the two capsule end centroids
      rgba: (4,) the rgba to display in the renderer
    """

    radius: jax.Array
    length: jax.Array
    rgba: Optional[jax.Array] = None


@struct.dataclass
class Box(Geometry):
    """A box.

    Attributes:
      halfsize: (3,) half sizes for each box side
      rgba: (4,) the rgba to display in the renderer
    """

    halfsize: jax.Array
    rgba: Optional[jax.Array] = None


@struct.dataclass
class Cylinder(Geometry):
    """A cylinder.

    Attributes:
      radius: (1,) radius of the top and bottom of the cylinder
      length: (1,) length of the cylinder
      rgba: (4,) the rgba to display in the renderer
    """

    radius: jax.Array
    length: jax.Array
    rgba: Optional[jax.Array] = None


@struct.dataclass
class Plane(Geometry):
    """An infinite plane whose normal points at +z in its coordinate space.

    Attributes:
      rgba: (4,) the rgba to display in the renderer, currently unused
    """

    rgba: Optional[jax.Array] = None


@struct.dataclass
class Mesh(Geometry):
    """A mesh loaded from an OBJ or STL file.

    The mesh is expected to be in the counter-clockwise winding order.

    Attributes:
      vert: (num_verts, 3) spatial coordinates associated with each vertex
      face: (num_faces, num_face_vertices) vertices associated with each face
      rgba: (4,) the rgba to display in the renderer, currently unused
    """

    vert: jax.Array
    face: jax.Array
    rgba: Optional[jax.Array] = None


@struct.dataclass
class Convex(Geometry):
    """A convex mesh geometry.

    Attributes:
      vert: (num_verts, 3) spatial coordinates associated with each vertex
      face: (num_faces, num_face_vertices) vertices associated with each face
      unique_edge: (num_unique, 2) vert index associated with each unique edge
      rgba: (4,) the rgba to display in the renderer, currently unused
    """

    vert: jax.Array
    face: jax.Array
    unique_edge: jax.Array
    rgba: Optional[jax.Array] = None
