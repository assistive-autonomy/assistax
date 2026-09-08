"""
Low-level pytree manipulation utilities.

This module has no internal ``assistax`` dependencies (it imports only ``jax``),
so it can be shared by ``assistax.baselines.utils``, ``assistax.wrappers.aht`` and
the rendering tools without creating a circular import.
"""

import jax
import jax.numpy as jnp


def _tree_take(pytree, indices, axis=None):
    """
    Take elements from each leaf of a pytree along a specified axis.

    Args:
        pytree: JAX pytree (nested structure of arrays)
        indices: Indices to take from each array
        axis: Axis along which to take indices (None for flat indexing)

    Returns:
        Pytree with same structure but indexed arrays
    """
    return jax.tree.map(lambda x: x.take(indices, axis=axis), pytree)


def _tree_shape(pytree):
    """
    Get the shape of each leaf in a pytree.

    Args:
        pytree: JAX pytree (nested structure of arrays)

    Returns:
        Pytree with same structure but shapes instead of arrays
    """
    return jax.tree.map(lambda x: x.shape, pytree)


def _unstack_tree(pytree):
    """
    Unstack a pytree along the first axis, yielding a list of pytrees.

    Converts a pytree where each leaf has shape (N, ...) into a list of N pytrees
    where each leaf has shape (...).

    Args:
        pytree: JAX pytree with arrays of shape (N, ...)

    Returns:
        List of N pytrees, each with arrays of shape (...)
    """
    leaves, treedef = jax.tree_util.tree_flatten(pytree)
    unstacked_leaves = zip(*leaves)
    return [jax.tree_util.tree_unflatten(treedef, leaves)
            for leaves in unstacked_leaves]


def _stack_tree(pytree_list, axis=0):
    """
    Stack a list of pytrees along a specified axis.

    Args:
        pytree_list: List of pytrees with compatible structures
        axis: Axis along which to stack

    Returns:
        Single pytree with stacked arrays
    """
    return jax.tree.map(
        lambda *leaf: jnp.stack(leaf, axis=axis),
        *pytree_list
    )
