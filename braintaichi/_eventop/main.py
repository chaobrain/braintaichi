# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
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
# ==============================================================================


from typing import Union, Tuple

import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

from .csrmm import raw_event_csrmm_taichi
from .csrmv import event_csrmv_taichi

__all__ = [
    'event_csrmv',
    'event_csrmm',
]


def event_csrmm(
    data: Union[jax.typing.ArrayLike, u.Quantity],
    indices: jax.typing.ArrayLike,
    indptr: jax.typing.ArrayLike,
    matrix: jax.typing.ArrayLike,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
):
    """Product of CSR sparse matrix and a dense event matrix.

    Args:
        data : array of shape ``(nse,)``, float.
        indices : array of shape ``(nse,)``
        indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
        matrix : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
                 dtype ``data.dtype``
        shape : length-2 tuple representing the matrix shape
        transpose : boolean specifying whether to transpose the sparse matrix
                    before computing.

    Returns:
        C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix-matrix product product.
    """
    return raw_event_csrmm_taichi(data, indices, indptr, matrix, shape=shape, transpose=transpose)[0]


def event_csrmv(
    data: Union[jax.typing.ArrayLike, u.Quantity],
    indices: jax.Array,
    indptr: jax.Array,
    events: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
) -> jax.Array:
    """Product of a sparse CSR matrix and a dense event vector.

    This function supports JAX transformations, including `jit()`, `grad()`,
    `vmap()` and `pmap()`.

    Parameters
    ----------
    data: ndarray, float
      An array of shape ``(nse,)``.
    indices: ndarray
      An array of shape ``(nse,)``.
    indptr: ndarray
      An array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``.
    events: ndarray
      An array of shape ``(shape[0] if transpose else shape[1],)``
      and dtype ``data.dtype``.
    shape: tuple
      A length-2 tuple representing the matrix shape.
    transpose: bool
      A boolean specifying whether to transpose the sparse matrix
      before computing.
      If ``transpose=True``, the operator will compute based on the
      event-driven property of the ``events`` vector.

    Returns
    -------
    y : Array
      The array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
    """
    # checking
    data = jnp.atleast_1d(data)
    if np.ndim(data) == 1:
        if data.shape[0] not in [1, indices.shape[0]]:
            raise ValueError('The size of data should be 1 or be consistent with indices.'
                             f'But we got {data.shape} != {indices.shape}, {data.shape} != 1.')
    else:
        raise ValueError('data should be a scalar or 1D vector. '
                         f'But we got {np.ndim(data)}-D array.')
    if np.ndim(indices) != 1:
        raise ValueError('indices should be a 1D vector with integer type.')
    if np.ndim(indptr) != 1:
        raise ValueError('indptr should be a 1D vector with integer type.')
    if indices.dtype not in [jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]:
        raise ValueError(
            'indices should be a 1D vector with int8, int16, int32, int64, uint8, uint16, uint32 or uint64 type.')
    if indptr.dtype not in [jnp.int8, jnp.int16, jnp.int32, jnp.int64, jnp.uint8, jnp.uint16, jnp.uint32, jnp.uint64]:
        raise ValueError(
            'indptr should be a 1D vector with int8, int16, int32, int64, uint8, uint16, uint32 or uint64 type.')
    if np.ndim(events) != 1:
        raise ValueError('events should be a 1D vector.')
    if len(shape) != 2:
        raise ValueError('shape should be a length-2 tuple.')
    if transpose:
        if events.shape[0] != shape[0]:
            raise ValueError(f'Shape mismatch, vec ({events.shape[0]},) @ mat {shape}.')
    else:
        if events.shape[0] != shape[1]:
            raise ValueError(f'Shape mismatch, mat {shape} @ vec ({events.shape[0]},).')

    # if the shape of indices is (0,), then we return a zero vector
    if indices.shape[0] == 0:
        return jnp.zeros(shape[1] if transpose else shape[0], dtype=data.dtype)

    return event_csrmv_taichi(data, indices, indptr, events, shape=shape, transpose=transpose)[0]
