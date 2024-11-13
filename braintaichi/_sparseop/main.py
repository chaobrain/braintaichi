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

from typing import Tuple, Union

import brainunit as u
import jax
from jax import numpy as jnp, dtypes, default_backend

from braintaichi._misc import set_module_as
from .coomv import _coomv_cusparse_p
from .csrmm import raw_csrmm_taichi
from .csrmv import raw_csrmv_taichi

__all__ = [
    'coomv',
    'csrmv',
    'csrmm',
]


@set_module_as('braintaichi')
def coomv(
    data: Union[jax.typing.ArrayLike, u.Quantity],
    row: jax.typing.ArrayLike,
    col: jax.typing.ArrayLike,
    vector: jax.typing.ArrayLike,
    *,
    shape: Tuple[int, int],
    rows_sorted: bool = False,
    cols_sorted: bool = False,
    transpose: bool = False,
    method: str = 'cusparse'
):
    """Product of COO sparse matrix and a dense vector using cuSPARSE algorithm.

    This function supports JAX transformations, including `jit()`, `grad()`,
    `vmap()` and `pmap()`.

    Parameters
    ----------
    data: ndarray, float
      An array of shape ``(nse,)``.
    row: ndarray
      An array of shape ``(nse,)``.
    col: ndarray
      An array of shape ``(nse,)`` and dtype ``row.dtype``.
    vector: ndarray
      An array of shape ``(shape[0] if transpose else shape[1],)`` and
      dtype ``data.dtype``.
    shape: tuple of int
      The shape of the sparse matrix.
    rows_sorted: bool
      Row index are sorted.
    cols_sorted: bool
      Column index are sorted.
    transpose: bool
      A boolean specifying whether to transpose the sparse matrix
      before computing.
    method: str
      The method used to compute the matrix-vector multiplication.

    Returns
    -------
    y: ndarray
      An array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
    """

    data = jnp.atleast_1d(jnp.asarray(data))
    row = jnp.asarray(row)
    col = jnp.asarray(col)
    vector = jnp.asarray(vector)

    if method == 'cusparse':
        if default_backend() != 'cpu':
            if data.shape[0] == 1:
                data = jnp.ones(row.shape, dtype=data.dtype) * data
            if row.dtype in [jnp.uint32, jnp.uint64]:
                row = jnp.asarray(row, dtype=dtypes.canonicalize_dtype(jnp.int64))
            if col.dtype in [jnp.uint32, jnp.uint64]:
                col = jnp.asarray(col, dtype=dtypes.canonicalize_dtype(jnp.int64))
        return _coomv_cusparse_p.bind(data,
                                      row,
                                      col,
                                      vector,
                                      shape=shape,
                                      rows_sorted=rows_sorted,
                                      cols_sorted=cols_sorted,
                                      transpose=transpose)

    else:
        raise ValueError


@set_module_as('braintaichi')
def csrmm(
    data: Union[jax.typing.ArrayLike, u.Quantity],
    indices: jax.typing.ArrayLike,
    indptr: jax.typing.ArrayLike,
    matrix: jax.typing.ArrayLike,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
):
    """
    Product of CSR sparse matrix and a dense matrix.

    Args:
        data : array of shape ``(nse,)``.
        indices : array of shape ``(nse,)``
        indptr : array of shape ``(shape[0] + 1,)`` and dtype ``indices.dtype``
        B : array of shape ``(shape[0] if transpose else shape[1], cols)`` and
        dtype ``data.dtype``
        shape : length-2 tuple representing the matrix shape
        transpose : boolean specifying whether to transpose the sparse matrix
        before computing.

    Returns:
        C : array of shape ``(shape[1] if transpose else shape[0], cols)``
        representing the matrix-matrix product.
    """
    return raw_csrmm_taichi(data, indices, indptr, matrix, shape=shape, transpose=transpose)[0]


@set_module_as('braintaichi')
def csrmv(
    data: Union[jax.typing.ArrayLike, u.Quantity],
    indices: jax.typing.ArrayLike,
    indptr: jax.typing.ArrayLike,
    vector: jax.typing.ArrayLike,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
):
    """Product of CSR sparse matrix and a dense vector using cuSPARSE algorithm.

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
    vector: ndarray
      An array of shape ``(shape[0] if transpose else shape[1],)``
      and dtype ``data.dtype``.
    shape: tuple of int
      A length-2 tuple representing the matrix shape.
    transpose: bool
      A boolean specifying whether to transpose the sparse matrix
      before computing.
    method: str
      The method used to compute Matrix-Vector Multiplication. Default is ``taichi``.
      The candidate methods are:

      - ``None``: default using Taichi kernel.
      - ``cusparse``: using cuSPARSE library.
      - ``scalar``:
      - ``vector``:
      - ``adaptive``:

    Returns
    -------
    y : ndarry
      The array of shape ``(shape[1] if transpose else shape[0],)`` representing
      the matrix vector product.
    """

    data = jnp.atleast_1d(data)

    if vector.dtype == jnp.bool_:
        vector = jnp.asarray(vector, dtype=data.dtype)

    if data.dtype not in [jnp.float16, jnp.float32, jnp.float64]:
        raise TypeError('Only support float16, float32 or float64 type. '
                        f'But we got {data.dtype}.')
    if data.dtype != vector.dtype:
        raise TypeError('The types of data and vector should be the same. '
                        f'But we got {data.dtype} != {vector.dtype}.')
    assert data.ndim == indices.ndim == indptr.ndim == vector.ndim == 1
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError('indices should be a 1D vector with integer type.')
    if not jnp.issubdtype(indptr.dtype, jnp.integer):
        raise ValueError('indptr should be a 1D vector with integer type.')

    # if the shape of indices is (0,), then we return a zero vector
    if indices.shape[0] == 0:
        return jnp.zeros(shape[1] if transpose else shape[0], dtype=data.dtype)

    return raw_csrmv_taichi(data, indices, indptr, vector, shape=shape, transpose=transpose)[0]
