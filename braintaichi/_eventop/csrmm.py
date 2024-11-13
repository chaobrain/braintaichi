# Copyright 2024- BrainPy Ecosystem Limited. All Rights Reserved.
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

# -*- coding: utf-8 -*-

from typing import Tuple, Union

import brainunit as u
import jax
import numpy as np
import taichi as ti
from jax import numpy as jnp
from jax.experimental.sparse import csr
from jax.interpreters import ad

from braintaichi._primitive._batch_utils import register_general_batching
from braintaichi._primitive._xla_custom_op import XLACustomOp
from braintaichi._sparseop.csrmm import raw_csrmm_taichi as normal_csrmm
from braintaichi._sparseop.utils import csr_to_coo


def raw_event_csrmm_taichi(
    data: Union[jax.typing.ArrayLike, u.Quantity],
    indices: jax.typing.ArrayLike,
    indptr: jax.typing.ArrayLike,
    matrix: jax.typing.ArrayLike,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
):
    assert len(shape) == 2

    data = jnp.atleast_1d(data)
    if np.ndim(data) == 1:
        if data.shape[0] not in [1, indices.shape[0]]:
            raise ValueError('The size of data should be 1 or be consistent with indices.'
                             f'But we got {data.shape} != {indices.shape}, {data.shape} != 1.')

    assert data.ndim == indices.ndim == indptr.ndim == 1
    assert matrix.ndim == 2
    assert indptr.shape[0] == shape[0] + 1
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise ValueError('indices should be a 1D vector with integer type.')
    if not jnp.issubdtype(indptr.dtype, jnp.integer):
        raise ValueError('indptr should be a 1D vector with integer type.')

    out_shape = shape[1] if transpose else shape[0]
    result_shape = (out_shape, matrix.shape[1])
    # if the shape of indices is (0,), then we return a zero matrix
    if indices.shape[0] == 0:
        return [jnp.zeros(result_shape, dtype=data.dtype), ]

    assert matrix.shape[0] == (shape[0] if transpose else shape[1])

    # homo -> taichi
    # heter -> cusparse
    if data.shape[0] != 1:
        if matrix.dtype == jnp.bool_:
            # change dtype to float
            matrix = matrix.astype(jnp.float64 if jax.config.read('jax_enable_x64') else jnp.float32)
        return [_csr_matmat_cusparse_p.bind(data, indices, indptr, matrix, shape=shape, transpose=transpose), ]
    else:
        if transpose:
            if matrix.dtype == jnp.bool_:
                prim = _event_csr_matmat_transpose_homo_p
            else:
                return normal_csrmm(data, indices, indptr, matrix, shape=shape, transpose=transpose)
        else:
            if matrix.dtype == jnp.bool_:
                prim = _event_csr_matmat_bool_homo_p
            else:
                return normal_csrmm(data, indices, indptr, matrix, shape=shape, transpose=transpose)
        return prim(data,
                    indices,
                    indptr,
                    matrix,
                    outs=[jax.ShapeDtypeStruct(result_shape, dtype=data.dtype)],
                    transpose=transpose,
                    shape=shape)


# taichi kernels

@ti.kernel
def _event_csr_matmat_transpose_heter(values: ti.types.ndarray(ndim=1),
                                      col_indices: ti.types.ndarray(ndim=1),
                                      row_ptr: ti.types.ndarray(ndim=1),
                                      matrix: ti.types.ndarray(ndim=2),
                                      out: ti.types.ndarray(ndim=2)):
    for col_i, row_k in ti.ndrange(out.shape[1], out.shape[0]):
        for row_j in range(matrix.shape[0]):
            if matrix[row_j, col_i] != 0.:
                for j in range(row_ptr[row_j], row_ptr[row_j + 1]):
                    if col_indices[j] == row_k:
                        out[row_k, col_i] += values[j] * matrix[row_j, col_i]


@ti.kernel
def _event_csr_matmat_transpose_bool_heter(values: ti.types.ndarray(ndim=1),
                                           col_indices: ti.types.ndarray(ndim=1),
                                           row_ptr: ti.types.ndarray(ndim=1),
                                           matrix: ti.types.ndarray(ndim=2),
                                           out: ti.types.ndarray(ndim=2)):
    for col_i, row_k in ti.ndrange(out.shape[1], out.shape[0]):
        for row_j in range(matrix.shape[0]):
            if matrix[row_j, col_i]:
                for j in range(row_ptr[row_j], row_ptr[row_j + 1]):
                    if col_indices[j] == row_k:
                        out[row_k, col_i] += values[j] * matrix[row_j, col_i]


@ti.kernel
def _event_csr_matmat_heter(values: ti.types.ndarray(ndim=1),
                            col_indices: ti.types.ndarray(ndim=1),
                            row_ptr: ti.types.ndarray(ndim=1),
                            matrix: ti.types.ndarray(ndim=2),
                            out: ti.types.ndarray(ndim=2)):
    for row_i, col_k in ti.ndrange(out.shape[0], out.shape[1]):
        r = 0.
        for row_j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            if matrix[col_indices[row_j], col_k] != 0.:
                r += values[row_j] * matrix[col_indices[row_j], col_k]
        out[row_i, col_k] = r


@ti.kernel
def _event_csr_matmat_bool_heter(values: ti.types.ndarray(ndim=1),
                                 col_indices: ti.types.ndarray(ndim=1),
                                 row_ptr: ti.types.ndarray(ndim=1),
                                 matrix: ti.types.ndarray(ndim=2),
                                 out: ti.types.ndarray(ndim=2)):
    for row_i, col_k in ti.ndrange(out.shape[0], out.shape[1]):
        r = 0.
        for row_j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            if matrix[col_indices[row_j], col_k]:
                r += values[row_j] * matrix[col_indices[row_j], col_k]
        out[row_i, col_k] = r


@ti.kernel
def _event_csr_matmat_transpose_homo(values: ti.types.ndarray(ndim=1),
                                     col_indices: ti.types.ndarray(ndim=1),
                                     row_ptr: ti.types.ndarray(ndim=1),
                                     matrix: ti.types.ndarray(ndim=2),
                                     out: ti.types.ndarray(ndim=2)):
    value = values[0]
    for col_i, row_k in ti.ndrange(out.shape[1], out.shape[0]):
        for row_j in range(matrix.shape[0]):
            if matrix[row_j, col_i] != 0.:
                for j in range(row_ptr[row_j], row_ptr[row_j + 1]):
                    if col_indices[j] == row_k:
                        out[row_k, col_i] += value * matrix[row_j, col_i]


@ti.kernel
def _event_csr_matmat_transpose_bool_homo(values: ti.types.ndarray(ndim=1),
                                          col_indices: ti.types.ndarray(ndim=1),
                                          row_ptr: ti.types.ndarray(ndim=1),
                                          matrix: ti.types.ndarray(ndim=2),
                                          out: ti.types.ndarray(ndim=2)):
    value = values[0]
    for col_i, row_k in ti.ndrange(out.shape[1], out.shape[0]):
        for row_j in range(matrix.shape[0]):
            if matrix[row_j, col_i]:
                for j in range(row_ptr[row_j], row_ptr[row_j + 1]):
                    if col_indices[j] == row_k:
                        out[row_k, col_i] += value * matrix[row_j, col_i]


@ti.kernel
def _event_csr_matmat_homo(values: ti.types.ndarray(ndim=1),
                           col_indices: ti.types.ndarray(ndim=1),
                           row_ptr: ti.types.ndarray(ndim=1),
                           matrix: ti.types.ndarray(ndim=2),
                           out: ti.types.ndarray(ndim=2)):
    value = values[0]
    for row_i, col_k in ti.ndrange(out.shape[0], out.shape[1]):
        r = 0.
        for row_j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            if matrix[col_indices[row_j], col_k] != 0.:
                r += matrix[col_indices[row_j], col_k]
        out[row_i, col_k] = r * value


@ti.kernel
def _event_csr_matmat_bool_homo(values: ti.types.ndarray(ndim=1),
                                col_indices: ti.types.ndarray(ndim=1),
                                row_ptr: ti.types.ndarray(ndim=1),
                                matrix: ti.types.ndarray(ndim=2),
                                out: ti.types.ndarray(ndim=2)):
    value = values[0]
    for row_i, col_k in ti.ndrange(out.shape[0], out.shape[1]):
        r = 0.
        for row_j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            if matrix[col_indices[row_j], col_k]:
                r += matrix[col_indices[row_j], col_k]
        out[row_i, col_k] = r * value


def _event_csr_matmat_jvp_values(val_dot, values, col_indices, row_ptr, matrix, *, outs, transpose, shape):
    return normal_csrmm(val_dot, col_indices, row_ptr, matrix, shape=shape, transpose=transpose)


def _event_csr_matmat_jvp_matrix(mat_dot, values, col_indices, row_ptr, matrix, *, outs, transpose, shape):
    return normal_csrmm(values, col_indices, row_ptr, mat_dot, shape=shape, transpose=transpose)


def _event_csr_matmat_transpose(
    ct, data, indices, indptr, matrix, *, outs, transpose, shape,
):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(matrix):
        ct_matrix = raw_event_csrmm_taichi(data, indices, indptr, ct[0], shape=shape, transpose=not transpose)[0]
        return data, indices, indptr, (ad.Zero(matrix) if type(ct[0]) is ad.Zero else ct_matrix)

    else:
        if type(ct[0]) is ad.Zero:
            ct_data = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_data = raw_event_csrmm_taichi(jnp.ones(1),
                                                 indices, indptr, matrix,
                                                 shape=shape,
                                                 transpose=transpose)[0]
                ct_data = jnp.sum(ct[0] * ct_data)
            else:  # heter
                matrix = jnp.asarray(matrix)
                row, col = csr_to_coo(indices, indptr)
                ct_data = (ct[0][row] * matrix[col]).sum(1)
        return ct_data, indices, indptr, matrix


def _define_op(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    prim.defjvp(_event_csr_matmat_jvp_values, None, None, _event_csr_matmat_jvp_matrix)
    prim.def_transpose_rule(_event_csr_matmat_transpose)
    return prim


# transpose heter
_event_csr_matmat_transpose_heter_p = _define_op(cpu_kernel=_event_csr_matmat_transpose_heter,
                                                 gpu_kernel=_event_csr_matmat_transpose_heter)

# no transpose heter
_event_csr_matmat_heter_p = _define_op(cpu_kernel=_event_csr_matmat_heter,
                                       gpu_kernel=_event_csr_matmat_heter)

# transpose homo
_event_csr_matmat_transpose_homo_p = _define_op(cpu_kernel=_event_csr_matmat_transpose_homo,
                                                gpu_kernel=_event_csr_matmat_transpose_homo)

# no transpose homo
_event_csr_matmat_homo_p = _define_op(cpu_kernel=_event_csr_matmat_homo,
                                      gpu_kernel=_event_csr_matmat_homo)

# bool transpose heter
_event_csr_matmat_transpose_bool_heter_p = _define_op(cpu_kernel=_event_csr_matmat_transpose_bool_heter,
                                                      gpu_kernel=_event_csr_matmat_transpose_bool_heter)

# bool no transpose heter
_event_csr_matmat_bool_heter_p = _define_op(cpu_kernel=_event_csr_matmat_bool_heter,
                                            gpu_kernel=_event_csr_matmat_bool_heter)

# bool transpose homo
_event_csr_matmat_transpose_bool_homo_p = _define_op(cpu_kernel=_event_csr_matmat_transpose_bool_homo,
                                                     gpu_kernel=_event_csr_matmat_transpose_bool_homo)

# bool no transpose homo
_event_csr_matmat_bool_homo_p = _define_op(cpu_kernel=_event_csr_matmat_bool_homo,
                                           gpu_kernel=_event_csr_matmat_bool_homo)

# heter CUSPARSE
_csr_matmat_cusparse_p = csr.csr_matmat_p
register_general_batching(_csr_matmat_cusparse_p)
