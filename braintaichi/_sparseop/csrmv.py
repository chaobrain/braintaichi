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
import taichi as ti
from jax import numpy as jnp
from jax.experimental.sparse import csr
from jax.interpreters import ad

from braintaichi._primitive._batch_utils import register_general_batching
from braintaichi._primitive._xla_custom_op import XLACustomOp
from .utils import csr_to_coo


def raw_csrmv_taichi(
    data: Union[jax.typing.ArrayLike, u.Quantity],
    indices: jax.typing.ArrayLike,
    indptr: jax.typing.ArrayLike,
    vector: jax.typing.ArrayLike,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
):
    out_shape = shape[1] if transpose else shape[0]
    if data.shape[0] != 1:
        if jax.devices()[0].platform == 'gpu':
            return [_csr_matvec_cusparse_p.bind(data, indices, indptr, vector, shape=shape, transpose=transpose)]
        else:
            if transpose:
                prim = _csr_matvec_transpose_heter_p
            else:
                prim = _csr_matvec_heter_p
    else:
        if transpose:
            prim = _csr_matvec_transpose_homo_p
        else:
            prim = _csr_matvec_homo_p

    return prim(data,
                indices,
                indptr,
                vector,
                outs=[jax.ShapeDtypeStruct((out_shape,), dtype=data.dtype)],
                transpose=transpose,
                shape=shape)


# -------------
# CPU operators
# -------------
@ti.kernel
def _sparse_csr_matvec_transpose_homo_cpu(values: ti.types.ndarray(ndim=1),
                                          col_indices: ti.types.ndarray(ndim=1),
                                          row_ptr: ti.types.ndarray(ndim=1),
                                          vector: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
    value = values[0]
    ti.loop_config(serialize=True)
    for row_i in range(row_ptr.shape[0] - 1):
        for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            out[col_indices[j]] += value * vector[row_i]


@ti.kernel
def _sparse_csr_matvec_transpose_heter_cpu(values: ti.types.ndarray(ndim=1),
                                           col_indices: ti.types.ndarray(ndim=1),
                                           row_ptr: ti.types.ndarray(ndim=1),
                                           vector: ti.types.ndarray(ndim=1),
                                           out: ti.types.ndarray(ndim=1)):
    ti.loop_config(serialize=True)
    for row_i in range(row_ptr.shape[0] - 1):
        for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            out[col_indices[j]] += vector[row_i] * values[j]


@ti.kernel
def _sparse_csr_matvec_homo_cpu(values: ti.types.ndarray(ndim=1),
                                col_indices: ti.types.ndarray(ndim=1),
                                row_ptr: ti.types.ndarray(ndim=1),
                                vector: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
    value = values[0]
    # ti.loop_config(serialize=True)
    for row_i in range(row_ptr.shape[0] - 1):
        r = 0.
        for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            r += vector[col_indices[j]]
        out[row_i] = r * value


@ti.kernel
def _sparse_csr_matvec_heter_cpu(values: ti.types.ndarray(ndim=1),
                                 col_indices: ti.types.ndarray(ndim=1),
                                 row_ptr: ti.types.ndarray(ndim=1),
                                 vector: ti.types.ndarray(ndim=1),
                                 out: ti.types.ndarray(ndim=1)):
    # ti.loop_config(serialize=True)
    for row_i in range(row_ptr.shape[0] - 1):
        r = 0.
        for j in range(row_ptr[row_i], row_ptr[row_i + 1]):
            r += values[j] * vector[col_indices[j]]
        out[row_i] = r


# -------------
# GPU operators
# -------------

@ti.kernel
def _sparse_csr_matvec_transpose_homo_gpu(values: ti.types.ndarray(ndim=1),
                                          col_indices: ti.types.ndarray(ndim=1),
                                          row_ptr: ti.types.ndarray(ndim=1),
                                          vector: ti.types.ndarray(ndim=1),
                                          out: ti.types.ndarray(ndim=1)):
    value = values[0]
    for i in range((row_ptr.shape[0] - 1) * 32):
        row_i = i >> 5
        index = i & 31
        j = row_ptr[row_i] + index
        end_index = row_ptr[row_i + 1]
        while j < end_index:
            out[col_indices[j]] += value * vector[row_i]
            j += 32


@ti.kernel
def _sparse_csr_matvec_homo_gpu(values: ti.types.ndarray(ndim=1),
                                col_indices: ti.types.ndarray(ndim=1),
                                row_ptr: ti.types.ndarray(ndim=1),
                                vector: ti.types.ndarray(ndim=1),
                                out: ti.types.ndarray(ndim=1)):
    value = values[0]
    for i in range((row_ptr.shape[0] - 1) * 32):
        row_i = i >> 5
        index = i & 31
        r = 0.
        j = row_ptr[row_i] + index
        end_index = row_ptr[row_i + 1]
        while j < end_index:
            r += vector[col_indices[j]]
            j += 32
        out[row_i] += value * r


@ti.kernel
def _sparse_csr_matvec_transpose_heter_gpu(values: ti.types.ndarray(ndim=1),
                                           col_indices: ti.types.ndarray(ndim=1),
                                           row_ptr: ti.types.ndarray(ndim=1),
                                           vector: ti.types.ndarray(ndim=1),
                                           out: ti.types.ndarray(ndim=1)):
    for i in range((row_ptr.shape[0] - 1) * 32):
        row_i = i >> 5
        index = i & 31
        j = row_ptr[row_i] + index
        end_index = row_ptr[row_i + 1]
        while j < end_index:
            out[col_indices[j]] += values[j] * vector[row_i]
            j += 32


@ti.kernel
def _sparse_csr_matvec_heter_gpu(values: ti.types.ndarray(ndim=1),
                                 col_indices: ti.types.ndarray(ndim=1),
                                 row_ptr: ti.types.ndarray(ndim=1),
                                 vector: ti.types.ndarray(ndim=1),
                                 out: ti.types.ndarray(ndim=1)):
    for i in range((row_ptr.shape[0] - 1) * 32):
        row_i = i >> 5
        index = i & 31
        r = 0.
        j = row_ptr[row_i] + index
        end_index = row_ptr[row_i + 1]
        while j < end_index:
            r += values[j] * vector[col_indices[j]]
            j += 32
        out[row_i] += r  # TODO: warp-level primitive


def _sparse_csr_matvec_jvp_values(val_dot, values, col_indices, row_ptr, vector, *, outs, transpose, shape):
    return raw_csrmv_taichi(val_dot, col_indices, row_ptr, vector, shape=shape, transpose=transpose)


def _sparse_csr_matvec_jvp_vector(vec_dot, values, col_indices, row_ptr, vector, *, outs, transpose, shape):
    return raw_csrmv_taichi(values, col_indices, row_ptr, vec_dot, shape=shape, transpose=transpose)


def _sparse_csr_matvec_transpose(
    ct, data, indices, indptr, vector, *, outs, transpose, shape,
):
    if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
        raise ValueError("Cannot transpose with respect to sparse indices.")
    if ad.is_undefined_primal(vector):
        ct_vector = raw_csrmv_taichi(data, indices, indptr, ct[0], shape=shape, transpose=not transpose)[0]
        return data, indices, indptr, (ad.Zero(vector) if type(ct[0]) is ad.Zero else ct_vector)

    else:
        if type(ct[0]) is ad.Zero:
            ct_data = ad.Zero(data)
        else:
            if data.aval.shape[0] == 1:  # scalar
                ct_data = raw_csrmv_taichi(jnp.ones(1), indices, indptr, vector, shape=shape, transpose=transpose)[0]
                ct_data = jnp.inner(ct[0], ct_data)
            else:
                row, col = csr_to_coo(indices, indptr)
                ct_data = vector[row] * ct[0][col] if transpose else vector[col] * ct[0][row]

        return ct_data, indices, indptr, vector


def _define_op(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    prim.defjvp(_sparse_csr_matvec_jvp_values, None, None, _sparse_csr_matvec_jvp_vector)
    prim.def_transpose_rule(_sparse_csr_matvec_transpose)
    return prim


# transpose homo
_csr_matvec_transpose_homo_p = _define_op(cpu_kernel=_sparse_csr_matvec_transpose_homo_cpu,
                                          gpu_kernel=_sparse_csr_matvec_transpose_homo_gpu)

# no transpose homo
_csr_matvec_homo_p = _define_op(cpu_kernel=_sparse_csr_matvec_homo_cpu,
                                gpu_kernel=_sparse_csr_matvec_homo_gpu)

# transpose heter
_csr_matvec_transpose_heter_p = _define_op(cpu_kernel=_sparse_csr_matvec_transpose_heter_cpu,
                                           gpu_kernel=_sparse_csr_matvec_transpose_heter_gpu)

# no transpose heter
_csr_matvec_heter_p = _define_op(cpu_kernel=_sparse_csr_matvec_heter_cpu,
                                 gpu_kernel=_sparse_csr_matvec_heter_gpu)

# heter cusparse
_csr_matvec_cusparse_p = csr.csr_matvec_p
register_general_batching(_csr_matvec_cusparse_p)
