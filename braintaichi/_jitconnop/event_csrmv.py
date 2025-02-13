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


from typing import Tuple

import jax
import taichi as ti
from jax import numpy as jnp

from braintaichi._misc import _get_dtype
from braintaichi._primitive._xla_custom_op import XLACustomOp
from .csrmv import (_general_checking,
                    raw_mv_prob_homo,
                    raw_mv_prob_uniform,
                    raw_mv_prob_normal,
                    _mv_prob_homo_transpose,
                    _mv_prob_uniform_transpose,
                    _mv_prob_normal_transpose)
from braintaichi.rand._taichi_rand import (lfsr88_key, lfsr88_random_integers, lfsr88_uniform, lfsr88_normal)


# -------------
# CPU function
# -------------
# For each non-zero event value, it generates a random key using a
# function lfsr88_key and then uses this key to compute random integers
# and update the out array based on the computed indices and weight.
#
# The function is likely designed to be parallelized.

@ti.kernel
def _event_mv_prob_homo_bool_cpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
        if events[i_col]:
            key = lfsr88_key(seed0 + i_col)
            key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
            while i_row < num_row:
                out[i_row] += weight0
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_homo_outdim_parallel_bool_cpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
        r = 0.
        key = lfsr88_key(seed0 + i_row)
        key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
        while i_col < num_col:
            if events[i_col]:
                r += weight0
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] = r


# -------------
# GPU function
# -------------
# Contrary to the CPU functions, for each column,
# this function will 32 threads (one warp) to make
# the just-in-time random generation parallelized.

@ti.kernel
def _event_mv_prob_homo_bool_gpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col]:
            index = i & 31
            i_row = step * index - 1
            end = ti.min(i_row + step, num_row)
            key = lfsr88_key(seed0 + i)
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_row += inc
            while i_row < end:
                out[i_row] += weight0
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_homo_outdim_parallel_bool_gpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.u32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_row * 32):
        i_row = i >> 5
        index = i & 31
        i_col = step * index - 1
        end_col = ti.min(i_col + step, num_col)
        r = 0.
        key = lfsr88_key(seed0 + i)
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc
        while i_col < end_col:
            r += weight0 * events[i_col]  # TODO: speed comparison without if else
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] += r  # TODO: warp-level reduction


def _reverse(shape):
    return shape[::-1]


# -------------
# CPU function
# -------------
# For each non-zero event value, it generates a random key using a
# function lfsr88_key and then uses this key to compute random integers
# and update the out array based on the computed indices and weight.
#
# The function is likely designed to be parallelized.

@ti.kernel
def _event_mv_prob_homo_cpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
        if events[i_col] != 0.:
            key = lfsr88_key(seed0 + i_col)
            key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
            while i_row < num_row:
                out[i_row] += weight0
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_homo_outdim_parallel_cpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
        r = 0.
        key = lfsr88_key(seed0 + i_row)
        key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
        while i_col < num_col:
            if events[i_col] != 0.:
                r += weight0
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] = r  # TODO: warp-level reduction


# -------------
# GPU function
# -------------
# Contrary to the CPU functions, for each column,
# this function will 32 threads (one warp) to make
# the just-in-time random generation parallelized.

@ti.kernel
def _event_mv_prob_homo_gpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col] != 0.:
            index = i & 31
            i_row = step * index - 1
            end = ti.min(i_row + step, num_row)
            key = lfsr88_key(seed0 + i)
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_row += inc
            while i_row < end:
                out[i_row] += weight0
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_homo_outdim_parallel_gpu(
    events: ti.types.ndarray(ndim=1),
    weight: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    weight0 = weight[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_row * 32):
        i_row = i >> 5
        index = i & 31
        i_col = step * index - 1
        end_col = ti.min(i_col + step, num_col)
        r = 0.
        key = lfsr88_key(seed0 + i)
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc
        while i_col < end_col:
            r += weight0 * events[i_col]  # TODO: speed comparison with if else
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] += r  # TODO: warp-level reduction


def _event_mv_prob_homo_jvp_events(
    evt_dot, events, weight, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_homo(evt_dot, weight, clen, seed,
                            shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _event_mv_prob_homo_jvp_weight(
    w_dot, events, weight, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_homo(events, w_dot, clen, seed,
                            shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _event_checking(vector, clen, seed, shape, outdim_parallel, transpose, *weights):
    assert _get_dtype(vector) in [jnp.bool_, jnp.float16, jnp.float32, jnp.float64]
    return _general_checking(vector, clen, seed, shape, outdim_parallel, transpose, *weights)


def raw_event_mv_prob_homo(
    events: jax.Array,
    weight: jax.Array,  # vector with size 1
    conn_len: jax.Array,  # vector with size 1
    seed: jax.Array,  # vector with size 1
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    mat_shape, out_shape = _event_checking(events, conn_len, seed, shape, outdim_parallel, transpose, weight)

    if outdim_parallel:
        if events.dtype == jnp.bool_:
            prim = _event_mv_prob_homo_outdim_parallel_bool_p
        else:
            prim = _event_mv_prob_homo_outdim_parallel_p
    else:
        if events.dtype == jnp.bool_:
            prim = _event_mv_prob_homo_bool_p
        else:
            prim = _event_mv_prob_homo_p

    return prim(events,
                weight,
                conn_len,
                seed,
                outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=weight.dtype)],
                shape=mat_shape,
                transpose=transpose,
                outdim_parallel=outdim_parallel)


def _define_event_mv_prob_homo_prim(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    prim.defjvp(_event_mv_prob_homo_jvp_events,
                _event_mv_prob_homo_jvp_weight,
                None,
                None)
    prim.def_transpose_rule(_mv_prob_homo_transpose)
    return prim


# outdim_parallel = True, events.dtype = jnp.bool_
_event_mv_prob_homo_outdim_parallel_bool_p = _define_event_mv_prob_homo_prim(
    cpu_kernel=_event_mv_prob_homo_outdim_parallel_bool_cpu,
    gpu_kernel=_event_mv_prob_homo_outdim_parallel_bool_gpu
)

# outdim_parallel = False, events.dtype = jnp.bool_
_event_mv_prob_homo_bool_p = _define_event_mv_prob_homo_prim(
    cpu_kernel=_event_mv_prob_homo_bool_cpu,
    gpu_kernel=_event_mv_prob_homo_bool_gpu
)

# outdim_parallel = True, events.dtype != jnp.bool_
_event_mv_prob_homo_outdim_parallel_p = _define_event_mv_prob_homo_prim(
    cpu_kernel=_event_mv_prob_homo_outdim_parallel_cpu,
    gpu_kernel=_event_mv_prob_homo_outdim_parallel_gpu
)

# outdim_parallel = False, events.dtype != jnp.bool_
_event_mv_prob_homo_p = _define_event_mv_prob_homo_prim(
    cpu_kernel=_event_mv_prob_homo_cpu,
    gpu_kernel=_event_mv_prob_homo_gpu
)


@ti.kernel
def _event_mv_prob_uniform_bool_cpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
        if events[i_col]:
            key = lfsr88_key(seed0 + i_col)
            key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
            while i_row < num_row:
                key, row_v = lfsr88_uniform(key, w_min0, w_max0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_uniform_outdim_parallel_bool_cpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
        r = 0.
        key = lfsr88_key(seed0 + i_row)
        key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
        while i_col < num_col:
            key, row_v = lfsr88_uniform(key, w_min0, w_max0)
            if events[i_col]:
                r += row_v
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] = r


@ti.kernel
def _event_mv_prob_uniform_bool_gpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col]:
            index = i & 31
            i_row = step * index - 1
            end = ti.min(i_row + step, num_row)
            key = lfsr88_key(seed0 + i)
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_row += inc
            while i_row < end:
                key, row_v = lfsr88_uniform(key, w_min0, w_max0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_uniform_outdim_parallel_bool_gpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.u32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_row * 32):
        i_row = i >> 5
        index = i & 31
        i_col = step * index - 1
        end_col = ti.min(i_col + step, num_col)
        r = 0.
        key = lfsr88_key(seed0 + i)
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc
        while i_col < end_col:
            key, row_v = lfsr88_uniform(key, w_min0, w_max0)
            r += row_v * events[i_col]  # TODO: speed comparison without if else
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] += r  # TODO: warp-level reduction


@ti.kernel
def _event_mv_prob_uniform_cpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
        if events[i_col] != 0.:
            key = lfsr88_key(seed0 + i_col)
            key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
            while i_row < num_row:
                key, row_v = lfsr88_uniform(key, w_min0, w_max0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_uniform_outdim_parallel_cpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
        r = 0.
        key = lfsr88_key(seed0 + i_row)
        key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
        while i_col < num_col:
            key, row_v = lfsr88_uniform(key, w_min0, w_max0)
            if events[i_col] != 0.:
                r += row_v
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] = r  # TODO: warp-level reduction


@ti.kernel
def _event_mv_prob_uniform_gpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col] != 0.:
            index = i & 31
            i_row = step * index - 1
            end = ti.min(i_row + step, num_row)
            key = lfsr88_key(seed0 + i)
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_row += inc
            while i_row < end:
                key, row_v = lfsr88_uniform(key, w_min0, w_max0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_uniform_outdim_parallel_gpu(
    events: ti.types.ndarray(ndim=1),
    w_min: ti.types.ndarray(ndim=1),
    w_max: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_min0 = w_min[0]
    w_max0 = w_max[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_row * 32):
        i_row = i >> 5
        index = i & 31
        i_col = step * index - 1
        end_col = ti.min(i_col + step, num_col)
        r = 0.
        key = lfsr88_key(seed0 + i)
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc
        while i_col < end_col:
            key, row_v = lfsr88_uniform(key, w_min0, w_max0)
            r += row_v * events[i_col]  # TODO: speed comparison with if else
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] += r  # TODO: warp-level reduction


def _event_mv_prob_uniform_jvp_events(
    evt_dot, events, w_low, w_high, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_uniform(evt_dot, w_low, w_high, clen, seed,
                               shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _event_mv_prob_uniform_jvp_w_low(
    w_dot, events, w_low, w_high, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_uniform(events, w_dot, w_high, clen, seed,
                               shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _event_mv_prob_uniform_jvp_w_high(
    w_dot, events, w_low, w_high, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_uniform(events, w_low, w_dot, clen, seed,
                               shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def raw_event_mv_prob_uniform(
    events: jax.Array,
    w_low: jax.Array,  # vector with size 1
    w_high: jax.Array,  # vector with size 1
    conn_len: jax.Array,  # vector with size 1
    seed: jax.Array,  # vector with size 1
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    mat_shape, out_shape = _event_checking(events, conn_len, seed, shape, outdim_parallel, transpose, w_low, w_high)

    if outdim_parallel:
        if events.dtype == jnp.bool_:
            prim = _event_mv_prob_uniform_outdim_parallel_bool_p
        else:
            prim = _event_mv_prob_uniform_outdim_parallel_p
    else:
        if events.dtype == jnp.bool_:
            prim = _event_mv_prob_uniform_bool_p
        else:
            prim = _event_mv_prob_uniform_p

    return prim(events,
                w_low,
                w_high,
                conn_len,
                seed,
                outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=w_low.dtype)],
                shape=mat_shape,
                transpose=transpose,
                outdim_parallel=outdim_parallel)


def _define_event_mv_prob_uniform_prim(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    prim.defjvp(_event_mv_prob_uniform_jvp_events,
                _event_mv_prob_uniform_jvp_w_low,
                _event_mv_prob_uniform_jvp_w_high,
                None,
                None)
    prim.def_transpose_rule(_mv_prob_uniform_transpose)
    return prim


# outdim_parallel = True, events.dtype = jnp.bool_
_event_mv_prob_uniform_outdim_parallel_bool_p = _define_event_mv_prob_uniform_prim(
    cpu_kernel=_event_mv_prob_uniform_outdim_parallel_bool_cpu,
    gpu_kernel=_event_mv_prob_uniform_outdim_parallel_bool_gpu
)

# outdim_parallel = False, events.dtype = jnp.bool_
_event_mv_prob_uniform_bool_p = _define_event_mv_prob_uniform_prim(
    cpu_kernel=_event_mv_prob_uniform_bool_cpu,
    gpu_kernel=_event_mv_prob_uniform_bool_gpu
)

# outdim_parallel = True, events.dtype != jnp.bool_
_event_mv_prob_uniform_outdim_parallel_p = _define_event_mv_prob_uniform_prim(
    cpu_kernel=_event_mv_prob_uniform_outdim_parallel_cpu,
    gpu_kernel=_event_mv_prob_uniform_outdim_parallel_gpu
)

# outdim_parallel = False, events.dtype != jnp.bool_
_event_mv_prob_uniform_p = _define_event_mv_prob_uniform_prim(
    cpu_kernel=_event_mv_prob_uniform_cpu,
    gpu_kernel=_event_mv_prob_uniform_gpu
)


@ti.kernel
def _event_mv_prob_normal_bool_cpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
        if events[i_col]:
            key = lfsr88_key(seed0 + i_col)
            key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
            while i_row < num_row:
                key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_normal_outdim_parallel_bool_cpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
        r = 0.
        key = lfsr88_key(seed0 + i_row)
        key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
        while i_col < num_col:
            key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
            if events[i_col]:
                r += row_v
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] = r


@ti.kernel
def _event_mv_prob_normal_bool_gpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col]:
            index = i & 31
            i_row = step * index - 1
            end = ti.min(i_row + step, num_row)
            key = lfsr88_key(seed0 + i)
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_row += inc
            while i_row < end:
                key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_normal_outdim_parallel_bool_gpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.u32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_row * 32):
        i_row = i >> 5
        index = i & 31
        i_col = step * index - 1
        end_col = ti.min(i_col + step, num_col)
        r = 0.
        key = lfsr88_key(seed0 + i)
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc
        while i_col < end_col:
            key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
            r += row_v * events[i_col]  # TODO: speed comparison without if else
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] += r  # TODO: warp-level reduction


@ti.kernel
def _event_mv_prob_normal_cpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_col in range(num_col):
        if events[i_col] != 0.:
            key = lfsr88_key(seed0 + i_col)
            key, i_row = lfsr88_random_integers(key, 0, clen0 - 1)
            while i_row < num_row:
                key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_normal_outdim_parallel_cpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]

    for i_row in range(num_row):
        r = 0.
        key = lfsr88_key(seed0 + i_row)
        key, i_col = lfsr88_random_integers(key, 0, clen0 - 1)
        while i_col < num_col:
            key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
            if events[i_col] != 0.:
                r += row_v
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] = r


@ti.kernel
def _event_mv_prob_normal_gpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_col * 32):
        i_col = i >> 5
        if events[i_col] != 0.:
            index = i & 31
            i_row = step * index - 1
            end = ti.min(i_row + step, num_row)
            key = lfsr88_key(seed0 + i)
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_row += inc
            while i_row < end:
                key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
                out[i_row] += row_v
                key, inc = lfsr88_random_integers(key, 1, clen0)
                i_row += inc


@ti.kernel
def _event_mv_prob_normal_outdim_parallel_gpu(
    events: ti.types.ndarray(ndim=1),
    w_mu: ti.types.ndarray(ndim=1),
    w_sigma: ti.types.ndarray(ndim=1),
    clen: ti.types.ndarray(ndim=1),
    seed: ti.types.ndarray(ndim=1),
    out: ti.types.ndarray(ndim=1)
):
    num_row = out.shape[0]
    num_col = events.shape[0]
    w_mu0 = w_mu[0]
    w_sigma0 = w_sigma[0]
    clen0 = clen[0]
    seed0 = seed[0]
    step = ti.uint32(ti.max((num_row + 1) >> 5, 1))

    for i in range(num_row * 32):
        i_row = i >> 5
        index = i & 31
        i_col = step * index - 1
        end_col = ti.min(i_col + step, num_col)
        r = 0.
        key = lfsr88_key(seed0 + i)
        key, inc = lfsr88_random_integers(key, 1, clen0)
        i_col += inc
        while i_col < end_col:
            key, row_v = lfsr88_normal(key, w_mu0, w_sigma0)
            r += row_v * events[i_col]  # TODO: speed comparison with if else
            key, inc = lfsr88_random_integers(key, 1, clen0)
            i_col += inc
        out[i_row] += r  # TODO: warp-level reduction


def _event_mv_prob_normal_jvp_events(
    evt_dot, events, w_mu, w_sigma, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_normal(evt_dot, w_mu, w_sigma, clen, seed,
                              shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _event_mv_prob_normal_jvp_w_mu(
    w_dot, events, w_mu, w_sigma, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_normal(events, w_dot, w_sigma, clen, seed,
                              shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def _event_mv_prob_normal_jvp_w_sigma(
    w_dot, events, w_mu, w_sigma, clen, seed, *, outs, shape, transpose, outdim_parallel
):
    shape = _reverse(shape) if transpose else shape
    return raw_mv_prob_normal(events, w_mu, w_dot, clen, seed,
                              shape=shape, transpose=transpose, outdim_parallel=outdim_parallel)


def raw_event_mv_prob_normal(
    events: jax.Array,
    w_mu: jax.Array,  # vector with size 1
    w_sigma: jax.Array,  # vector with size 1
    conn_len: jax.Array,  # vector with size 1
    seed: jax.Array,  # vector with size 1
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    mat_shape, out_shape = _event_checking(events, conn_len, seed, shape, outdim_parallel, transpose, w_mu, w_sigma)

    if outdim_parallel:
        if events.dtype == jnp.bool_:
            prim = _event_mv_prob_normal_outdim_parallel_bool_p
        else:
            prim = _event_mv_prob_normal_outdim_parallel_p
    else:
        if events.dtype == jnp.bool_:
            prim = _event_mv_prob_normal_bool_p
        else:
            prim = _event_mv_prob_normal_p

    return prim(events,
                w_mu,
                w_sigma,
                conn_len,
                seed,
                outs=[jax.ShapeDtypeStruct(shape=out_shape, dtype=w_mu.dtype)],
                shape=mat_shape,
                transpose=transpose,
                outdim_parallel=outdim_parallel)


def _define_event_mv_prob_normal_prim(cpu_kernel, gpu_kernel):
    prim = XLACustomOp(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)
    prim.defjvp(_event_mv_prob_normal_jvp_events,
                _event_mv_prob_normal_jvp_w_mu,
                _event_mv_prob_normal_jvp_w_sigma,
                None,
                None)
    prim.def_transpose_rule(_mv_prob_normal_transpose)
    return prim


# outdim_parallel = True, events.dtype = jnp.bool_
_event_mv_prob_normal_outdim_parallel_bool_p = _define_event_mv_prob_normal_prim(
    cpu_kernel=_event_mv_prob_normal_outdim_parallel_bool_cpu,
    gpu_kernel=_event_mv_prob_normal_outdim_parallel_bool_gpu
)

# outdim_parallel = False, events.dtype = jnp.bool_
_event_mv_prob_normal_bool_p = _define_event_mv_prob_normal_prim(
    cpu_kernel=_event_mv_prob_normal_bool_cpu,
    gpu_kernel=_event_mv_prob_normal_bool_gpu
)

# outdim_parallel = True, events.dtype != jnp.bool_
_event_mv_prob_normal_outdim_parallel_p = _define_event_mv_prob_normal_prim(
    cpu_kernel=_event_mv_prob_normal_outdim_parallel_cpu,
    gpu_kernel=_event_mv_prob_normal_outdim_parallel_gpu
)

# outdim_parallel = False, events.dtype != jnp.bool_
_event_mv_prob_normal_p = _define_event_mv_prob_normal_prim(
    cpu_kernel=_event_mv_prob_normal_cpu,
    gpu_kernel=_event_mv_prob_normal_gpu
)
