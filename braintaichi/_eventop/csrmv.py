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

"""

Key points for the operator customization:

1. `index` has two kinds of types: int32, int64
2. `data` has two kinds of types: float32, float64
3. `events` has three kinds of types: bool (True or False), float32, float64

"""

from typing import Union, Tuple, Callable

import jax
import jax.numpy as jnp
import taichi as ti
from jax.interpreters import ad

from braintaichi._primitive._xla_custom_op import XLACustomOp
from braintaichi._sparseop.csrmv import raw_csrmv_taichi as normal_csrmv_taichi
from braintaichi._sparseop.utils import csr_to_coo


def event_csrmv_taichi(
    data: Union[float, jax.Array],
    indices: jax.Array,
    indptr: jax.Array,
    events: jax.Array,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    float_as_event: bool = True,
):
    homo = jnp.size(data) == 1

    if transpose:
        if events.dtype == jnp.bool_:
            if homo:
                prim = bool_transpose_homo_p
            else:
                prim = bool_transpose_heter_p
        else:

            if homo:
                if float_as_event:
                    prim = float_transpose_homo_p
                else:
                    prim = bfloat_transpose_homo_p
            else:
                if float_as_event:
                    prim = float_transpose_heter_p
                else:
                    prim = bfloat_transpose_heter_p

    else:
        if events.dtype == jnp.bool_:
            if homo:
                prim = bool_homo_p
            else:
                prim = bool_heter_p
        else:

            if homo:
                if float_as_event:
                    prim = float_homo_p
                else:
                    prim = bfloat_homo_p
            else:
                if float_as_event:
                    prim = float_heter_p
                else:
                    prim = bfloat_heter_p

    # computing
    return prim(
        data,
        indices,
        indptr,
        events,
        outs=[jax.ShapeDtypeStruct(shape=(shape[1] if transpose else shape[0],), dtype=data.dtype)],
        transpose=transpose,
        shape=shape,
        float_as_event=float_as_event
    )


class EventCSRMatVec(XLACustomOp):
    def __init__(
        self,
        cpu_kernel: Callable,
        gpu_kernel: Callable,
    ):
        super().__init__(cpu_kernel=cpu_kernel, gpu_kernel=gpu_kernel)

        self.defjvp(self.jvp_weights, None, None, self.jvp_events)
        self.def_transpose_rule(self.transpose_rule)

    def jvp_weights(
        self,
        val_dot,
        weights,
        indices,
        indptr,
        events,
        *,
        outs,
        transpose,
        shape,
        float_as_event,
    ):
        return event_csrmv_taichi(
            val_dot,
            indices,
            indptr,
            events,
            shape=shape,
            transpose=transpose,
            float_as_event=float_as_event
        )

    def jvp_events(
        self,
        evt_dot,
        weights,
        indices,
        indptr,
        events,
        *,
        outs,
        transpose,
        shape,
        float_as_event
    ):
        return normal_csrmv_taichi(
            weights,
            indices,
            indptr,
            evt_dot,
            shape=shape,
            transpose=transpose,
        )

    def transpose_rule(
        self,
        ct,
        weights,
        indices,
        indptr,
        events,
        *,
        outs,
        transpose,
        shape,
        float_as_event
    ):
        if ad.is_undefined_primal(indices) or ad.is_undefined_primal(indptr):
            raise ValueError("Cannot transpose with respect to sparse indices.")
        if ad.is_undefined_primal(events):
            ct_events = normal_csrmv_taichi(
                weights,
                indices,
                indptr,
                ct[0],
                shape=shape,
                transpose=not transpose
            )[0]
            return weights, indices, indptr, (ad.Zero(events) if type(ct[0]) is ad.Zero else ct_events)
        else:
            if type(ct[0]) is ad.Zero:
                ct_values = ad.Zero(weights)
            else:
                if weights.aval.shape[0] == 1:  # scalar
                    ct_values = event_csrmv_taichi(
                        jnp.ones(1, dtype=weights.dtype),
                        indices,
                        indptr,
                        events,
                        shape=shape,
                        transpose=transpose,
                        float_as_event=float_as_event,
                    )[0]
                    ct_values = jnp.inner(ct[0], ct_values)
                else:  # heterogeneous values
                    row, col = csr_to_coo(indices, indptr)
                    ct_values = events[row] * ct[0][col] if transpose else events[col] * ct[0][row]
            return ct_values, indices, indptr, events


class BoolTransposeHomo(EventCSRMatVec):
    def __init__(self):
        @ti.kernel
        def transpose_bool_homo_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                if events[row_i]:
                    for j in range(indptr[row_i], indptr[row_i + 1]):
                        out[indices[j]] += value

        # 1. GPU kernels are different from the CPU ones, since the GPU kernels need
        #    to use warp-level parallelism to achieve the best performance.

        @ti.kernel
        def transpose_bool_homo_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                if events[row_i]:
                    j = indptr[row_i] + index
                    end_index = indptr[row_i + 1]
                    while j < end_index:
                        out[indices[j]] += value
                        j += 32

        super().__init__(
            cpu_kernel=transpose_bool_homo_cpu,
            gpu_kernel=transpose_bool_homo_gpu,
        )


class BFloatTransposeHome(EventCSRMatVec):
    def __init__(self):
        @ti.kernel
        def transpose_homo_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                if events[row_i] != 0.:
                    for j in range(indptr[row_i], indptr[row_i + 1]):
                        out[indices[j]] += value

        @ti.kernel
        def transpose_homo_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                if events[row_i] != 0.:
                    j = indptr[row_i] + index
                    end_index = indptr[row_i + 1]
                    while j < end_index:
                        out[indices[j]] += value
                        j += 32

        super().__init__(
            cpu_kernel=transpose_homo_cpu,
            gpu_kernel=transpose_homo_gpu,
        )


class FloatTransposeHome(EventCSRMatVec):
    def __init__(self):
        @ti.kernel
        def transpose_homo_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                event = events[row_i]
                if event != 0.:
                    for j in range(indptr[row_i], indptr[row_i + 1]):
                        out[indices[j]] += value * event

        @ti.kernel
        def transpose_homo_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                event = events[row_i]
                if event != 0.:
                    j = indptr[row_i] + index
                    end_index = indptr[row_i + 1]
                    while j < end_index:
                        out[indices[j]] += value * event
                        j += 32

        super().__init__(
            cpu_kernel=transpose_homo_cpu,
            gpu_kernel=transpose_homo_gpu,
        )


class BoolHomo(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def bool_homo_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            # ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                r = 0.
                for j in range(indptr[row_i], indptr[row_i + 1]):
                    if events[indices[j]]:
                        r += value
                out[row_i] = r

        # TODO
        # It is important to note that the following warp-based kernels
        # should be improved, since the atomic_add for each thread is not
        # very efficient. Instead, the warp-level reduction primitive
        # should be used.
        # see ``warp_reduce_sum()`` function in tifunc.py.
        # However, currently Taichi does not support general warp-level primitives.

        @ti.kernel
        def bool_homo_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                r = 0.
                j = indptr[row_i] + index
                end_index = indptr[row_i + 1]
                while j < end_index:
                    if events[indices[j]]:
                        r += value
                    j += 32
                out[row_i] += r  # TODO: warp-level primitive

        super().__init__(
            cpu_kernel=bool_homo_cpu,
            gpu_kernel=bool_homo_gpu,
        )


class BFloatHomo(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def homo_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            # ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                r = 0.
                for j in range(indptr[row_i], indptr[row_i + 1]):
                    if events[indices[j]] != 0.:
                        r += value
                out[row_i] = r

        @ti.kernel
        def homo_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                r = 0.
                j = indptr[row_i] + index
                end_index = indptr[row_i + 1]
                while j < end_index:
                    if events[indices[j]] != 0.:
                        r += value
                    j += 32
                out[row_i] += r  # TODO: warp-level primitive

        super().__init__(
            cpu_kernel=homo_cpu,
            gpu_kernel=homo_gpu,
        )


class FloatHomo(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def homo_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            # ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                r = 0.
                for j in range(indptr[row_i], indptr[row_i + 1]):
                    e = events[indices[j]]
                    if e != 0.:
                        r += value * e
                out[row_i] = r

        @ti.kernel
        def homo_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            value = values[0]
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                r = 0.
                j = indptr[row_i] + index
                end_index = indptr[row_i + 1]
                while j < end_index:
                    e = events[indices[j]]
                    if e != 0.:
                        r += value * e
                    j += 32
                out[row_i] += r  # TODO: warp-level primitive

        super().__init__(
            cpu_kernel=homo_cpu,
            gpu_kernel=homo_gpu,
        )


class BoolTransposeHeter(EventCSRMatVec):
    def __init__(self):

        # 1. The benchmarking shows that the performance of the following transpose
        #    kernels is maximized when using serialized mode
        # 2. Since our Taichi-JAX kernel does not support the non-differentiable/non-jittable
        #    arguments, we have to define each kernel separately when the
        #    non-differentiable/non-jittable arguments are different.

        @ti.kernel
        def transpose_bool_heter_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                if events[row_i]:
                    for j in range(indptr[row_i], indptr[row_i + 1]):
                        out[indices[j]] += values[j]

        @ti.kernel
        def transpose_bool_heter_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                if events[row_i]:
                    j = indptr[row_i] + index
                    end_index = indptr[row_i + 1]
                    while j < end_index:
                        out[indices[j]] += values[j]
                        j += 32

        super().__init__(
            cpu_kernel=transpose_bool_heter_cpu,
            gpu_kernel=transpose_bool_heter_gpu,
        )


class BFloatTransposeHeter(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def transpose_heter_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                if events[row_i] != 0.:
                    for j in range(indptr[row_i], indptr[row_i + 1]):
                        out[indices[j]] += values[j]

        @ti.kernel
        def transpose_heter_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                if events[row_i] != 0.:
                    j = indptr[row_i] + index
                    end_index = indptr[row_i + 1]
                    while j < end_index:
                        out[indices[j]] += values[j]
                        j += 32

        super().__init__(
            cpu_kernel=transpose_heter_cpu,
            gpu_kernel=transpose_heter_gpu
        )


class FloatTransposeHeter(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def transpose_heter_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                e = events[row_i]
                if e != 0.:
                    for j in range(indptr[row_i], indptr[row_i + 1]):
                        out[indices[j]] += values[j] * e

        @ti.kernel
        def transpose_heter_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                e = events[row_i]
                if e != 0.:
                    j = indptr[row_i] + index
                    end_index = indptr[row_i + 1]
                    while j < end_index:
                        out[indices[j]] += values[j] * e
                        j += 32

        super().__init__(
            cpu_kernel=transpose_heter_cpu,
            gpu_kernel=transpose_heter_gpu
        )


class BoolHeter(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def bool_heter_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            # ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                r = 0.
                for j in range(indptr[row_i], indptr[row_i + 1]):
                    if events[indices[j]]:
                        r += values[j]
                out[row_i] = r

        @ti.kernel
        def bool_heter_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                r = 0.
                j = indptr[row_i] + index
                end_index = indptr[row_i + 1]
                while j < end_index:
                    if events[indices[j]]:
                        r += values[j]
                    j += 32
                out[row_i] += r  # TODO: warp-level primitive

        super().__init__(
            cpu_kernel=bool_heter_cpu,
            gpu_kernel=bool_heter_gpu
        )


class BFloatHeter(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def heter_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            # ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                r = 0.
                for j in range(indptr[row_i], indptr[row_i + 1]):
                    if events[indices[j]] != 0.:
                        r += values[j]
                out[row_i] = r

        @ti.kernel
        def heter_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                r = 0.
                j = indptr[row_i] + index
                end_index = indptr[row_i + 1]
                while j < end_index:
                    if events[indices[j]] != 0.:
                        r += values[j]
                    j += 32
                out[row_i] += r  # TODO: warp-level primitive

        super().__init__(
            cpu_kernel=heter_cpu,
            gpu_kernel=heter_gpu,
        )


class FloatHeter(EventCSRMatVec):
    def __init__(self):

        @ti.kernel
        def heter_cpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            # ti.loop_config(serialize=True)
            for row_i in range(indptr.shape[0] - 1):
                r = 0.
                for j in range(indptr[row_i], indptr[row_i + 1]):
                    e = events[indices[j]]
                    if e != 0.:
                        r += values[j] * e
                out[row_i] = r

        @ti.kernel
        def heter_gpu(
            values: ti.types.ndarray(ndim=1),
            indices: ti.types.ndarray(ndim=1),
            indptr: ti.types.ndarray(ndim=1),
            events: ti.types.ndarray(ndim=1),
            out: ti.types.ndarray(ndim=1)
        ):
            for i in range((indptr.shape[0] - 1) * 32):
                row_i = i >> 5
                index = i & 31
                r = 0.
                j = indptr[row_i] + index
                end_index = indptr[row_i + 1]
                while j < end_index:
                    e = events[indices[j]]
                    if e != 0.:
                        r += values[j] * e
                    j += 32
                out[row_i] += r  # TODO: warp-level primitive

        super().__init__(
            cpu_kernel=heter_cpu,
            gpu_kernel=heter_gpu,
        )


bool_transpose_homo_p = BoolTransposeHomo()
bfloat_transpose_homo_p = BFloatTransposeHome()
float_transpose_homo_p = FloatTransposeHome()
bool_homo_p = BoolHomo
bfloat_homo_p = BFloatHomo()
float_homo_p = FloatHomo()
bool_transpose_heter_p = BoolTransposeHeter()
bfloat_transpose_heter_p = BFloatTransposeHeter()
float_transpose_heter_p = FloatTransposeHeter()
bool_heter_p = BoolHeter()
bfloat_heter_p = BFloatHeter()
float_heter_p = FloatHeter()
