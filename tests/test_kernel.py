# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


import taichi as ti

import braintaichi as bti


# define the custom kernel

@ti.kernel
def transpose_bool_homo_kernel(
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


kernel = bti.XLACustomOp(
    cpu_kernel=transpose_bool_homo_kernel,
    gpu_kernel=transpose_bool_homo_kernel,
)

# run with the sample data

import numpy as np
import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix

csr = csr_matrix((np.random.rand(10, 10) < 0.5).astype(float))
events = np.random.rand(10) < 0.5

print(
    kernel(jnp.array(csr.data),
           jnp.array(csr.indices),
           jnp.array(csr.indptr),
           events,
           outs=[jax.ShapeDtypeStruct([10], dtype=jnp.float32)])
)
