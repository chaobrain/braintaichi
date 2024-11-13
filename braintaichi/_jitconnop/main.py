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

from typing import Tuple, Optional

import jax
import numpy as np
from jax import numpy as jnp

from braintaichi._misc import set_module_as
from .csrmv import raw_mv_prob_homo, raw_mv_prob_uniform, raw_mv_prob_normal
from .event_csrmv import raw_event_mv_prob_homo, raw_event_mv_prob_uniform, raw_event_mv_prob_normal

__all__ = [
    'jitc_mv_prob_homo',
    'jitc_mv_prob_uniform',
    'jitc_mv_prob_normal',
    'jitc_event_mv_prob_homo',
    'jitc_event_mv_prob_uniform',
    'jitc_event_mv_prob_normal',
]


@set_module_as('braintaichi')
def jitc_mv_prob_homo(
    vector: jax.typing.ArrayLike,
    weight: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a scalar `weight` at each position.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

       This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

       Note that the just-in-time generated :math:`M` (`transpose=False`) is
       different from the generated :math:`M^T` (`transpose=True`).

       If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
       matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
       the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    vector: Array, ndarray
      The vector.
    weight: float
      The value of the random matrix.
    conn_prob: float
      The connection probability.
    shape: tuple of int
      The matrix shape.
    seed: int
      The random number generation seed.
    transpose: bool
      Transpose the random matrix or not.
    outdim_parallel: bool
      Perform the parallel random generations along the out dimension or not.
      It can be used to set the just-in-time generated :math:M^T: is the same
      as the just-in-time generated :math:`M` when ``transpose=True``.

    Returns
    -------
    out: Array, ndarray
      The output of :math:`y = M @ v`.
    """
    vector = jnp.asarray(vector)
    if isinstance(weight, float):
        weight = jnp.asarray(weight, dtype=vector.dtype)
    weight = jnp.atleast_1d(jnp.asarray(weight))
    conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
    clen = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.asarray(seed, dtype=jnp.uint32)
    seed = jnp.atleast_1d(seed)
    return raw_mv_prob_homo(vector, weight, clen, seed, shape=shape,
                            transpose=transpose, outdim_parallel=outdim_parallel)[0]


@set_module_as('braintaichi')
def jitc_mv_prob_uniform(
    vector: jax.Array,
    w_low: float,
    w_high: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a uniform distribution for its value.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

       This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

       Note that the just-in-time generated :math:`M` (`transpose=False`) is
       different from the generated :math:`M^T` (`transpose=True`).

       If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
       matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
       the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    vector: Array, ndarray
      The vector.
    w_low: float
      Lower boundary of the output interval.
    w_high: float
      Upper boundary of the output interval.
    conn_prob: float
      The connection probability.
    shape: tuple of int
      The matrix shape.
    seed: int
      The random number generation seed.
    transpose: bool
      Transpose the random matrix or not.
    outdim_parallel: bool
      Perform the parallel random generations along the out dimension or not.
      It can be used to set the just-in-time generated :math:M^T: is the same
      as the just-in-time generated :math:`M` when ``transpose=True``.

    Returns
    -------
    out: Array, ndarray
      The output of :math:`y = M @ v`.
    """
    vector = jnp.asarray(vector)
    if isinstance(w_low, float): w_low = jnp.asarray(w_low, dtype=vector.dtype)
    if isinstance(w_high, float): w_high = jnp.asarray(w_high, dtype=vector.dtype)
    w_low = jnp.atleast_1d(jnp.asarray(w_low))
    w_high = jnp.atleast_1d(jnp.asarray(w_high))
    conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
    conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
    return raw_mv_prob_uniform(vector, w_low, w_high, conn_len, seed, shape=shape,
                               transpose=transpose, outdim_parallel=outdim_parallel)[0]


@set_module_as('braintaichi')
def jitc_mv_prob_normal(
    vector: jax.Array,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    r"""Perform the :math:`y=M@v` operation,
    where :math:`M` is just-in-time randomly generated with a normal distribution for its value.

    This operator support ``jit()``, ``vmap()``, ``grad()`` and ``pmap()`` etc. transformations
    on CPU and GPU devices.

    .. warning::

       This API may change in the future.

    In this operation, :math:`M` is the random matrix with a connection probability
    `conn_prob`, and at each connection the value is the same scalar `weight`.

    When ``transpose=True``, we perform an operation of :math:`y=M^T@v`.

    .. note::

       Note that the just-in-time generated :math:`M` (`transpose=False`) is
       different from the generated :math:`M^T` (`transpose=True`).

       If you pursue the same :math:`M` and :math:`M^T` when performing the just-in-time
       matrix generation, you should set ``outdim_parallel=True``, with the sacrifice of
       the speed compared with ``outdim_parallel=False``.

    Parameters
    ----------
    vector: Array, ndarray
      The vector.
    w_mu: float
      Mean (centre) of the distribution.
    w_sigma: float
      Standard deviation (spread or “width”) of the distribution. Must be non-negative.
    conn_prob: float
      The connection probability.
    shape: tuple of int
      The matrix shape.
    seed: int
      The random number generation seed.
    transpose: bool
      Transpose the random matrix or not.
    outdim_parallel: bool
      Perform the parallel random generations along the out dimension or not.
      It can be used to set the just-in-time generated :math:M^T: is the same
      as the just-in-time generated :math:`M` when ``transpose=True``.

    Returns
    -------
    out: Array, ndarray
      The output of :math:`y = M @ v`.
    """
    vector = jnp.asarray(vector)
    if isinstance(w_mu, float): w_mu = jnp.asarray(w_mu, dtype=vector.dtype)
    if isinstance(w_sigma, float): w_sigma = jnp.asarray(w_sigma, dtype=vector.dtype)
    w_mu = jnp.atleast_1d(jnp.asarray(w_mu))
    w_sigma = jnp.atleast_1d(jnp.asarray(w_sigma))
    conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
    conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
    return raw_mv_prob_normal(vector, w_mu, w_sigma, conn_len, seed, shape=shape,
                              transpose=transpose, outdim_parallel=outdim_parallel)[0]


@set_module_as('braintaichi')
def jitc_event_mv_prob_homo(
    events: jax.Array,
    weight: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    events = jnp.asarray(events)
    weight = jnp.asarray(weight)
    if jnp.ndim(weight) < 1:
        weight = jnp.expand_dims(weight, axis=0)
    conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
    conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
    return raw_event_mv_prob_homo(events, weight, conn_len, seed,
                                  shape=shape,
                                  transpose=transpose,
                                  outdim_parallel=outdim_parallel)[0]


jitc_event_mv_prob_homo.__doc__ = jitc_mv_prob_homo.__doc__


@set_module_as('braintaichi')
def jitc_event_mv_prob_uniform(
    events: jax.Array,
    w_low: float,
    w_high: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    events = jnp.asarray(events)
    if isinstance(w_low, float): w_low = jnp.asarray(w_low)
    if isinstance(w_high, float): w_high = jnp.asarray(w_high)
    w_low = jnp.atleast_1d(jnp.asarray(w_low))
    w_high = jnp.atleast_1d(jnp.asarray(w_high))
    conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
    conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
    return raw_event_mv_prob_uniform(events, w_low, w_high, conn_len, seed, shape=shape,
                                     transpose=transpose, outdim_parallel=outdim_parallel)[0]


jitc_event_mv_prob_uniform.__doc__ = jitc_mv_prob_uniform.__doc__


@set_module_as('braintaichi')
def jitc_event_mv_prob_normal(
    events: jax.Array,
    w_mu: float,
    w_sigma: float,
    conn_prob: float,
    seed: Optional[int] = None,
    *,
    shape: Tuple[int, int],
    transpose: bool = False,
    outdim_parallel: bool = True,
) -> jax.Array:
    events = jnp.asarray(events)
    if isinstance(w_mu, float): w_mu = jnp.asarray(w_mu)
    if isinstance(w_sigma, float): w_sigma = jnp.asarray(w_sigma)
    w_mu = jnp.atleast_1d(jnp.asarray(w_mu))
    w_sigma = jnp.atleast_1d(jnp.asarray(w_sigma))
    conn_len = jnp.ceil(1 / conn_prob) * 2 - 1
    conn_len = jnp.asarray(jnp.atleast_1d(conn_len), dtype=jnp.int32)
    if seed is None:
        with jax.ensure_compile_time_eval():
            seed = np.random.randint(0, int(1e8), 1)
    seed = jnp.atleast_1d(jnp.asarray(seed, dtype=jnp.uint32))
    return raw_event_mv_prob_normal(events, w_mu, w_sigma, conn_len, seed, shape=shape,
                                    transpose=transpose, outdim_parallel=outdim_parallel)[0]


jitc_event_mv_prob_normal.__doc__ = jitc_mv_prob_normal.__doc__
