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

from jax.core import Primitive

__all__ = [
    'defjvp',
]


def defjvp(primitive: Primitive, *jvp_rules):
    """
    Define JVP rules for any JAX primitive.

    This function is similar to ``jax.interpreters.ad.defjvp``.
    However, this JAX function only supports primitives with ``multiple_results=False``.
    ``braintaichi.defjvp`` enables to define the independent JVP rule for
    each input parameter no matter ``multiple_results=False/True``.

    Args:
      primitive: Primitive, XLACustomOp.
      *jvp_rules: The JVP translation rule for each primal.
    """
    from brainstate.event._xla_custom_op import defjvp as defjvp_custom_op
    defjvp_custom_op(primitive, *jvp_rules)
