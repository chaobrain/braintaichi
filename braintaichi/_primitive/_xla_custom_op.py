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

from functools import partial
from typing import Callable, Sequence, Tuple, Protocol, Optional, Union

import jax
import numpy as np
from jax.interpreters import xla, batching, ad, mlir

from ._ad_support import defjvp
from ._batch_utils import register_general_batching
from ._mlir_translation_rule import (
    register_taichi_aot_mlir_cpu_translation_rule,  # noqa
    register_taichi_aot_mlir_gpu_translation_rule,  # noqa
)

__all__ = [
    'XLACustomOp',
]


class ShapeDtype(Protocol):

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def dtype(self) -> np.dtype:
        ...


_idx = 0


class XLACustomOp:
    """
    Creating a XLA custom call operator.

    Args:
      cpu_kernel: Callable. The function defines the computation on CPU backend.
      gpu_kernel: Callable. The function defines the computation on GPU backend.
      batching_translation: Callable. The batching translation rule of JAX.
      jvp_translation: Callable. The JVP translation rule of JAX.
      transpose_translation: Callable. The transpose translation rule of JAX.
      name: str. The primitive name.
    """

    __module__ = 'braintaichi'

    def __init__(
        self,
        cpu_kernel: Callable = None,
        gpu_kernel: Union[Callable, str] = None,
        batching_translation: Callable = None,
        jvp_translation: Callable = None,
        transpose_translation: Callable = None,
        name: str = None,
    ):
        # set cpu_kernel and gpu_kernel
        self.cpu_kernel = cpu_kernel
        self.gpu_kernel = gpu_kernel

        # primitive
        if name is None:
            global _idx
            name = f'braintaichi_custom_op_{_idx}'
            _idx += 1
        self.primitive = jax.core.Primitive(name)
        self.primitive.multiple_results = True

        # abstract evaluation
        self.primitive.def_abstract_eval(_abstract_eval)
        self.primitive.def_impl(partial(xla.apply_primitive, self.primitive))

        # cpu function
        if cpu_kernel is not None:
            register_taichi_aot_mlir_cpu_translation_rule(self.primitive, cpu_kernel)

        # gpu function
        if gpu_kernel is not None:
            register_taichi_aot_mlir_gpu_translation_rule(self.primitive, gpu_kernel)

        # batching rule
        if batching_translation is None:
            register_general_batching(self.primitive)
        else:
            batching.primitive_batchers[self.primitive] = batching_translation

        # jvp rule
        if jvp_translation is not None:
            ad.primitive_jvps[self.primitive] = jvp_translation

        # transpose rule
        if transpose_translation is not None:
            ad.primitive_transposes[self.primitive] = transpose_translation

    def __call__(self, *ins, outs: Optional[Sequence[ShapeDtype]], **kwargs):
        outs = tuple([_transform_to_shapedarray(o) for o in outs])
        ins = jax.tree.map(jax.numpy.asarray, ins)
        return self.primitive.bind(*ins, outs=outs, **kwargs)

    def def_abstract_eval(self, fun):
        """Define the abstract evaluation function.

        Args:
          fun: The abstract evaluation function.
        """
        self.primitive.def_abstract_eval(fun)

    def def_batching_rule(self, fun):
        """Define the batching rule.

        Args:
          fun: The batching rule.
        """
        batching.primitive_batchers[self.primitive] = fun

    def def_jvp_rule(self, fun):
        """Define the JVP rule.

        Args:
          fun: The JVP rule.
        """
        ad.primitive_jvps[self.primitive] = fun

    def defjvp(self, *jvp_rules):
        """
        Define the JVP rule. Similar to ``jax.interpreters.ad.defjvp``,
        but supports the Primitive with multiple results.

        Args:
          jvp_rules: The JVP rules.
        """
        defjvp(self.primitive, *jvp_rules)

    def def_transpose_rule(self, fun):
        """Define the transpose rule.

        Args:
          fun: The transpose rule.
        """
        ad.primitive_transposes[self.primitive] = fun

    def def_xla_translation(self, platform, fun):
        """Define the XLA translation rule.

        Args:
          platform: str. The computing platform.
          fun: The XLA translation rule.
        """
        xla.backend_specific_translations[platform][self.primitive] = fun

    def def_mlir_lowering(self, platform, fun):
        """Define the MLIR lowering rule.

        Args:
          platform: str. The computing platform.
          fun: The lowering rule.
        """
        mlir.register_lowering(self.primitive, fun, platform)


def _abstract_eval(*args, **kwargs):
    return [
        jax.core.ShapedArray(out_shape.shape, out_shape.dtype)
        for out_shape in kwargs['outs']
    ]


def _transform_to_shapedarray(a):
    return jax.core.ShapedArray(a.shape, a.dtype)
