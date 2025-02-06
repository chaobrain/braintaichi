# Leveraging Taichi Lang to Customize Brain Dynamics Operators


<p align="center">
  	<img alt="Header image of braintaichi." src="https://github.com/chaoming0625/braintaichi/blob/main/docs/_static/braintaichi.png" width=50%>
</p> 



<p align="center">
	<a href="https://pypi.org/project/braintaichi/"><img alt="Supported Python Version" src="https://img.shields.io/pypi/pyversions/braintaichi"></a>
	<a href="https://github.com/chaoming0625/braintaichi/blob/main/LICENSE"><img alt="LICENSE" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
    <a href='https://braintaichi.readthedocs.io/en/latest/?badge=latest'>
        <img src='https://readthedocs.org/projects/braintaichi/badge/?version=latest' alt='Documentation Status' />
    </a>  	
    <a href="https://badge.fury.io/py/braintaichi"><img alt="PyPI version" src="https://badge.fury.io/py/braintaichi.svg"></a>
    <a href="https://github.com/chaoming0625/braintaichi/actions/workflows/CI.yml"><img alt="Continuous Integration" src="https://github.com/chaoming0625/braintaichi/actions/workflows/CI.yml/badge.svg"></a>
</p>


[``braintaichi``](https://github.com/chaoming0625/braintaichi) leverages Taichi Lang to customize brain dynamics operators.



## Quick Start

```python


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

out = kernel(
    jnp.array(csr.data),
    jnp.array(csr.indices),
    jnp.array(csr.indptr),
    events,
    outs=[jax.ShapeDtypeStruct([10], dtype=jnp.float32)]
)
print(out)
```


## Installation

You can install ``braintaichi`` via pip:

```bash
pip install braintaichi --upgrade
```

## Documentation

The official documentation is hosted on Read the Docs: [https://braintaichi.readthedocs.io](https://braintaichi.readthedocs.io)


## See also the BDP ecosystem

We are building the brain dynamics programming ecosystem: https://ecosystem-for-brain-dynamics.readthedocs.io/


## Citation

If you think `braintaichi` is significant in your work, please consider to cite the following pubilication:

```bibtex

@inproceedings{wang2024brainpy,
    title={A differentiable brain simulator bridging brain simulation and brain-inspired computing},
    author={Wang, Chaoming and Zhang, Tianqiu and He, Sichao and Gu, Hongyaoxing and Li, Shangyang and Wu, Si},
    booktitle={The Twelfth International Conference on Learning Representations},
    year={2024}
}
```


