``braintools`` documentation
============================

`braintools <https://github.com/brainpy/braintools>`_ implements the common toolboxes for brain dynamics programming (BDP).

----


Installation
^^^^^^^^^^^^

.. tab-set::

    .. tab-item:: CPU

       .. code-block:: bash

          pip install -U braintools[cpu]

    .. tab-item:: GPU (CUDA 11.0)

       .. code-block:: bash

          pip install -U braintools[cuda11] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    .. tab-item:: GPU (CUDA 12.0)

       .. code-block:: bash

          pip install -U braintools[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

    .. tab-item:: TPU

       .. code-block:: bash

          pip install -U braintools[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


----


See also the BDP ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


- `brainpy <https://github.com/brainpy/BrainPy>`_: The solution for the general-purpose brain dynamics programming.

- `braincore <https://github.com/brainpy/braincore>`_: The core system for the next generation of BrainPy framework.

- `braintools <https://github.com/brainpy/braintools>`_: The tools for the brain dynamics simulation and analysis.

- `brainscale <https://github.com/brainpy/brainscale>`_: The scalable online learning for biological spiking neural networks.



.. toctree::
   :hidden:
   :maxdepth: 2

   api.rst

