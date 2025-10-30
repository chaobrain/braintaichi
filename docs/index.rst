``braintaichi`` documentation
=============================

`braintaichi <https://github.com/chaoming0625/braintaichi>`_  leverages `Taichi Lang <https://www.taichi-lang.org/>`_ to customize brain dynamics operators.

``braintaichi`` is a high-performance computing library designed for brain dynamics simulation. By combining the power of Taichi Lang's GPU acceleration capabilities with JAX's automatic differentiation framework, it enables researchers to efficiently implement and customize neural network operators for large-scale brain modeling. The library provides specialized operators for sparse connectivity patterns, event-driven computations, and just-in-time connectivity generation, making it ideal for simulating complex neural systems with millions of neurons and synapses.

**Key Features:**

- **High Performance**: Utilizes Taichi Lang for efficient GPU/CPU parallel computing
- **JAX Integration**: Seamless integration with JAX ecosystem for automatic differentiation
- **Flexible Operators**: Support for sparse, event-driven, and JIT connectivity operators
- **Easy Customization**: Simple API for defining custom brain dynamics kernels
- **Cross-platform**: Works on Linux, Windows, and macOS with CPU/GPU support

----


Installation
^^^^^^^^^^^^

.. code-block:: bash

   pip install -U braintaichi


----


See also the ecosystem
^^^^^^^^^^^^^^^^^^^^^^^^^^


We are building the `brain modeling ecosystem <https://brainmodeling.readthedocs.io/>`_.

----

.. toctree::
   :maxdepth: 1
   :caption: Tutorial
   :hidden:

   tutorial/quickstart.ipynb
   tutorial/braintaichi_intro.ipynb
   tutorial/complete_example.ipynb
   tutorial/advanced_optimization.ipynb


.. toctree::
   :maxdepth: 1
   :caption: API Documentation
   :hidden:

   apis/changelog.md
   apis/operator-registration.rst
   apis/sparse-operators.rst
   apis/event-operators.rst
   apis/jitconn-operators.rst