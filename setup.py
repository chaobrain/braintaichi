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

import io
import os
import re
import sys
import time

from setuptools import find_packages
from setuptools import setup

# version
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'braintaichi/', '__init__.py'), 'r') as f:
  init_py = f.read()
version = re.search('__version__ = "(.*)"', init_py).groups()[0]
if len(sys.argv) > 2 and sys.argv[2] == '--python-tag=py3':
  version = version
else:
  version += '.post{}'.format(time.strftime("%Y%m%d", time.localtime()))

# obtain long description from README
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
  README = f.read()

# installation packages
packages = find_packages(
  exclude=["docs*", "tests*", "examples*", "build*",
           "dist*", "braintaichi.egg-info*", "braintaichi/__pycache__*"]
)

# setup
setup(
  name='braintaichi',
  version=version,
  description='Integrating Taichi Lang into Brain Dynamics Programming.',
  long_description=README,
  long_description_content_type="text/markdown",
  author='BrainPy Team',
  author_email='chao.brain@qq.com',
  packages=packages,
  python_requires='>=3.9',
  install_requires=['numpy>=1.15', 'jax', 'brainstate'],
  url='https://github.com/brainpy/braintaichi',
  project_urls={
    "Bug Tracker": "https://github.com/brainpy/braintaichi/issues",
    "Documentation": "https://braintaichi.readthedocs.io/",
    "Source Code": "https://github.com/brainpy/braintaichi",
  },
  extras_require={
    'cpu': ['jaxlib', ],
    'cuda11': ['jaxlib[cuda11_pip]', ],
    'cuda12': ['jaxlib[cuda12_pip]', ],
    'tpu': ['jaxlib[tpu]'],
    'cpu_mini': ['jaxlib'],
    'cuda11_mini': ['jaxlib[cuda11_pip]'],
    'cuda12_mini': ['jaxlib[cuda12_pip]'],
  },
  keywords=('computational neuroscience, '
            'brain-inspired computation, '
            'brain dynamics programming'),
  classifiers=[
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Programming Language :: Python :: 3.11',
    'Programming Language :: Python :: 3.12',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: Apache Software License',
    'Topic :: Scientific/Engineering :: Bio-Informatics',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries',
  ],
  license='Apache-2.0 license',
)
