# -*- coding: utf-8 -*-


import glob
import os
import io
import platform
import re
import shutil
import subprocess
import sys
from setuptools import setup, Extension, find_namespace_packages
from setuptools.command.build_ext import build_ext

try:
    from distutils import sysconfig
except ModuleNotFoundError:
    import sysconfig
try:
    import pybind11
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        'Please install pybind11 before compiling braintaichi! '
        '\n'
        '> pip install pybind11'
    )


__minimal_taichi_version = (1, 7, 2)

try:
    import taichi as ti

    if ti.__version__ < __minimal_taichi_version:
        raise ModuleNotFoundError
except ModuleNotFoundError:
    taichi_installed = 'taichi' in globals()
    error_message = (f'Please update taichi to {__minimal_taichi_version} or above!'
                     if taichi_installed else
                     'Please install taichi before compiling braintaichi!')
    error_message += '\n> pip install taichi -U'
    raise ModuleNotFoundError(error_message)

# set taichi environments
taichi_path = ti.__path__[0]
taichi_lib_dir = os.path.join(taichi_path, '_lib', 'runtime')
taichi_c_api_install_dir = os.path.join(taichi_path, '_lib', 'c_api')
taichi_runtime_lib = os.path.join(taichi_c_api_install_dir, 'lib')
taichi_runtime_lib = os.path.realpath(taichi_runtime_lib)
if not os.path.exists(taichi_runtime_lib):
    taichi_runtime_lib = os.path.join(taichi_c_api_install_dir, 'lib64')
    taichi_runtime_lib = os.path.realpath(taichi_runtime_lib)
if not os.path.exists(taichi_runtime_lib):
    taichi_runtime_lib = os.path.join(taichi_c_api_install_dir, 'bin')
    taichi_runtime_lib = os.path.realpath(taichi_runtime_lib)

os.environ.update(
    {
        'TAICHI_C_API_INSTALL_DIR': taichi_c_api_install_dir,
        'TI_LIB_DIR': taichi_lib_dir,
        'BRAINPY_CUDA': 'yes'
    }
)

HERE = os.path.dirname(os.path.realpath(__file__))

# obtain long description from README
here = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    README = f.read()

# This custom class for building the extensions uses CMake to compile. You
# don't have to use CMake for this task, but I found it to be the easiest when
# compiling ops with GPU support since setuptools doesn't have great CUDA
# support.
class CMakeBuildExt(build_ext):
    def build_extensions(self):
        # Work out the relevant Python paths to pass to CMake,
        # adapted from the PyTorch build system
        if platform.system() == "Windows":
            cmake_python_library = "{}/libs/python{}.lib".format(
                sysconfig.get_config_var("prefix"),
                sysconfig.get_config_var("VERSION"),
            )
            if not os.path.exists(cmake_python_library):
                cmake_python_library = "{}/libs/python{}.lib".format(
                    sys.base_prefix,
                    sysconfig.get_config_var("VERSION"),
                )
        else:
            cmake_python_library = "{}/{}".format(sysconfig.get_config_var("LIBDIR"),
                                                  sysconfig.get_config_var("INSTSONAME"))
        cmake_python_include_dir = sysconfig.get_python_inc()
        install_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath("dummy")))
        print("install_dir", install_dir)
        os.makedirs(install_dir, exist_ok=True)
        
        gcc_path = shutil.which("gcc-11") or "/usr/bin/gcc-11"  # 默认回退路径
        gxx_path = shutil.which("g++-11") or "/usr/bin/g++-11"
        
        cmake_args = [
            "-DPYTHON_LIBRARY={}".format(os.path.join(sysconfig.get_config_var('LIBDIR'))),
            "-DPYTHON_INCLUDE_DIRS={}".format(sysconfig.get_python_inc()),
            "-DPYTHON_INCLUDE_DIR={}".format(sysconfig.get_python_inc()),
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            # CMake 中找不到 CUDA_CUDA_LIBRARY
            # 解决方案: https://blog.csdn.net/comedate/article/details/128334923
            "-DCMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs",
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DPython_LIBRARIES={}".format(cmake_python_library),
            "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            "-DCMAKE_BUILD_TYPE={}".format("Debug" if self.debug else "Release"),
            "-DCMAKE_PREFIX_PATH={}".format(os.path.dirname(pybind11.get_cmake_dir())),
            "-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda",
            # "-DCUDA_CUDART_LIBRARY=/usr/local/cuda/lib64/libcudart.so",
            # "-DCUDA_CUSPARSE_LIBRARY=/usr/local/cuda/lib64/libcusparse.so",
            "-DCMAKE_CXX_FLAGS=-O3",
            "-DCMAKE_C_FLAGS=-O3",
            "-DCMAKE_CUDA_FLAGS={}".format(
                '-O3 '
                '-arch=sm_86 '
                # '-gencode=arch=compute_30,code=sm_30 '
                # cuda11x-start
                # '-gencode=arch=compute_35,code=sm_35 '
                # '-gencode=arch=compute_37,code=sm_37 '
                # cuda12x-start
                '-gencode=arch=compute_50,code=sm_50 '
                '-gencode=arch=compute_52,code=sm_52 '
                '-gencode=arch=compute_60,code=sm_60 '
                '-gencode=arch=compute_61,code=sm_61 '
                '-gencode=arch=compute_70,code=sm_70 '
                '-gencode=arch=compute_75,code=sm_75 '
                '-gencode=arch=compute_80,code=sm_80 '
                '-gencode=arch=compute_86,code=sm_86 '
                '-gencode=arch=compute_87,code=sm_87 '
                # cuda11x-end
                '-gencode=arch=compute_89,code=sm_89 '
                '-gencode=arch=compute_90,code=sm_90 '
            ),
            "-DCUDACXX=/usr/local/cuda/bin/nvcc",
            f"-DCMAKE_C_COMPILER={gcc_path}",
            f"-DCMAKE_CXX_COMPILER={gxx_path}",
        ]
        if os.environ.get("BRAINPY_CUDA", "no").lower() == "yes":
            cmake_args.append("-DBRAINPY_CUDA=yes")
        print(" ".join(cmake_args))

        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(["cmake"] + cmake_args + [HERE], cwd=self.build_temp)

        # Build all the extensions
        super().build_extensions()

        # Finally run install
        subprocess.check_call(["cmake", "--build", ".", "--target", "install", "-j24"], cwd=self.build_temp)

    def build_extension(self, ext):
        subprocess.check_call(["cmake", "--build", ".", "--target", "gpu_ops", "-j24"], cwd=self.build_temp)


# version control
with open(os.path.join(HERE, 'braintaichi', '__init__.py'), 'r') as f:
    init_py = f.read()
    __version__ = re.search('__version__ = "(.*)"', init_py).groups()[0]

# build
setup(
    name='braintaichi',
    version=__version__,
    description='Leveraging Taichi Lang to customize brain dynamics operators',
    long_description=README,
    long_description_content_type="text/markdown",
    author='BrainTaichi Developers',
    author_email='chao.brain@qq.com',
    packages=find_namespace_packages(exclude=['lib*', 'docs*', 'tests*', 'win_dll*', 'out*', 'cmake*', 'scripts*']),
    include_package_data=True,
    package_data={
        'braintaichi': ['*.so']
    },
    install_requires=['brainunit', 'brainstate', 'jax', 'jaxlib', 'taichi'],
    extras_require={"test": "pytest"},
    python_requires='>=3.10',
    url='https://github.com/chaoming0625/braintaichi',
    ext_modules=[
        Extension("gpu_ops",
                  ['lib/gpu_ops.cc'] + glob.glob("lib/*.cu"),
                  extra_link_args=["-rpath", taichi_runtime_lib]),
        Extension("cpu_ops",
                  glob.glob("lib/cpu_*.cc") + glob.glob("lib/cpu_*.cpp"),
                  extra_link_args=["-rpath", taichi_runtime_lib]),
    ],
    cmdclass={"build_ext": CMakeBuildExt},
    license='Apache-2.0 license',
    keywords=('event-driven computation, '
              'sparse computation'),
    classifiers=[
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
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
)
