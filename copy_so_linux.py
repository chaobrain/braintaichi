import taichi as ti
import glob
import shutil
import subprocess
import os

taichi_path = ti.__path__[0]
taichi_c_api_install_dir = os.path.join(taichi_path, '_lib', 'c_api')

taichi_runtime_lib = os.path.join(taichi_c_api_install_dir, 'lib')

taichi_runtime_lib = os.path.realpath(taichi_runtime_lib)
if not os.path.exists(taichi_runtime_lib):
    taichi_runtime_lib = os.path.join(taichi_c_api_install_dir, 'lib64')
    taichi_runtime_lib = os.path.realpath(taichi_runtime_lib)
if not os.path.exists(taichi_runtime_lib):
    taichi_runtime_lib = os.path.join(taichi_c_api_install_dir, 'bin')
    taichi_runtime_lib = os.path.realpath(taichi_runtime_lib)


taichi_lib_target_dir ='project/brainpylib'

env_script_path = '/project/set_env.sh'
# if not os.path.isfile(env_script_path):
#     os.mknod(env_script_path)

with open(env_script_path, 'w') as env_script:
    env_script.write(f'export taichi_lib_target_dir="{taichi_lib_target_dir}"\n')
    env_script.write(f'export taichi_runtime_lib_dir="{taichi_runtime_lib}"\n')

print(f"Environment setup script created at {env_script_path}")

# so_files = glob.glob(os.path.join(taichi_runtime_lib, '*.so'))
# for so_file in so_files:
#     subprocess.check_call(['cp', '-v', so_file, taichi_lib_target_dir])

# for lib in glob.glob(os.path.join(taichi_runtime_lib, '*.so')):
#     shutil.copy(lib, taichi_lib_target_dir)
#     print('cp {} {}'.format(lib, taichi_lib_target_dir))

