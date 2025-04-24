#! /bin/sh

pip install taichi==1.7.3
chmod +x ./copy_so_macOS.py
python copy_so_macOS.py

# Ensure the set_env.sh script is actually created and is not empty
source project/set_env.sh
cp "$taichi_runtime_lib_dir"/libtaichi_c_api.dylib project/braintaichi/

ls project/braintaichi/

