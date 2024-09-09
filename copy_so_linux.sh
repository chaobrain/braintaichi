#! /bin/sh

yum install glibc-devel -y

yum install --disableplugin=fastmirror -y python3-devel.x86_64

pip install taichi==1.7.2
chmod +x ./copy_so_linux.py
python copy_so_linux.py

# Ensure the set_env.sh script is actually created and is not empty
dir /project
if [ -s set_env.sh ]; then
    source /project/set_env.sh
    cp "$taichi_runtime_lib_dir"/libtaichi_c_api.so /project/braintaichi/
else
    echo "Environment setup script 'set_env.sh' is missing or empty."
    exit 1
fi
