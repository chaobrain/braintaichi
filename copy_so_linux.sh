#! /bin/sh

yum install glibc-devel -y

yum install --disableplugin=fastmirror -y python3-devel.x86_64

pip install taichi==1.7.3
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

# GCC 11
yum install -y centos-release-scl
yum install -y devtoolset-11-gcc devtoolset-11-gcc-c++
echo "source /opt/rh/devtoolset-11/enable" >> ~/.bashrc
source /opt/rh/devtoolset-11/enable

gcc --version

export CC=/opt/rh/devtoolset-11/root/usr/bin/gcc
export CXX=/opt/rh/devtoolset-11/root/usr/bin/g++