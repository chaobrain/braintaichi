#! /bin/sh

set -e
set -x

#pip install "cmake==3.22.*"
rm setup.py
mv setup_cuda.py setup.py
# mv setup_cpu.py setup.py

yum -y install yum-utils

# Install CUDA 12.0
yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

yum install --setopt=obsoletes=0 -y \
   cuda-12-8-12.8.1-1 \
   cuda-cudart-devel-12-8-12.8.90-1 \
   
ln -s cuda-12.8 /usr/local/cuda

# yum install glibc-devel

yum install -y python3-devel

pip_config_dir="${HOME}/.pip"
pip_config_file="${pip_config_dir}/pip.conf"

mkdir -p "${pip_config_dir}"

if [[ ! -f "${pip_config_file}" ]]; then
    touch "${pip_config_file}"
fi

echo "[global]" >> "${pip_config_file}"
# echo "index-url = https://mirrors.aliyun.com/pypi/simple/" >> "${pip_config_file}"
echo "index-url = https://pypi.org/simple" >> "${pip_config_file}"

echo "Pip mirror sources have been set!"