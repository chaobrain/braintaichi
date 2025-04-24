#!/bin/bash

# Get the current Python version
python_version=$(python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")

# Set the CIBW_BUILD environment variable based on the Python version
if [[ $python_version == "3.10" ]]; then
    cp ci/mac/arm64/pyproject3.10.toml pyproject.toml
elif [[ $python_version == "3.11" ]]; then
    cp ci/mac/arm64/pyproject3.11.toml pyproject.toml
elif [[ $python_version == "3.12" ]]; then
    cp ci/mac/arm64/pyproject3.12.toml pyproject.toml
else
    echo "Unsupported Python version: $python_version"
    exit 1
fi