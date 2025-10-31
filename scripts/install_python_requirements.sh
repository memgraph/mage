#!/bin/bash
set -euo pipefail
# run me as `memgraph` user

CI=false
CACHE_PRESENT=false
CUDA=false
ARCH=amd64

while [[ $# -gt 0 ]]; do
  case $1 in
    --ci)
      CI=true
      shift
      ;;
    --cache-present)
      CACHE_PRESENT=true
      ;;
    --cuda)
      CUDA=true
      shift
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [[ "$CUDA" == true && "$ARCH" ! "amd64" ]]; then
    echo "CUDA is only supported on amd64 architecture."
    exit 1
fi

export PIP_BREAK_SYSTEM_PACKAGES=1
if [[ "$CUDA" == true ]]; then
    requirements_file="requirements-gpu.txt"
else
    requirements_file="requirements.txt"
fi

if [ "$CI" = true ]; then
    python3 -m pip install --no-cache-dir -r "/tmp/${requirements_file}"
    python3 -m pip install --no-cache-dir -r /tmp/auth_module-requirements.txt
else
    python3 -m pip install --no-cache-dir -r "/mage/python/${requirements_file}"
    python3 -m pip install --no-cache-dir -r /mage/python/tests/requirements.txt
    python3 -m pip install --no-cache-dir -r /usr/lib/memgraph/auth_module/requirements.txt
fi

if [ "$TARGETARCH" = "arm64" ]; then
    if [ "$CACHE_PRESENT" = "true" ]; then
        echo "Using cached torch packages"
        python3 -m pip install --no-index --find-links=/mage/wheels/ torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter
    else
        python3 -m pip install --no-cache-dir torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
    fi
    curl -o dgl-2.5.0-cp312-cp312-linux_aarch64.whl https://s3.eu-west-1.amazonaws.com/deps.memgraph.io/wheels/arm64/dgl-2.5.0-cp312-cp312-linux_aarch64.whl
    python3 -m pip install --no-cache-dir dgl-2.5.0-cp312-cp312-linux_aarch64.whl
else
    if [[ "$CUDA" == true ]]; then
      python3 -m pip install --no-cache-dir torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html
    elif [ "$CACHE_PRESENT" = "true" ]; then
        echo "Using cached torch packages"
        python3 -m pip install --no-index --find-links=/mage/wheels/ torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter
    else
        python3 -m pip install --no-cache-dir torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html
    fi
    python3 -m pip install --no-cache-dir dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.6/repo.html
fi
rm -fr /home/memgraph/.cache/pip
