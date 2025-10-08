#!/bin/bash

# check for --ci flag
if [ "$1" = "--ci" ]; then
  ci=true
else
  ci=false
fi

if [ "$ci" = true ]; then
    python3 -m pip install --no-cache-dir -r /tmp/python-requirements.txt --break-system-packages
    python3 -m pip install --no-cache-dir -r /tmp/auth_module-requirements.txt --break-system-packages
else
    python3 -m pip install --no-cache-dir -r /mage/python/requirements.txt --break-system-packages
    python3 -m pip install --no-cache-dir -r /mage/python/tests/requirements.txt --break-system-packages
    python3 -m pip install --no-cache-dir -r /usr/lib/memgraph/auth_module/requirements.txt --break-system-packages
fi

if [ "$TARGETARCH" = "arm64" ]; then
    if [ "$CACHE_PRESENT" = "true" ]; then
        echo "Using cached torch packages"
        python3 -m pip install --no-index --find-links=/mage/wheels/ torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter --break-system-packages
    else
        python3 -m pip install --no-cache-dir torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html --break-system-packages
    fi
    curl -o dgl-2.5.0-cp312-cp312-linux_aarch64.whl https://s3.eu-west-1.amazonaws.com/deps.memgraph.io/wheels/arm64/dgl-2.5.0-cp312-cp312-linux_aarch64.whl
    python3 -m pip install --no-cache-dir dgl-2.5.0-cp312-cp312-linux_aarch64.whl --break-system-packages
else
    if [ "$CACHE_PRESENT" = "true" ]; then
        echo "Using cached torch packages"
        python3 -m pip install --no-index --find-links=/mage/wheels/ torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter --break-system-packages
    else
        python3 -m pip install --no-cache-dir torch-sparse torch-cluster torch-spline-conv torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cpu.html --break-system-packages
    fi
    python3 -m pip install --no-cache-dir dgl==2.5.0 -f https://data.dgl.ai/wheels/torch-2.6/repo.html --break-system-packages
fi
rm -fr /home/memgraph/.cache/pip
