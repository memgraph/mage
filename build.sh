#!/bin/bash

set -Eeuo pipefail
script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

QUERY_MODULES_DIRECTORY="$script_dir/dist"
CPP_DIRECTORY="$script_dir/cpp"
PY_DIRECTORY="$script_dir/python"

mkdir -p "$QUERY_MODULES_DIRECTORY"

# Build C++.
pushd "$CPP_DIRECTORY"
mkdir -p build
cd build
cmake ..
make
cp ./*.so "$QUERY_MODULES_DIRECTORY"
popd

# Build Python.
pushd "$PY_DIRECTORY"
cp ./*.py "$QUERY_MODULES_DIRECTORY"
popd
