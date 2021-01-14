#!/bin/bash

set -Eeuo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

QUERY_MODULES_DIRECTORY="$SCRIPT_DIR/dist"
CPP_DIRECTORY="$SCRIPT_DIR/cpp"
PY_DIRECTORY="$SCRIPT_DIR/python"

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
