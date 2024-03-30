#!/bin/bash
set -Eeuo pipefail
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MAKE_VERBOSE="VERBOSE=1"
# MAKE_VERBOSE=""
# MAKE_PARALLEL="-j16"
MAKE_PARALLEL=""

install_latest_cmake () {
  # Latest CMake on Ubuntu 22.04
  wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo apt-key add -
  sudo apt-add-repository "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
  sudo apt-get install -y cmake
}

mkdir -p "$SCRIPT_DIR/build" && cd "$SCRIPT_DIR/build/"
# rm -rf dgl
# git clone --recurse-submodules -b 2.1.x https://github.com/dmlc/dgl.git
cd dgl && mkdir build -p && cd build
# rm -rf ./*
cmake -DUSE_CUDA=ON ..
# cmake -DUSE_CUDA=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..
make -j4

# # https://docs.dgl.ai/install/index.html#system-requirements
# # TODO(gitbuda): conda is required on the system
# bash script/create_dev_conda_env.sh -g 12.1
# bash script/build_dgl.sh -g
