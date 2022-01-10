# Copyright (c) 2016-2022 Memgraph Ltd. [https://memgraph.com]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# TODO(gitbuda): Add system install part, e.g., on Ubuntu apt install nvidia-cuda-toolkit -> NO because nvcc 11.0+
# cugraph requires cuda 11+, download from https://developer.nvidia.com/cuda-downloads
# Once installed cuda has to be added to the path because the default /usr/bin/nvcc is still e.g. 10
# export PATH="/usr/local/cuda-11/bin:$PATH"
# NOTE: Be careful, https://github.com/rapidsai/rapids-cmake, somehow take default /usr/bin/nvcc if available.
# NOTE: Old include (/usr/include/cuda* /usr/include/cu*) files should also be deleted
#       because nvcc gets tested against wrong include files.
# INSTALL: sudo apt install libblas-dev liblapack-dev libboost-all-dev
# NCCL is also required (NVIDIA Developer Program registration is required -> huge hustle).
# IMPORTANT: NCCL could be installed from  https://github.com/NVIDIA/nccl (NOTE: take care of versions/tags).
# TODO(gitbuda): Figure out how to compile cugraph in a regular way.
#                https://github.com/rapidsai/cugraph/blob/branch-22.02/SOURCEBUILD.md
# TODO(gitbuda): Figure out how to compile cugraph from Mage cmake (issue with faiss).
# NOTE: Order of the languages matters because cmake pics different compilers.
# FAIL: cugraph depends on gunrock, gunrock is an old repo -> v1.2 does not compile because of some templating issue inside the code.
# NOTE: compiling cugraph takes edges and it's complex -> allow adding linking an already compiled version of cugraph.
# NOTE: CUDA_ARCHITECTURES -> https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/).
# NOTE: Set CMAKE_CUDA_ARCHITECTURES to a specific architecture because compilation is going to be faster
#       CMAKE_CUDA_ARCHITECTURES="NATIVE"
#       CMAKE_CUDA_ARCHITECTURES="75"
#       CMAKE_CUDA_ARCHITECTURES="ALL"
# FAIL: branch-22.02 of rapidsai/raft doesn't have raft::handle_t::get_stream_view method... -> use main branch
# FAIL: rapidsai/raft main branch also doesn't work -> try cugraph (branch-21.12) because 22.02 branches are not compatible
# NOTE: cugraph in Debug mode does NOT compile.

option(MAGE_CUGRAPH_ENABLE "Enable cuGraph build" OFF)

if (MAGE_CUGRAPH_ENABLE)
  enable_language(CUDA)
  set(MAGE_CUGRAPH_REPO "https://github.com/rapidsai/cugraph.git" CACHE STRING "cuGraph GIT repo URL")
  set(MAGE_CUGRAPH_TAG "branch-21.12" CACHE STRING "cuGraph GIT tag to checkout" )
  set(MAGE_CUDA_ARCHITECTURES "NATIVE" CACHE STRING "Passed to cuGraph as CMAKE_CUDA_ARCHITECTURES")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(MAGE_CUGRAPH_ROOT ${PROJECT_BINARY_DIR}/cugraph)
  # TODO(gitbuda): Release is hardcoded here, put the CMAKE var or add additonal flag.
  ExternalProject_Add(cugraph-proj
    PREFIX            "${MAGE_CUGRAPH_ROOT}"
    INSTALL_DIR       "${MAGE_CUGRAPH_ROOT}"
    GIT_REPOSITORY    "${MAGE_CUGRAPH_REPO}"
    GIT_TAG           "${MAGE_CUGRAPH_TAG}"
    SOURCE_SUBDIR     "cpp"
    CMAKE_ARGS        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
                      "-DCMAKE_BUILD_TYPE=Release"
                      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                      "-DCMAKE_CUDA_ARCHITECTURES='${MAGE_CUDA_ARCHITECTURES}'"
                      "-DBUILD_STATIC_FAISS=ON"
                      "-DBUILD_TESTS=OFF"
                      "-DBUILD_CUGRAPH_MG_TESTS=OFF"
  )
  set(MAGE_CUGRAPH_INCLUDE_DIR "${MAGE_CUGRAPH_ROOT}/include")
  set(MAGE_CUGRAPH_LIBRARY_PATH "${MAGE_CUGRAPH_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}cugraph.so")
  add_library(mage_cugraph SHARED IMPORTED)
  set_target_properties(mage_cugraph PROPERTIES
    IMPORTED_LOCATION "${MAGE_CUGRAPH_LIBRARY_PATH}"
  )
  include_directories("${MAGE_CUGRAPH_INCLUDE_DIR}")
  add_dependencies(mage_cugraph cugraph-proj)
endif()

macro(add_cugraph_subdirectory subdirectory_name)
  if (MAGE_ENABLE_CUGRAPH)
    add_subdirectory("${subdirectory_name}")
  endif()
endmacro()
