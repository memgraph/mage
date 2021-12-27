# Memgraph Mage C++ Query Modules CMake configuration.

cmake_minimum_required(VERSION 3.9)

# TODO(gitbuda): CUDA should be optional part of mage.
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
# NOTE: Set CMAKE_CUDA_ARCHITECTURES to a specific architecture because compilation is going to be faster
#       CMAKE_CUDA_ARCHITECTURES="NATIVE"
#       CMAKE_CUDA_ARCHITECTURES="75"
#       CMAKE_CUDA_ARCHITECTURES="ALL"
# FAIL: branch-22.02 of rapidsai/raft doesn't have raft::handle_t::get_stream_view method... -> use main branch
# FAIL: rapidsai/raft main branch also doesn't work -> try cugraph (branch-21.12) because 22.02 branches are not compatible
project(memgraph-mage LANGUAGES CXX CUDA C)

# Export the compile commands so that we can use clang-tidy. Additional benefit
# is easier debugging of compilation and linker flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Required C++ standard.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compile flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall \
    -Werror=switch -Werror=switch-bool -Werror=return-type")
# Don't omit frame pointer in RelWithDebInfo, for additional callchain debug.
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")
# Release flags.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Install testing dependencies
find_package(Threads REQUIRED)

include(ExternalProject)

# Install gtest
set(GTEST_ROOT ${PROJECT_BINARY_DIR}/gtest)
ExternalProject_Add(gtest-proj
  PREFIX            ${GTEST_ROOT}
  INSTALL_DIR       ${GTEST_ROOT}
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           release-1.11.0
  CMAKE_ARGS        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
                    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARY_PATH ${GTEST_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_DEBUG_LIBRARY_PATH ${GTEST_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtestd.a)
add_library(mage_gtest STATIC IMPORTED)
set_target_properties(mage_gtest PROPERTIES
  IMPORTED_LOCATION ${GTEST_LIBRARY_PATH}
  IMPORTED_LOCATION_DEBUG ${GTEST_DEBUG_LIBRARY_PATH}
  INTERFACE_LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
include_directories("${GTEST_INCLUDE_DIR}")
add_dependencies(mage_gtest gtest-proj)

# Install cuGraph (optional)
option(MAGE_ENABLE_CUGRAPH "Enable cuGraph build" OFF)
if (MAGE_ENABLE_CUGRAPH)
  set(MAGE_CUGRAPH_REPO "https://github.com/rapidsai/cugraph.git" CACHE STRING "cuGraph GIT repo URL")
  set(MAGE_CUGRAPH_TAG "branch-21.12" CACHE STRING "cuGraph GIT tag to checkout" )
  set(MAGE_CUDA_ARCHITECTURES "NATIVE" CACHE STRING "Passed to cuGraph as CMAKE_CUDA_ARCHITECTURES")
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
  set(MAGE_CUGRAPH_ROOT ${PROJECT_BINARY_DIR}/cugraph)
  ExternalProject_Add(cugraph-proj
    PREFIX            "${MAGE_CUGRAPH_ROOT}"
    INSTALL_DIR       "${MAGE_CUGRAPH_ROOT}"
    GIT_REPOSITORY    "${MAGE_CUGRAPH_REPO}"
    GIT_TAG           "${MAGE_CUGRAPH_TAG}"
    SOURCE_SUBDIR     "cpp"
    CMAKE_ARGS        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
                      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
                      "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                      "-DCMAKE_CUDA_ARCHITECTURES='${MAGE_CUDA_ARCHITECTURES}'"
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

# Add OpenMP compiling option

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# mg_procedure library
add_library(mg_procedure INTERFACE)
target_include_directories(mg_procedure INTERFACE ${CMAKE_SOURCE_DIR}/mg_procedure)

# mg_utility library
add_subdirectory(mg_utility)

function(add_query_module target_name version src)
  add_library(${target_name} SHARED ${src})
  set_target_properties(${target_name} PROPERTIES SOVERSION ${version})
  target_link_libraries(${target_name} PRIVATE mg_procedure)
endfunction()

macro(add_cugraph_subdirectory subdirectory_name)
  if (MAGE_ENABLE_CUGRAPH)
    add_subdirectory("${subdirectory_name}")
  endif()
endmacro()

# Available query modules.
include(CTest)
enable_testing()
add_subdirectory(connectivity_module)
add_subdirectory(biconnected_components_module)
add_subdirectory(bridges_module)
add_subdirectory(cycles_module)
add_subdirectory(bipartite_matching_module)
add_subdirectory(betweenness_centrality_module)
add_subdirectory(pagerank_module)
add_subdirectory(uuid_module)
add_cugraph_subdirectory(ccugraph_alg_module)
