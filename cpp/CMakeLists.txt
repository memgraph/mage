# Memgraph Mage C++ Query Modules CMake configuration.

cmake_minimum_required(VERSION 3.9)

project(memgraph-mage)

# Export the compile commands so that we can use clang-tidy. Additional benefit
# is easier debugging of compilation and linker flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Required C++ standard.
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

# Install gtest

include(ExternalProject)
set(GTEST_ROOT ${PROJECT_BINARY_DIR}/gtest)
ExternalProject_Add(gtest-proj
  PREFIX            ${GTEST_ROOT}
  INSTALL_DIR       ${GTEST_ROOT}
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           v1.10.x
  CMAKE_ARGS        "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
                    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARY_PATH ${GTEST_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_DEBUG_LIBRARY_PATH ${GTEST_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtestd.a)
add_library(gtest STATIC IMPORTED)
set_target_properties(gtest PROPERTIES
  IMPORTED_LOCATION ${GTEST_LIBRARY_PATH}
  IMPORTED_LOCATION_DEBUG ${GTEST_DEBUG_LIBRARY_PATH}
  INTERFACE_LINK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
include_directories("${GTEST_INCLUDE_DIR}")
add_dependencies(gtest gtest-proj)

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

# Available query modules.
include(CTest)
enable_testing()
add_subdirectory(betweenness_centrality_module)
add_subdirectory(biconnected_components_module)
add_subdirectory(bipartite_matching_module)
add_subdirectory(bridges_module)
add_subdirectory(connectivity_module)
add_subdirectory(cycles_module)
add_subdirectory(community_detection_module)
add_subdirectory(pagerank_module)
add_subdirectory(uuid_module)
add_subdirectory(katz_centrality_module)
