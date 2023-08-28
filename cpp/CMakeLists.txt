# Memgraph Mage C++ Query Modules CMake configuration.
cmake_minimum_required(VERSION 3.14)
set(MEMGRAPH_MAGE_PROJECT_NAME "memgraph-mage")
project("${MEMGRAPH_MAGE_PROJECT_NAME}" LANGUAGES C CXX)
include(FetchContent)


# setup CMake module path, defines path for include() and find_package()
# https://cmake.org/cmake/help/latest/variable/CMAKE_MODULE_PATH.html
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cugraph_cmake)

# Export the compile commands so that we can use clang-tidy. Additional benefit
# is easier debugging of compilation and linker flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Required C++ standard.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compile flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall \
    -Werror=switch -Werror=switch-bool")

# Don't omit frame pointer in RelWithDebInfo, for additional callchain debug.
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -fno-omit-frame-pointer")

# Release flags.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

find_package(Threads REQUIRED)
include(ExternalProject)

# Install cuGraph (optional)
include(cugraph)

# Install testing dependencies (gtest)
set(GTEST_ROOT ${PROJECT_BINARY_DIR}/gtest)
ExternalProject_Add(gtest-proj
  PREFIX ${GTEST_ROOT}
  INSTALL_DIR ${GTEST_ROOT}
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG release-1.11.0
  CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
set(GTEST_INCLUDE_DIR ${GTEST_ROOT}/include)
set(GTEST_LIBRARY_PATH ${GTEST_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_DEBUG_LIBRARY_PATH ${GTEST_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}gtestd.a)
add_library(mage_gtest STATIC IMPORTED)
set_target_properties(mage_gtest PROPERTIES
  IMPORTED_LOCATION "${GTEST_LIBRARY_PATH}"
  IMPORTED_LOCATION_DEBUG "${GTEST_DEBUG_LIBRARY_PATH}"
  INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
include_directories("${GTEST_INCLUDE_DIR}")
add_dependencies(mage_gtest gtest-proj)

# Add OpenMP compiling option
find_package(OpenMP)

if(OPENMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Add stduuid - cross-platform UUID implementation
set(UUID_ROOT ${PROJECT_BINARY_DIR}/uuid)
ExternalProject_Add(uuid-proj
  PREFIX ${UUID_ROOT}
  INSTALL_DIR ${UUID_ROOT}
  GIT_REPOSITORY https://github.com/mariusbancila/stduuid.git
  GIT_TAG v1.2.2
  CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
set(UUID_INCLUDE_DIR ${UUID_ROOT}/include)
add_library(mage_uuid INTERFACE)
target_include_directories(mage_uuid INTERFACE ${UUID_INCLUDE_DIR}/gsl)
target_include_directories(mage_uuid INTERFACE ${UUID_INCLUDE_DIR})
add_dependencies(mage_uuid uuid-proj)

# Add mgclient
set(MGCLIENT_ROOT ${PROJECT_BINARY_DIR}/mgclient)
ExternalProject_Add(mgclient-proj
  PREFIX ${MGCLIENT_ROOT}
  INSTALL_DIR ${MGCLIENT_ROOT}
  GIT_REPOSITORY https://github.com/memgraph/mgclient.git
  GIT_TAG v1.4.2
  CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  "-DBUILD_CPP_BINDINGS=ON"
  ${MACOSX_OPENSSL_ROOTDIR_FLAG})

set(MGCLIENT_INCLUDE_DIRS ${MGCLIENT_ROOT}/include ${MGCLIENT_ROOT}/mgclient_cpp/include)
set(MGCLIENT_LIBRARY_PATH ${MGCLIENT_ROOT}/lib/${CMAKE_FIND_LIBRARY_PREFIXES}mgclient.a)

add_library(mgclient STATIC IMPORTED)
set_target_properties(mgclient PROPERTIES
  IMPORTED_LOCATION "${MGCLIENT_LIBRARY_PATH}"
  INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
add_dependencies(mgclient mgclient-proj)

# mg_utility library
add_subdirectory(mg_utility)

FetchContent_Declare(fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
  GIT_TAG 9.1.0
)
FetchContent_MakeAvailable(fmt)

function(add_query_module target_name version src)
  add_library(${target_name} SHARED ${src})
  set_target_properties(${target_name} PROPERTIES SOVERSION "${version}")
  target_link_libraries(${target_name} PRIVATE mg_utility)
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
add_subdirectory(degree_centrality_module)
add_subdirectory(graph_util_module)
add_subdirectory(node_similarity_module)
add_subdirectory(distance_calculator)
add_subdirectory(do_module)
add_subdirectory(periodic_module)
add_subdirectory(map_module)
add_subdirectory(collections_module)
add_subdirectory(text_module)
add_subdirectory(label_module)
add_cugraph_subdirectory(cugraph_module)
