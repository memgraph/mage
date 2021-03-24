# Memgraph Mage C++ Query Modules CMake configuration.

cmake_minimum_required(VERSION 3.8)

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

# Install gtest
include(ExternalProject)
ExternalProject_Add(gtest-proj
        PREFIX            gtest
        GIT_REPOSITORY    https://github.com/google/googletest.git
        GIT_TAG           release-1.8.1
        CMAKE_ARGS        "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
                          "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
                          "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
                          "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
        INSTALL_DIR       "${PROJECT_BINARY_DIR}/gtest")
ExternalProject_Get_Property(gtest-proj install_dir)
set(GTEST_ROOT ${install_dir})
set(GTEST_INCLUDE_DIRS ${GTEST_ROOT}/include
        CACHE INTERNAL "Path to gtest include directory")
set(GTEST_LIBRARY_PATH ${GTEST_ROOT}/lib/${MGCLIENT_FIND_LIBRARY_PREFIXES}gtest.a)
set(GTEST_DEBUG_LIBRARY_PATH ${GTEST_ROOT}/lib/${MGCLIENT_FIND_LIBRARY_PREFIXES}gtestd.a)
add_library(gtest STATIC IMPORTED)
set_target_properties(gtest PROPERTIES
        IMPORTED_LOCATION ${GTEST_LIBRARY_PATH}
        IMPORTED_LOCATION_DEBUG ${GTEST_DEBUG_LIBRARY_PATH}
        INTERFACE_LINK_LIBRARIES "${CMAKE_THREAD_LIBS_INIT}")
add_dependencies(gtest gtest-proj)


# mg_procedure library
add_library(mg_procedure INTERFACE)
target_include_directories(mg_procedure INTERFACE ${CMAKE_SOURCE_DIR}/mg_procedure)

# mg_procedure library
add_subdirectory(mg_utility)

function(add_query_module target_name version src)
  add_library(${target_name} SHARED ${src})
  set_target_properties(${target_name} PROPERTIES SOVERSION ${version})
  set_target_properties(${target_name} PROPERTIES PREFIX "cpp_")
  target_link_libraries(${target_name} PRIVATE mg_procedure)
endfunction()

# Available query modules.
add_subdirectory(connectivity_module)
add_subdirectory(biconnected_components_module)