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

# mg_procedure library
add_library(mg_procedure INTERFACE)
target_include_directories(mg_procedure INTERFACE ${CMAKE_SOURCE_DIR}/mg_procedure)

# mg_procedure library
add_subdirectory(mg_utility)

function(add_query_module target_name version src)
  add_library(${target_name} SHARED ${src})
  set_target_properties(${target_name} PROPERTIES SOVERSION ${version})
  target_link_libraries(${target_name} PRIVATE mg_procedure)
endfunction()

# Available query modules.
add_subdirectory(connectivity_module)