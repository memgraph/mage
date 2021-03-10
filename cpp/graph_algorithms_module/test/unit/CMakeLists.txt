set(test_prefix unit__)

add_custom_target(unit)

function(add_unit_test test_cpp)
  # get exec name (remove extension from the abs path)
  get_filename_component(exec_name ${test_cpp} NAME_WE)
  set(target_name ${test_prefix}${exec_name})
  add_executable(${target_name} ${test_cpp})
  # OUTPUT_NAME sets the real name of a target when it is built and can be
  # used to help create two targets of the same name even though CMake
  # requires unique logical target names
  set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${exec_name})
  # TODO: this is a temporary workaround the test build warnings
  target_compile_options(${target_name} PRIVATE -Wno-comment -Wno-sign-compare
                                                -Wno-unused-variable)
  target_link_libraries(${target_name} glog gflags gtest gtest_main Threads::Threads)
  # register test
  add_test(${target_name} ${exec_name})
  # add to unit target
  add_dependencies(unit ${target_name})
endfunction(add_unit_test)

add_unit_test(algorithms_biconnected_components.cpp)
target_link_libraries(${test_prefix}algorithms_biconnected_components libgraph glog gflags)

add_unit_test(algorithms_bipartite_matching.cpp)
target_link_libraries(${test_prefix}algorithms_bipartite_matching libgraph)

add_unit_test(algorithms_bridges.cpp)
target_link_libraries(${test_prefix}algorithms_bridges libgraph glog gflags)

add_unit_test(algorithms_cutsets.cpp)
target_link_libraries(${test_prefix}algorithms_cutsets libgraph glog gflags)

add_unit_test(algorithms_cycles.cpp)
target_link_libraries(${test_prefix}algorithms_cycles libgraph glog gflags)

