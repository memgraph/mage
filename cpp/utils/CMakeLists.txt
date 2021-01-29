set(utils_src
    mg_utils.cpp)

add_library(utils STATIC ${utils_src})
set_target_properties(utils PROPERTIES POSITION_INDEPENDENT_CODE True)
target_include_directories(utils PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(utils PRIVATE mg_procedure)
