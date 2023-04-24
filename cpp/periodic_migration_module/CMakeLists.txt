include(GNUInstallDirs)

# Add all module files related to graph util module
set(periodic_migration_src
    periodic_migration.cpp)

add_query_module(periodic_migration 1 "${periodic_migration_src}")


if(WIN32)
  set(MGCONSOLE_ON_WINDOWS TRUE)
elseif(APPLE)
  set(MGCONSOLE_ON_OSX TRUE)
elseif(UNIX)
  set(MGCONSOLE_ON_LINUX TRUE)
endif()

MESSAGE(warning "${CMAKE_INSTALL_LIBDIR}")

if(MGCONSOLE_ON_LINUX)
  # On Debian, the libdir has multiarch path which we don't want while searching for
  # dependancy libs
  string(REGEX MATCH "lib[^/]*" MG_INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR})
else()
  set(MG_INSTALL_LIB_DIR "lib")
endif()


ExternalProject_Add(mgclient-proj
  PREFIX mgclient
  GIT_REPOSITORY https://github.com/memgraph/mgclient.git
  GIT_TAG v1.3.0
  CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>"
  "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
  "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
  "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  ${MACOSX_OPENSSL_ROOTDIR_FLAG}
  INSTALL_DIR "${PROJECT_BINARY_DIR}/mgclient")

ExternalProject_Get_Property(mgclient-proj install_dir)
set(MGCLIENT_ROOT ${install_dir})
set(MGCLIENT_INCLUDE_DIRS ${MGCLIENT_ROOT}/include)
set(MGCLIENT_LIBRARY_PATH ${MGCLIENT_ROOT}/${MG_INSTALL_LIB_DIR}/libmgclient.a)
set(MGCLIENT_LIBRARY mgclient)

add_library(${MGCLIENT_LIBRARY} STATIC IMPORTED)
set_target_properties(${MGCLIENT_LIBRARY} PROPERTIES
  IMPORTED_LOCATION ${MGCLIENT_LIBRARY_PATH}
  INTERFACE_LINK_LIBRARIES Threads::Threads)
add_dependencies(${MGCLIENT_LIBRARY} mgclient-proj)


target_compile_definitions(periodic_migration PRIVATE MGCLIENT_STATIC_DEFINE)
target_link_libraries(periodic_migration PRIVATE mgclient fmt::fmt)
