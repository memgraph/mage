# Nitin A. Gawande, PNNL
# Oct 19, 2018

# Include cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -fopenmp -O0 -fPIC")

find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# set varaibles using nomenclature used in Makefile
set(MODULE_DIR louvain_module/grappolo)
set(COFOLDER BasicCommunitiesDetection)
set(UTFOLDER Utility)
set(CLFOLDER Coloring)
set(FSFOLDER FullSyncOptimization)
set(DEFINSTR DefineStructure)

set(IO_HEADERS ${CMAKE_SOURCE_DIR}/${MODULE_DIR}/${DEFINSTR})
set(GRAPPOLO_HEADERS ${CMAKE_SOURCE_DIR}/${MODULE_DIR}/${DEFINSTR} PARENT_SCOPE)

add_subdirectory(BasicCommunitiesDetection)
add_subdirectory(Utility)
add_subdirectory(Coloring)
add_subdirectory(FullSyncOptimization)