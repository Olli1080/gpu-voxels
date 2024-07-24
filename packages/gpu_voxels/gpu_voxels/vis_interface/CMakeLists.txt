project(gpu_voxels_vis_interface
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/vis_interface)

#generate_export_header(${PROJECT_NAME} 
#  EXPORT_MACRO_NAME
#    ICL_CORE_IMPORT_EXPORT
#  EXPORT_FILE_NAME
#    "${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}/ImportExport.h"
#)

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_sources(${PROJECT_NAME}
  PRIVATE    
    src/VisProvider.cpp
    src/VisVoxelMap.cpp
    src/VisVoxelList.cpp
    src/VisTemplateVoxelList.cpp
    src/VisPrimitiveArray.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/VisualizerInterface.h
    ${INCLUDE_DIR}/VisProvider.h
    ${INCLUDE_DIR}/VisVoxelMap.h
    ${INCLUDE_DIR}/VisVoxelList.h
    ${INCLUDE_DIR}/VisTemplateVoxelList.h
    ${INCLUDE_DIR}/VisTemplateVoxelList.hpp
    ${INCLUDE_DIR}/VisPrimitiveArray.h
#    ${INCLUDE_DIR}/ImportExport.h
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_voxel
    gpu_voxels_core
    gpu_voxels_helpers
    gpu_voxels_logging
    gpu_voxels_primitive_array
    gpu_voxels_voxelmap
    gpu_voxels_voxellist
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_VIS_INTERFACE_)

install(
  TARGETS 
    ${PROJECT_NAME}
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${INCLUDE_DIR}
)

#add_subdirectory(tests)
#ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})
