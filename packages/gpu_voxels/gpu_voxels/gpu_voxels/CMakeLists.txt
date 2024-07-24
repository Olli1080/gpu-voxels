project(gpu_voxels
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels)

#generate_export_header(${PROJECT_NAME} 
#  EXPORT_MACRO_NAME
#    ICL_CORE_IMPORT_EXPORT
#  EXPORT_FILE_NAME
#    "${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}/ImportExport.h"
#)

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_sources(${PROJECT_NAME}
  PRIVATE    
    src/GpuVoxels.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/GpuVoxels.h
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_core
    gpu_voxels_dh_robot
    gpu_voxels_voxelmap
    gpu_voxels_voxellist
    gpu_voxels_octree
    gpu_voxels_primitive_array
    gpu_voxels_vis_interface
    gpu_voxels_helpers
    gpu_voxels_logging
    gpu_voxels_primitive_array
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_)

install(
  TARGETS 
    ${PROJECT_NAME}
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${INCLUDE_DIR}
)

#add_subdirectory(test)