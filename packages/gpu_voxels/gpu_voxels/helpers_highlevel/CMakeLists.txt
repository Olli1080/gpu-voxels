project(gpu_voxels_distance_map_converter
  LANGUAGES CXX CUDA
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/helpers_highlevel)

#generate_export_header(${PROJECT_NAME} 
#  EXPORT_MACRO_NAME
#    ICL_CORE_IMPORT_EXPORT
#  EXPORT_FILE_NAME
#    "${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}/ImportExport.h"
#)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
  PRIVATE    
    src/DistanceMapConverter.cu
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/DistanceMapConverter.h
#    ${INCLUDE_DIR}/ImportExport.h
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_logging
    gpu_voxels_voxelmap
    gpu_voxels_voxellist
    gpu_voxels_voxel
    gpu_voxels_core
    gpu_voxels_helpers
    gpu_voxels_vis_interface
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_DIST_MAP_CONV_)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(
  TARGETS 
    ${PROJECT_NAME}
  EXPORT
    gpu_voxels-targets
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${INCLUDE_DIR}
)
