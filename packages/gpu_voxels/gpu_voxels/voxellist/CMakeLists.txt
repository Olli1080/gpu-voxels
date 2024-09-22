project(gpu_voxels_voxellist
  LANGUAGES CXX CUDA
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/voxellist)

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
    src/AbstractVoxelList.cu
    src/BitVoxelList.cu
    src/CountingVoxelList.cu
    src/TemplateVoxelList.cu
    src/VoxelList.cu
    src/VoxelListOperations.cu
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/AbstractVoxelList.h
    ${INCLUDE_DIR}/BitVoxelList.h
    ${INCLUDE_DIR}/BitVoxelList.hpp
    ${INCLUDE_DIR}/CountingVoxelList.h
    ${INCLUDE_DIR}/TemplateVoxelList.h
    ${INCLUDE_DIR}/TemplateVoxelList.hpp
    ${INCLUDE_DIR}/VoxelList.h
    ${INCLUDE_DIR}/kernels/VoxelListOperations.h
    ${INCLUDE_DIR}/kernels/VoxelListOperations.hpp
#    ${INCLUDE_DIR}/ImportExport.h
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_logging
    gpu_voxels_helpers
    gpu_voxels_core
    gpu_voxels_voxel
    gpu_voxels_voxelmap
    gpu_voxels_octree
    gpu_voxels_vis_interface
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_VOXELLIST_)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES all)

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

#add_subdirectory(tests)
#ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})