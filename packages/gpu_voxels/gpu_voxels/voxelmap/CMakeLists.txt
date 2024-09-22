project(gpu_voxels_voxelmap
  LANGUAGES CXX CUDA
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/voxelmap)

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
    src/AbstractVoxelMap.cu
    src/DistanceVoxelMap.cu
    src/ProbVoxelMap.cu
    src/TemplateVoxelMap.cu
    src/VoxelMap.cu
    src/VoxelMapOperations.cu
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/kernels/VoxelMapOperations.h
    ${INCLUDE_DIR}/kernels/VoxelMapOperationsPBA.h
    ${INCLUDE_DIR}/kernels/VoxelMapOperations.hpp
    ${INCLUDE_DIR}/kernels/VoxelMapOperationsPBA.hpp
    ${INCLUDE_DIR}/AbstractVoxelMap.h
    ${INCLUDE_DIR}/BitVoxelMap.h
    ${INCLUDE_DIR}/BitVoxelMap.hpp
    ${INCLUDE_DIR}/ProbVoxelMap.h
    ${INCLUDE_DIR}/ProbVoxelMap.cuhpp
    ${INCLUDE_DIR}/TemplateVoxelMap.h
    ${INCLUDE_DIR}/TemplateVoxelMap.hpp
    ${INCLUDE_DIR}/VoxelMap.h
    ${INCLUDE_DIR}/VoxelMap.hpp
    ${INCLUDE_DIR}/DistanceVoxelMap.h
    #${INCLUDE_DIR}/DistanceVoxelMap.hpp
#    ${INCLUDE_DIR}/ImportExport.h
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    icl_core
    icl_core_config
    icl_core_logging
    icl_core_performance_monitor
    gpu_voxels_logging
    gpu_voxels_helpers
    gpu_voxels_core
    gpu_voxels_voxel
    gpu_voxels_voxellist
    gpu_voxels_vis_interface
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_VOXELMAP_)
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

#SET(ICMAKER_CUDA_CPPDEFINES ${ICMAKER_CUDA_CPPDEFINES} --ptxas-options=-v)
#add_definitions(-DIC_PERFORMANCE_MONITOR)