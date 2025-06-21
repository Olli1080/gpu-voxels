project(gpu_voxels_voxel
  LANGUAGES CXX CUDA
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/voxel)

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
    src/BitVoxel.cu
    src/CountingVoxel.cu
    src/DefaultCollider.cu
    src/SVCollider.cu
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/AbstractVoxel.h
    ${INCLUDE_DIR}/BitVoxel.h
    ${INCLUDE_DIR}/BitVoxel.cuhpp
    ${INCLUDE_DIR}/CountingVoxel.h
    ${INCLUDE_DIR}/DefaultCollider.h
    ${INCLUDE_DIR}/DefaultCollider.hpp
    ${INCLUDE_DIR}/DistanceVoxel.h
    ${INCLUDE_DIR}/DistanceVoxel.hpp
    ${INCLUDE_DIR}/ProbabilisticVoxel.h
    ${INCLUDE_DIR}/ProbabilisticVoxel.hpp
    ${INCLUDE_DIR}/SVCollider.h
    ${INCLUDE_DIR}/SVCollider.hpp
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
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
#target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_VOXEL_)
target_compile_options(${PROJECT_NAME} INTERFACE
    # If using CUDA w/ NVCC...
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Wno-deprecated-gpu-targets>
  )
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