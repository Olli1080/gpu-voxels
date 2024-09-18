project(gpu_voxels_core
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/core)

#generate_export_header(${PROJECT_NAME} 
#  EXPORT_MACRO_NAME
#    ICL_CORE_IMPORT_EXPORT
#  EXPORT_FILE_NAME
#    "${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}/ImportExport.h"
#)

target_include_directories(${PROJECT_NAME} PUBLIC include)

target_sources(${PROJECT_NAME}
  PRIVATE    
    src/GpuVoxelsMap.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/constants.hpp
    ${INCLUDE_DIR}/GpuVoxelsMap.h
    ${INCLUDE_DIR}/ManagedMap.h
    ${INCLUDE_DIR}/ManagedPrimitiveArray.h
#    ${INCLUDE_DIR}/ImportExport.h
)

set_source_files_properties(src/GpuVoxelsMap.cpp PROPERTIES LANGUAGE CUDA)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_logging
    gpu_voxels_helpers
    gpu_voxels_primitive_array
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_CORE_)

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