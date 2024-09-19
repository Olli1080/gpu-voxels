project(gpu_voxels_helpers
  LANGUAGES CXX CUDA
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/helpers)

#generate_export_header(${PROJECT_NAME} 
#  EXPORT_MACRO_NAME
#    ICL_CORE_IMPORT_EXPORT
#  EXPORT_FILE_NAME
#    "${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}/ImportExport.h"
#)
find_package(CUDAToolkit REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_sources(${PROJECT_NAME}
  PRIVATE
    #src/cuda_datatypes.cpp
    src/HeightMapLoader.cu
    src/PointcloudFileHandler.cpp
    src/BinvoxFileReader.cpp
    src/XyzFileReader.cpp
    src/MathHelpers.cpp
    src/GeometryGeneration.cu

    src/cuda_handling.cpp
    src/BitVector.cu
    src/PointCloud.cu
    src/MetaPointCloud.cu
    src/MetaPointCloudOperations.cu
    #src/HelperOperations.cu
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/cuda_vectors.hpp
    ${INCLUDE_DIR}/cuda_matrices.hpp
    ${INCLUDE_DIR}/cuda_datatypes.hpp
    ${INCLUDE_DIR}/common_defines.h
    ${INCLUDE_DIR}/cuda_handling.h
    ${INCLUDE_DIR}/cuda_handling.hpp
    ${INCLUDE_DIR}/HeightMapLoader.h
    ${INCLUDE_DIR}/PointcloudFileHandler.h
    ${INCLUDE_DIR}/BinvoxFileReader.h
    ${INCLUDE_DIR}/FileReaderInterface.h
    ${INCLUDE_DIR}/XyzFileReader.h
    ${INCLUDE_DIR}/PointCloud.h
    ${INCLUDE_DIR}/MetaPointCloud.h
    ${INCLUDE_DIR}/BitVector.cuhpp
    ${INCLUDE_DIR}/BitVector.h
    ${INCLUDE_DIR}/MathHelpers.h
    ${INCLUDE_DIR}/GeometryGeneration.h
    ${INCLUDE_DIR}/CollisionInterfaces.h

    ${INCLUDE_DIR}/cuda_handling.h
    ${INCLUDE_DIR}/cuda_handling.hpp
    ${INCLUDE_DIR}/PointCloud.h
    ${INCLUDE_DIR}/kernels/MetaPointCloudOperations.h
    #${INCLUDE_DIR}/kernels/HelperOperations.h

    #${INCLUDE_DIR}/kernels/HelperOperations.h
    #${INCLUDE_DIR}/kernels/MetaPointCloudOperations.h
#    ${INCLUDE_DIR}/ImportExport.h
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_logging 
)

find_package(Stb REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})

if(ON) #TODO:: replace with pcl question
  find_package(PCL REQUIRED COMPONENTS io)
  target_include_directories(${PROJECT_NAME} PRIVATE ${PCL_INCLUDE_DIRS})
  target_link_directories(${PROJECT_NAME} PRIVATE ${PCL_LIBRARY_DIRS})
  target_compile_definitions(${PROJECT_NAME} INTERFACE ${PCL_DEFINITIONS} PUBLIC _BUILD_GVL_WITH_PCL_SUPPORT_)

  target_sources(${PROJECT_NAME}
  PRIVATE
    src/PcdFileReader.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/PcdFileReader.h
  )

  if (ON) #TODO:: check for openni2

    target_compile_definitions(${PROJECT_NAME} PUBLIC _BUILD_GVL_WITH_KINECT_SUPPORT_)

    target_sources(${PROJECT_NAME}
    PRIVATE
      src/Kinect.cpp
    PUBLIC FILE_SET HEADERS 
    BASE_DIRS ${INCLUDE_DIR}
    FILES
      ${INCLUDE_DIR}/Kinect.h
    )
  endif()
endif()

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_HELPERS_)

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

if(OFF)

IF(ROS_FOUND)
  ICMAKER_GLOBAL_CPPDEFINES(-D_BUILD_GVL_WITH_TF_SUPPORT_)
  ICMAKER_ADD_SOURCES(
    tfHelper.cpp
  )
  ICMAKER_ADD_HEADERS(
    tfHelper.h
  )
  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
    ROS
  )
ENDIF(ROS_FOUND)

endif()