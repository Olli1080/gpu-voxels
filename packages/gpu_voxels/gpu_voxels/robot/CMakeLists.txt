project(gpu_voxels_robot
  LANGUAGES CXX CUDA
)
add_library(gpu_voxels_dh_robot)

set(INCLUDE_DIR include/gpu_voxels/robot/dh_robot)

#generate_export_header(${PROJECT_NAME} 
#  EXPORT_MACRO_NAME
#    ICL_CORE_IMPORT_EXPORT
#  EXPORT_FILE_NAME
#    "${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}/ImportExport.h"
#)

target_include_directories(gpu_voxels_dh_robot PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(gpu_voxels_dh_robot
  PRIVATE    
    src/KinematicLink.cpp
    src/KinematicChain.cu
    include/gpu_voxels/robot/robot_interface.h
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/KinematicLink.h
    ${INCLUDE_DIR}/KinematicChain.h
#    ${INCLUDE_DIR}/ImportExport.h
)
set_property(TARGET gpu_voxels_dh_robot PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET gpu_voxels_dh_robot PROPERTY CUDA_ARCHITECTURES all)
target_compile_options(gpu_voxels_dh_robot INTERFACE
    # If using CUDA w/ NVCC...
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Wno-deprecated-gpu-targets>
  )

target_link_libraries(gpu_voxels_dh_robot
  PUBLIC
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_logging
    gpu_voxels_helpers
)

target_compile_features(gpu_voxels_dh_robot PUBLIC cxx_std_20)
target_compile_definitions(gpu_voxels_dh_robot PUBLIC _IC_BUILDER_GPU_VOXELS_DH_ROBOT_)

install(
  TARGETS 
    gpu_voxels_dh_robot
  EXPORT
    gpu_voxels-targets
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${INCLUDE_DIR}
)

###########

install(FILES include/gpu_voxels/robot/robot_interface.h
        DESTINATION include/gpu_voxels/robot)

###########
if(OFF)
find_package(urdfdom CONFIG REQUIRED)
find_package(orocos_kdl CONFIG REQUIRED)


add_library(gpu_voxels_urdf_robot)

set(INCLUDE_DIR include/gpu_voxels/robot/urdf_robot)

#generate_export_header(${PROJECT_NAME} 
#  EXPORT_MACRO_NAME
#    ICL_CORE_IMPORT_EXPORT
#  EXPORT_FILE_NAME
#    "${CMAKE_CURRENT_SOURCE_DIR}/${INCLUDE_DIR}/ImportExport.h"
#)

target_include_directories(gpu_voxels_urdf_robot PUBLIC include)

target_sources(gpu_voxels_urdf_robot
  PRIVATE    
    src/robot_to_gpu.cu
    src/node.cpp
    src/robot.cpp
    src/robot_joint.cpp
    src/robot_link.cpp
    src/urdf_robot.cpp
    include/gpu_voxels/robot/robot_interface.h
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/node.h
    ${INCLUDE_DIR}/robot.h
    ${INCLUDE_DIR}/robot_joint.h
    ${INCLUDE_DIR}/robot_link.h
    ${INCLUDE_DIR}/urdf_robot.h
    ${INCLUDE_DIR}/robot_to_gpu.h
#    ${INCLUDE_DIR}/ImportExport.h
)

target_link_libraries(gpu_voxels_urdf_robot
  PRIVATE
    icl_core
    icl_core_config
    icl_core_logging
    gpu_voxels_logging
    gpu_voxels_helpers
    urdfdom::urdf_parser urdfdom::urdfdom_model urdfdom::urdfdom_world urdfdom::urdfdom_sensor
    orocos-kdl
    #kdl_parser
)

target_compile_features(gpu_voxels_urdf_robot PUBLIC cxx_std_20)
target_compile_definitions(gpu_voxels_urdf_robot 
    PUBLIC 
        _BUILD_GVL_WITH_URDF_SUPPORT_
        _IC_BUILDER_GPU_VOXELS_URDF_ROBOT_)

install(
  TARGETS 
    gpu_voxels_urdf_robot
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${INCLUDE_DIR}
)



#add_subdirectory(tests)
#ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})
endif()