project(gpu_voxels_octree
  LANGUAGES CXX CUDA
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/octree)

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
    src/EnvironmentNodes.cu
    src/kernel_PointCloud.cu
    src/Octree.cu
    src/PointCloud.cu
    src/PointCloudWrapper.cu
    src/RobotNodes.cu
    src/Sensor.cu
    src/VisNTree.cu
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/kernels/kernel_common.h
    ${INCLUDE_DIR}/kernels/kernel_Octree.h
    ${INCLUDE_DIR}/kernels/kernel_PointCloud.h
    ${INCLUDE_DIR}/kernels/kernel_traverse.h
    ${INCLUDE_DIR}/kernels/shared_voxel.cuh
    ${INCLUDE_DIR}/load_balancer/AbstractLoadBalancer.cuh
    ${INCLUDE_DIR}/load_balancer/AbstractLoadBalancer.h
    ${INCLUDE_DIR}/load_balancer/Extract.cuh
    ${INCLUDE_DIR}/load_balancer/Extract.h
    ${INCLUDE_DIR}/load_balancer/Intersect.cuh
    ${INCLUDE_DIR}/load_balancer/Intersect.h
    ${INCLUDE_DIR}/load_balancer/IntersectVMap.cuh
    ${INCLUDE_DIR}/load_balancer/IntersectVMap.h
    ${INCLUDE_DIR}/load_balancer/LoadBalancer.cuh
    ${INCLUDE_DIR}/load_balancer/Propagate.cuh
    ${INCLUDE_DIR}/load_balancer/Propagate.h
    ${INCLUDE_DIR}/load_balancer/kernel_config/Extract.cuh
    ${INCLUDE_DIR}/load_balancer/kernel_config/Intersect.cuh
    ${INCLUDE_DIR}/load_balancer/kernel_config/IntersectVMap.cuh
    ${INCLUDE_DIR}/load_balancer/kernel_config/LoadBalance.cuh
    ${INCLUDE_DIR}/load_balancer/kernel_config/Propagate.cuh
    ${INCLUDE_DIR}/CommonValues.h
    ${INCLUDE_DIR}/DataTypes.h
    ${INCLUDE_DIR}/DefaultCollider.h
    ${INCLUDE_DIR}/EnvironmentNodes.h
    ${INCLUDE_DIR}/EnvNodesProbabilistic.h
    ${INCLUDE_DIR}/EnvNodesProbCommon.h
    ${INCLUDE_DIR}/GvlNTree.h
    ${INCLUDE_DIR}/GvlNTree.hpp
    ${INCLUDE_DIR}/Morton.h
    ${INCLUDE_DIR}/Nodes.h
    ${INCLUDE_DIR}/NTree.h
    ${INCLUDE_DIR}/NTree.hpp
    ${INCLUDE_DIR}/NTreeData.h
    ${INCLUDE_DIR}/Octree.h
    ${INCLUDE_DIR}/PointCloud.h
    ${INCLUDE_DIR}/PointCloudWrapper.h
    ${INCLUDE_DIR}/RobotNodes.h
    ${INCLUDE_DIR}/Sensor.h
    ${INCLUDE_DIR}/SensorModel.h
    ${INCLUDE_DIR}/VisNTree.h
    ${INCLUDE_DIR}/VisNTree.hpp
    ${INCLUDE_DIR}/Voxel.h
#    ${INCLUDE_DIR}/ImportExport.h
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    gpu_voxels_voxel
  PRIVATE
    gpu_voxels_dh_robot
    gpu_voxels_core
    gpu_voxels_voxelmap # REMOVE THIS LATER
    gpu_voxels_voxellist # REMOVE THIS LATER
    gpu_voxels_helpers
    gpu_voxels_logging
    gpu_voxels_vis_interface
    icl_core_performance_monitor
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)
target_compile_definitions(${PROJECT_NAME} PUBLIC _IC_BUILDER_GPU_VOXELS_OCTREE_)

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

# enable performance monitor
#add_definitions(-DIC_PERFORMANCE_MONITOR)

#ADD_SUBDIRECTORY(test)