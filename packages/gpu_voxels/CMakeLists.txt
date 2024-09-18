add_subdirectory(gpu_voxels/logging)
add_subdirectory(gpu_voxels/helpers)
add_subdirectory(gpu_voxels/primitive_array)
add_subdirectory(gpu_voxels/core)
add_subdirectory(gpu_voxels/voxel)
add_subdirectory(gpu_voxels/voxelmap)
add_subdirectory(gpu_voxels/voxellist)
add_subdirectory(gpu_voxels/octree)
add_subdirectory(gpu_voxels/robot)
add_subdirectory(gpu_voxels/vis_interface)
add_subdirectory(gpu_voxels/helpers_highlevel)
add_subdirectory(gpu_voxels/gpu_voxels)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/gpu_voxels-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/gpu_voxels-config.cmake"
  INSTALL_DESTINATION share/gpu_voxels
)

install(FILES
          "${CMAKE_CURRENT_BINARY_DIR}/gpu_voxels-config.cmake"
        DESTINATION share/gpu_voxels)

install(
  EXPORT gpu_voxels-targets
  FILE gpu_voxels-targets.cmake
  DESTINATION share/gpu_voxels
)