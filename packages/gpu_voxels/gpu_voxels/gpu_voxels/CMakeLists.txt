ADD_SUBDIRECTORY(robot)
ADD_SUBDIRECTORY(voxel)
ADD_SUBDIRECTORY(voxelmap)
ADD_SUBDIRECTORY(voxellist)
ADD_SUBDIRECTORY(vis_interface)
ADD_SUBDIRECTORY(octree)
ADD_SUBDIRECTORY(helpers_highlevel)

#------------- gpu_voxels library ---------------------------------
# This library resembles the high level interface to link against.
#------------------------------------------------------------------
ICMAKER_SET("gpu_voxels" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  GpuVoxels.h
  )

ICMAKER_ADD_SOURCES(
  GpuVoxels.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  )

# removing unknown pragma warnings due to OpenNI spam
ICMAKER_LOCAL_CPPDEFINES(-DGVL_EXPORT_SYMBOLS -Wno-unknown-pragmas)
ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_GPU_VOXELS_)
ICMAKER_INCLUDE_DIRECTORIES(${GPU_VOXELS_INCLUDE_DIRS})


ICMAKER_INTERNAL_DEPENDENCIES(EXPORT
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
  )

ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
  CUDA
  )


ICMAKER_DEPENDENCIES( OPTIONAL
  gpu_voxels_urdf_robot
)

ICMAKER_BUILD_LIBRARY()
ICMAKER_INSTALL_HEADERS(gpu_voxels)

ADD_SUBDIRECTORY(test)

