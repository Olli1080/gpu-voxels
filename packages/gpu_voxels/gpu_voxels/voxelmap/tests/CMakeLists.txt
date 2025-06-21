#------------- gpu_voxels_voxelmap_tests libaray -----------------------
ICMAKER_SET("gpu_voxels_voxelmap_tests" IDE_FOLDER ${GPU_VOXELS_IDE_FOLDER})

ICMAKER_ADD_HEADERS(
  Tests.h
  )

ICMAKER_ADD_SOURCES(
  Dummy.cpp
  )

ICMAKER_ADD_CUDA_FILES(
  kernels/VoxelMapOperations.h
  kernels/VoxelMapOperationsPBA.h
  kernels/VoxelMapOperations.hpp
  kernels/VoxelMapOperationsPBA.hpp
  kernels/VoxelMapOperations.cu
  kernels/VoxelMapTests.h
  kernels/VoxelMapTests.hpp
  kernels/VoxelMapTests.cu
  Tests.hpp
  Tests.cu
  )

ICMAKER_INTERNAL_DEPENDENCIES(
  icl_core
  icl_core_config
  icl_core_logging
  gpu_voxels_logging
  gpu_voxels_helpers
  gpu_voxels_core
  gpu_voxels_voxel
  )
