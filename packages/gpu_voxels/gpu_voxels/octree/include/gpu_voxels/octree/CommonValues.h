#ifndef __COMMON_VALUES_H_
#define __COMMON_VALUES_H_

#ifdef __CUDACC_VER_MAJOR__
#define HOST_INLINE 
#else
#define HOST_INLINE inline
#endif

HOST_INLINE constexpr int NUM_BLOCKS = 2688; // 8192 * 8;
HOST_INLINE constexpr int NUM_THREADS_PER_BLOCK = 128; // 32 // 32 * 8
HOST_INLINE constexpr int THRESHOLD_OCCUPANCY = 10;

//#define PROBABILISTIC_TREE
HOST_INLINE constexpr bool PACKING_OF_VOXEL = true;

#define DO_REBUILDS
#define PERFORMANCE_MEASUREMENT

// only include declaration of template class NTree.hpp once to speed up the build process
#define NTREE_PRECOMPILE

#undef TREAT_UNKNOWN_AS_COLLISION // When defined, intersections with unknown octree nodes will result in collisions

HOST_INLINE constexpr int VISUALIZER_SHIFT_X = 0;//7600
HOST_INLINE constexpr int VISUALIZER_SHIFT_Y = 0;//7600
HOST_INLINE constexpr int VISUALIZER_SHIFT_Z = 0;//8000

HOST_INLINE constexpr gpu_voxels::Probability INITIAL_PROBABILITY = static_cast<gpu_voxels::Probability>(0);  // probability used to initialize any new node
HOST_INLINE constexpr gpu_voxels::Probability INITIAL_FREE_SPACE_PROBABILITY = static_cast<gpu_voxels::Probability>(0);
HOST_INLINE constexpr gpu_voxels::Probability INITIAL_OCCUPIED_PROBABILITY = static_cast<gpu_voxels::Probability>(0);
HOST_INLINE constexpr gpu_voxels::Probability FREE_UPDATE_PROBABILITY = static_cast<gpu_voxels::Probability>(-10);
HOST_INLINE constexpr gpu_voxels::Probability OCCUPIED_UPDATE_PROBABILITY = static_cast<gpu_voxels::Probability>(100);

HOST_INLINE constexpr int KINECT_CUT_FREE_SPACE_Y = 0;//50
HOST_INLINE constexpr int KINECT_CUT_FREE_SPACE_X = 0;//80
HOST_INLINE constexpr bool KINECT_FREE_NAN_MEASURES = true;
//#define VISUALIZER_OBJECT_DATA_ONLY
//#define KINECT_FREE_SPACE_DEBUG
//#define KINECT_FREE_SPACE_DEBUG_2
HOST_INLINE constexpr int KINECT_WIDTH = 640;
HOST_INLINE constexpr int KINECT_HEIGHT = 480;

//#define KINECT_ORIENTATION Vector3f(M_PI / 2.0f, M_PI, 0) // up-side-down on PTU (roll, pitch yaw)
#define KINECT_ORIENTATION Vector3f(M_PI_2<float>, 0, 0); // normal (roll, pitch yaw)
#define INVALID_POINT Vector3ui(UINT_MAX, UINT_MAX, UINT_MAX);
HOST_INLINE constexpr int VOXELMAP_FLAG_SIZE = 1;

#endif