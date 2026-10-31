# gpu_voxels unit tests (GoogleTest). Use gvl_add_test() from cmake/Tests.cmake for every test target.
include("${CMAKE_CURRENT_LIST_DIR}/test_categories.cmake")

# Backend-independent tests: always built.
gvl_add_test(test_build_config
  SOURCES smoke/test_build_config.cpp
  DEFINITIONS
    GVL_TEST_EXPECT_CUDA=$<BOOL:${GVL_BACKEND_CUDA}>
    GVL_TEST_EXPECT_VULKAN=$<BOOL:${GVL_BACKEND_VULKAN}>)

gvl_add_test(test_icl_core_enum_helper
  SOURCES smoke/test_icl_core_enum_helper.cpp
  LIBS icl_core)

# Host Morton code and the CPU octree reference (no GPU needed; the reference is the oracle of the octree tests).
gvl_add_test(test_octree_reference
  SOURCES octree/test_octree_reference.cpp octree/test_octree_dynamic_reference.cpp)
target_include_directories(test_octree_reference PRIVATE
  "${CMAKE_SOURCE_DIR}/packages/gpu_voxels/gpu_voxels/vulkan/include")

# Backend tests, guarded by the backend option.
if(GVL_BACKEND_VULKAN)
  add_subdirectory(vulkan)
endif()
# if(GVL_BACKEND_CUDA)
#   add_subdirectory(cuda)
# endif()
