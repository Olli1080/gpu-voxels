# Vulkan backend tests. They need a Vulkan 1.3 device and, by default, the Khronos validation layer (Vulkan SDK).
# Environment: GVL_ALLOW_NO_VALIDATION=1 permits running without the validation layer,
#              GVL_VULKAN_DEVICE=<index> selects a physical device.
gvl_add_test(test_vulkan
  SOURCES
    vulkan_fixture.cpp
    test_context.cpp
    test_bit_voxel_map.cpp
    test_buffer.cpp
    test_device_address.cpp
    test_device_address_faults.cpp
    test_dispatch.cpp
    test_distance_voxel_map.cpp
    test_distance_pba.cpp
    test_kinematic_chain.cpp
    test_meta_point_cloud.cpp
    test_octree_build.cpp
    test_octree_chunks.cpp
    test_octree_insert.cpp
    test_octree_insert_levels.cpp
    test_octree_insert_sort.cpp
    test_octree_rebuild.cpp
    test_octree_timings.cpp
    test_octree_balancer.cpp
    test_octree_checkpoint.cpp
    test_octree_collision.cpp
    test_octree_tree_collision.cpp
    test_morton.cpp
    test_primitives.cpp
    test_size_limits.cpp
    test_prob_sensor.cpp
    test_prob_voxel_map.cpp
    test_voxel_list.cpp
    test_voxel_list_ops.cpp
  LIBS gpu_voxels_vulkan
  DEFINITIONS
    "GVL_TEST_SHADER_DIR=\"${CMAKE_CURRENT_BINARY_DIR}/shaders\"")

gvl_add_shaders(test_vulkan
  OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders"
  INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/packages/gpu_voxels/gpu_voxels/vulkan/shaders"
  SOURCES
    shaders/add_one.comp
    shaders/bda_chase.comp
    shaders/bda_read.comp
    shaders/bda_write.comp
    shaders/morton_check.comp
    shaders/scale_copy.comp
    shaders/int64_mix.comp)
