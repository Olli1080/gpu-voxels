project(gpu_voxels_vulkan
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/vulkan)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
  PRIVATE
    src/BitVoxelMap.cpp
    src/CommandBatch.cpp
    src/ComputePipeline.cpp
    src/Context.cpp
    src/DeviceBuffer.cpp
    src/DistanceVoxelMap.cpp
    src/KinematicChain.cpp
    src/Octree.cpp
    src/VoxelList.cpp
    src/MetaPointCloud.cpp
    src/ProbVoxelMap.cpp
    src/Primitives.cpp
    src/vma_impl.cpp
  PUBLIC FILE_SET HEADERS
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/BitVoxelMap.h
    ${INCLUDE_DIR}/CommandBatch.h
    ${INCLUDE_DIR}/ComputePipeline.h
    ${INCLUDE_DIR}/Context.h
    ${INCLUDE_DIR}/DeviceBuffer.h
    ${INCLUDE_DIR}/DistanceVoxelMap.h
    ${INCLUDE_DIR}/KinematicChain.h
    ${INCLUDE_DIR}/Morton.h
    ${INCLUDE_DIR}/MetaPointCloud.h
    ${INCLUDE_DIR}/Octree.h
    ${INCLUDE_DIR}/OctreeFormat.h
    ${INCLUDE_DIR}/Primitives.h
    ${INCLUDE_DIR}/ProbVoxelMap.h
    ${INCLUDE_DIR}/VoxelList.h
    ${INCLUDE_DIR}/vulkan_types.h
    ${INCLUDE_DIR}/vulkan_common.h
)

# Vulkan::Vulkan: loader + headers (vulkan.hpp). VMA is an implementation detail of the .cpp files.
target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Vulkan::Vulkan
    icl_core_logging
    gpu_voxels_logging
  PRIVATE
    GPUOpen::VulkanMemoryAllocator
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Kernels: GLSL -> SPIR-V at build time, embedded into the library (no shader files needed at run time).
gvl_embed_shaders(${PROJECT_NAME}
  INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/shaders
  SOURCES
    shaders/bit_voxel_clear_meaning.comp
    shaders/bit_voxel_collide.comp
    shaders/bit_voxel_collide_prob.comp
    shaders/bit_voxel_collide_types.comp
    shaders/bit_voxel_insert_coords.comp
    shaders/bit_voxel_insert_meta.comp
    shaders/bit_voxel_insert_meta_selfcoll.comp
    shaders/bit_voxel_insert_points.comp
    shaders/dist_edt_line.comp
    shaders/dist_edt_z.comp
    shaders/dist_extract.comp
    shaders/dist_insert.comp
    shaders/dist_merge_occupied.comp
    shaders/dist_query.comp
    shaders/fill_iota.comp
    shaders/list_collide_map.comp
    shaders/list_compact.comp
    shaders/list_entries_coords.comp
    shaders/list_entries_meta.comp
    shaders/list_entries_points.comp
    shaders/list_flags_count_ge.comp
    shaders/list_flags_map_occupied.comp
    shaders/list_gather_map.comp
    shaders/list_membership.comp
    shaders/list_normalize_counts.comp
    shaders/list_unique_heads.comp
    shaders/list_unique_merge.comp
    shaders/meta_transform.comp
    shaders/meta_transform_inplace.comp
    shaders/octree_clear_collision.comp
    shaders/octree_collide_det.comp
    shaders/octree_collide_prob.comp
    shaders/octree_collide_trees_det.comp
    shaders/octree_collide_trees_prob.comp
    shaders/octree_lb_count.comp
    shaders/octree_lb_move.comp
    shaders/octree_lb_trees_det.comp
    shaders/octree_lb_trees_prob.comp
    shaders/octree_lb_extract_det.comp
    shaders/octree_lb_extract_prob.comp
    shaders/octree_build_det.comp
    shaders/octree_build_prob.comp
    shaders/octree_compact_ids.comp
    shaders/octree_extract_det.comp
    shaders/octree_extract_prob.comp
    shaders/octree_find_det.comp
    shaders/octree_find_prob.comp
    shaders/octree_heads.comp
    shaders/octree_insert_walk_det.comp
    shaders/octree_insert_walk_prob.comp
    shaders/octree_insert_heads.comp
    shaders/octree_insert_count.comp
    shaders/octree_insert_link_det.comp
    shaders/octree_insert_link_prob.comp
    shaders/octree_insert_leaves_det.comp
    shaders/octree_insert_nodes_det.comp
    shaders/octree_insert_nodes_prob.comp
    shaders/octree_insert_leaves_prob.comp
    shaders/octree_propagate_down_det.comp
    shaders/octree_propagate_down_prob.comp
    shaders/octree_propagate_leaves_det.comp
    shaders/octree_propagate_leaves_prob.comp
    shaders/octree_propagate_up_det.comp
    shaders/octree_propagate_up_prob.comp
    shaders/octree_compact_scan_det.comp
    shaders/octree_compact_scan_prob.comp
    shaders/octree_compact_copy_det.comp
    shaders/octree_compact_copy_prob.comp
    shaders/octree_morton_ids.comp
    shaders/pba_p1_flood.comp
    shaders/pba_p1_propagate.comp
    shaders/pba_p1_update.comp
    shaders/pba_p2_backpointers.comp
    shaders/pba_p2_forward.comp
    shaders/pba_p2_merge.comp
    shaders/pba_p3_distances.comp
    shaders/pba_sanitize.comp
    shaders/pba_transpose.comp
    shaders/prob_apply_accumulated.comp
    shaders/prob_collide.comp
    shaders/prob_events_count.comp
    shaders/prob_events_fold.comp
    shaders/prob_events_write.comp
    shaders/prob_points_apply.comp
    shaders/radix_histogram.comp
    shaders/radix_histogram64.comp
    shaders/radix_scatter.comp
    shaders/radix_scatter64.comp
    shaders/scan_add.comp
    shaders/scan_blocks.comp)

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
