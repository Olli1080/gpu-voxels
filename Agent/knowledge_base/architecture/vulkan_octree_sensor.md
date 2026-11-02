# Vulkan octree: sensor pipeline (slice S6f)

Point cloud insert with free space, built from the CUDA `GvlNTree::insertPointCloudWithFreespaceCalculation` (design and CUDA findings: [Analysis/2026-11-02_s6f_sensor_free_space_design.md](../../Analysis/2026-11-02_s6f_sensor_free_space_design.md)). User decisions 2026-11-02: point cloud path only (no Kinect depth images), the result need not be identical to CUDA, CUDA parity is not a goal here. **Status: S6f-1 and S6f-2 of S6f-1 .. S6f-4 are done.**

## S6f-1: `Octree::voxelize_points` (2026-11-02)
`voxelize_points(points, pose, voxel_side_length, OctreeSensorModel{initial, update}, uint32_t* outside) -> vector<OctreeVoxelRun{id, count, occupancy}>`

| Step | What | Where |
|---|---|---|
| pose and voxel | `world = R * p + t` (first three rows of the row-major `Matrix4f`, the last row is ignored, as `Sensor::sensorCoordinatesToWorldCoordinates`), voxel = `floor(world / side)`, Morton id; a point outside the 2^14 grid (negative, too far, NaN, inf, also through the pose) gets the sentinel `2^42` and is counted | `octree_sensor_voxels.comp` |
| sort | radix sort of the ids (43 bits, the sentinel sorts behind every valid id) | `Primitives::record_sort_pairs64` |
| distinct | heads, exclusive scan, compaction (the code of the facade) | `record_distinct` |
| runs | `starts[k]` = position of the first id of run k, `counts[k]` = run length, `occupancy[k] = clamp(initial + update * count, -127, 127)` | `octree_run_starts.comp`, `octree_run_counts.comp` |

- **The side length is a parameter**, not the one of the tree: the pipeline voxelizes the same cloud on the free-space grid and on the object grid (CUDA: two `SensorDataProcessing::m_voxel_side_length`), and the tree is not touched.
- **Occupancy:** CUDA `SensorModel::estimateVoxelProbability(count)`. The shader limits `count` to 255 before the product: with `|update| >= 1` that already reaches the clamp from every initial value (`update = 0` does not depend on the count), so the product cannot overflow; the test compares with a reference that uses the unclamped count.
- **Differences to CUDA:** a negative voxel coordinate is dropped (CUDA truncates a negative float to uint: undefined); `removeInvalidPoints` is the NaN / inf drop above; the unit is metres and the side length a float (CUDA: millimetres, integer); the order of the float operations differs (`dot + t`, GPU may contract to fma), so a point within about 1e-6 relative of a voxel border can land in the neighbour voxel.
- Returns host vectors (tests, small clouds). The slices after this one use the device buffers of `Impl::voxelize` (`VoxelRuns`) directly.

## S6f-2: `Octree::cast_free_space_rays` (2026-11-02)
`cast_free_space_rays(free_voxels, origin) -> OctreeRayMap{min, size, counts}` (CUDA `computeFreeSpaceViaRayCast` + `kernel_rayInsert`, without the packing, which is S6f-3).

| Step | What | Where |
|---|---|---|
| box | smallest and largest x, y, z of all voxels (atomics on the ids decoded on the GPU), plus the origin on the host; `cells` = product of the sizes, refused above `set_ray_map_limit` (default 2^28 cells = 1 GiB) before anything is allocated | `octree_ray_bounds.comp` |
| rays | one thread per voxel walks to the origin: 6-connected, `1 + dx + dy + dz` cells including both end points (the walk of CUDA `RayCaster::rayCast`, the same as `prob_ray.glsl`, but from the voxel to the origin as in the octree kernel: from and to are swapped there, and the other direction visits other cells); every visited cell gets `atomicAdd(counts, 1)` | `octree_ray_cast.comp` |
| chunks | the rays are cast in chunks of `ray_chunk_steps / longest possible ray` rays (default 2^28 steps), one GPU submission each, so a long walk of many rays stays far below the driver watchdog (K25); `test_ray_submissions()` makes the chunking observable | `Octree::Impl::cast_rays` |

- **Exact counts instead of CUDA's racy writes:** CUDA writes `ns_FREE` (deterministic) or `v + max(-10, -127 - v)` (probabilistic) into a byte map without atomics and loses hits. The count here is exact; S6f-3 turns it into the value the probabilistic tree gets: `max(-10 * count, -127)`, which is what CUDA computes without races.
- **Dense box at the free-space grid**, like CUDA; the grid of the free space is coarse (a power of two times the tree resolution), so the box is small in practice. The limit is an addition (CUDA fails in `cudaMalloc`); a sparse map would only pay off for fine free-space grids.
- **CUDA bug not copied:** `kernel_rayInsert` swaps `from` and `to` in the grid-stride loop but re-assigns only `to`, so from the second voxel of a thread on the ray runs between two free-space voxels instead of to the origin (visible only with more voxels than threads); here every voxel walks to the origin.
- **Not done here:** the origin is a voxel on the free-space grid (the pipeline computes it from the pose in S6f-3: CUDA divides the origin by the free-space scale in integers); voxels are not sorted or deduplicated (every voxel casts one ray).

## Verification
- `tests/vulkan/test_octree_sensor.cpp`, 6 cases: random clouds against a double-precision host reference (3 poses, 3 side lengths, 5 sizes, 6 sensor models, both tree kinds), counts around the clamp (1 .. 70000 points in one voxel), points exactly on voxel borders with exact arithmetic (identity and a permutation pose, negative zero, 2^14 border, the last voxel), outside points of every kind and degenerate poses (below zero, NaN entry, flat, last row ignored), edge cases (empty cloud, invalid side lengths, tree untouched, no state), large clouds (300000 points in 100000 voxels and in 8 voxels). The test points are built from a chosen world position inside a voxel and the inverse pose.
- S6f-2: `test_octree_sensor.cpp`, 5 more cases (`OctreeRayCast*`): random voxel sets (5 sizes, 3 extents, 9 origins: inside, corner, far outside, on an axis) against the host translation of the CUDA walk (`ProbVoxelMapReference::walk`) summed per cell, in three chunk sizes; known rays (axis, origin voxel, diagonal, 100 voxels sharing cells, a pair where the walk to the origin differs from the walk from it); argument errors and the limit (100 cells pass, 99 do not); the grid-length rays; 40000 rays in many chunks; the number of submissions per chunk size.
- Mutation runners `tools/mutation/mutate_s6f1_voxelize.py` and `mutate_s6f2_ray_cast.py`: see [mutation_testing.md](../mutation_testing.md).
