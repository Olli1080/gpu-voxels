# Vulkan octree: sensor pipeline (slice S6f)

Point cloud insert with free space, built from the CUDA `GvlNTree::insertPointCloudWithFreespaceCalculation` (design and CUDA findings: [Analysis/2026-11-02_s6f_sensor_free_space_design.md](../../Analysis/2026-11-02_s6f_sensor_free_space_design.md)). User decisions 2026-11-02: point cloud path only (no Kinect depth images), the result need not be identical to CUDA, CUDA parity is not a goal here. **Status: S6f-1 of S6f-1 .. S6f-4 is done.**

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

## Verification
- `tests/vulkan/test_octree_sensor.cpp`, 6 cases: random clouds against a double-precision host reference (3 poses, 3 side lengths, 5 sizes, 6 sensor models, both tree kinds), counts around the clamp (1 .. 70000 points in one voxel), points exactly on voxel borders with exact arithmetic (identity and a permutation pose, negative zero, 2^14 border, the last voxel), outside points of every kind and degenerate poses (below zero, NaN entry, flat, last row ignored), edge cases (empty cloud, invalid side lengths, tree untouched, no state), large clouds (300000 points in 100000 voxels and in 8 voxels). The test points are built from a chosen world position inside a voxel and the inverse pose.
- Mutation runner `tools/mutation/mutate_s6f1_voxelize.py`: see [mutation_testing.md](../mutation_testing.md).
