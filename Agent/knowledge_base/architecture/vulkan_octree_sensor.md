# Vulkan octree: sensor pipeline (slice S6f)

Point cloud insert with free space, built from the CUDA `GvlNTree::insertPointCloudWithFreespaceCalculation` (design and CUDA findings: [Analysis/2026-11-02_s6f_sensor_free_space_design.md](../../Analysis/2026-11-02_s6f_sensor_free_space_design.md)). User decisions 2026-11-02: point cloud path only (no Kinect depth images), the result need not be identical to CUDA, CUDA parity is not a goal here. **Status: S6f-1 .. S6f-3 of S6f-1 .. S6f-4 are done.**

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

## S6f-3: `Octree::pack_free_space` (2026-11-02)
`pack_free_space(const OctreeRayMap&) -> vector<vector<OctreeFreeCube{id, occupancy}>>` (CUDA `packVoxel_Map` with `kernel_packMortonL0Map`): one list per level of the free-space grid (index = level, 15 entries), each sorted by Morton node id.

| Step | What | Where |
|---|---|---|
| level 0 | the counts are copied into a dense map of the box extended to **even cell coordinates** (so every 2x2x2 group is complete; CUDA `MapProperties::align`: min rounded down and max up to a multiple of 2^(level+1), absolute alignment); value of a free cell: deterministic tree 1, probabilistic `max(-10 * count, -127)` (count limited to 255 first) | `octree_free_values.comp` |
| per level | one invocation per group: all 8 cells free **and equal** -> one cell of the next level's map (deterministic tree: every free cell has value 1, so "all free"; probabilistic: equal occupancy, CUDA `isPackingPossible`: max - min == 0), else every free cell of the group is emitted (Morton node id of the cell, value). Two passes of the same code: count (`count_only`), then write into exactly sized lists; the next map is cleared first | `octree_free_pack.comp`, `Octree::Impl::pack_free_cells` |
| sort | the emitted cells of a level are sorted by id (radix sort of 64 bit keys with the value as payload) | `Primitives::record_sort_pairs64` |

- **Same rule as CUDA, stated once:** a probabilistic cube exists only where all 8 children carry the same value; values are clamped counts, so 13 rays and 400 rays are equal (both -127), 1 and 2 rays are not. The rule of the deterministic tree is "all free".
- **The loop** runs while parents exist; level 14 (the whole grid) can have no parents (a group needs 8 cells; the check throws an internal error if it ever finds one). The cells of a level are cleared by construction (a group that does not pack leaves its parent cell 0).
- **Limit:** the box extended to even coordinates must not exceed `set_ray_map_limit` (a single cell at an odd position takes 2x2x2 cells); the first pass of level 0 is the biggest map, the next ones are 1/8 each.
- **Ids are node ids on their level**, not level-0 ids: `insert_nodes(level, ids, ...)` takes them as they are; with a free-space grid that is coarser than the tree by 2^s per side, a cube of level l is the tree node of level l + s with the same id (CUDA scales the level-0 ids by the same factor and inserts at level l + s), which S6f-4 uses.
- **Not done here:** inserting the cubes into the tree (S6f-4), where the probabilistic values need an `insert_nodes` with a value per node.

## Verification
- `tests/vulkan/test_octree_sensor.cpp`, 6 cases: random clouds against a double-precision host reference (3 poses, 3 side lengths, 5 sizes, 6 sensor models, both tree kinds), counts around the clamp (1 .. 70000 points in one voxel), points exactly on voxel borders with exact arithmetic (identity and a permutation pose, negative zero, 2^14 border, the last voxel), outside points of every kind and degenerate poses (below zero, NaN entry, flat, last row ignored), edge cases (empty cloud, invalid side lengths, tree untouched, no state), large clouds (300000 points in 100000 voxels and in 8 voxels). The test points are built from a chosen world position inside a voxel and the inverse pose.
- S6f-2: `test_octree_sensor.cpp`, 5 more cases (`OctreeRayCast*`): random voxel sets (5 sizes, 3 extents, 9 origins: inside, corner, far outside, on an axis) against the host translation of the CUDA walk (`ProbVoxelMapReference::walk`) summed per cell, in three chunk sizes; known rays (axis, origin voxel, diagonal, 100 voxels sharing cells, a pair where the walk to the origin differs from the walk from it); argument errors and the limit (100 cells pass, 99 do not); the grid-length rays; 40000 rays in many chunks; the number of submissions per chunk size.
- S6f-3: 5 more cases (`OctreePack*`, `OctreeRaysAndPacking`): random maps (6 positions incl. the top corner of the grid, 7 sizes, 5 densities, 4 count ranges, both kinds) against a sparse host reference built on `std::map` groups; known cases (aligned 8-cube, the same shifted by one cell, a 2x2x2 group with unequal / clamped counts, one hole, a 64-cube at the top corner, empty maps); a 128^3 box over 7 levels; argument errors and the limit; rays then packing (host reference rays and GPU rays give the same cubes).
- Mutation runners `tools/mutation/mutate_s6f1_voxelize.py`, `mutate_s6f2_ray_cast.py` and `mutate_s6f3_pack.py`: see [mutation_testing.md](../mutation_testing.md).
