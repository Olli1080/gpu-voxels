# S6f: point cloud insert with free space (design from the CUDA sources)

- **Date:** 2026-11-02
- **Question:** What does `GvlNTree::insertPointCloudWithFreespaceCalculation` do exactly, and which parts do we build on the Vulkan octree (user decisions 2026-11-02: the point cloud path, no Kinect depth images, the result need not be bit-identical to CUDA, CUDA parity is not a goal here)?
- **Scope / non-goals:** semantics read from the CUDA sources (**inferred**, nothing was run under CUDA). Out: Kinect depth image preprocessing (`_processDepthImage`, `kernel_preprocessDepthImage`, border cuts), viewer, ROS. No code of S6f exists yet.
- **Status:** in progress: S6f-1 is done (see [vulkan_octree_sensor.md](../knowledge_base/architecture/vulkan_octree_sensor.md)); S6f-2 .. S6f-4 open

## Method
Read: `octree/NTree.hpp` (`insertVoxel` sensor variant ~2261, `computeFreeSpaceViaRayCast` ~1935, `packVoxel_Map` ~1568), `kernels/kernel_Octree.h` (`kernel_rayInsert` ~1750, `kernel_packMortonL0Map` ~2035), `kernels/kernel_common.h` (`handleRayHit`, `isPackingPossible`, `maxMinReduction`, `getRayCastInit`), `src/Sensor.cu` (`processSensorData(const Vector3f*)`, `_processSensorData`), `src/PointCloud.cu` + `src/kernel_PointCloud.cu` (`transformKinectPointCloud_simple`, `kernel_voxelize*`), `SensorModel.h`, `CommonValues.h`. Not read: `MapProperties::align()` / `createNextLevelMap()` / `computeMortonOffsetPtr`.

## Evidence and findings
1. **One point cloud, two voxelizations (inferred, high).** `processSensorData` copies the points into a free-space and an object cloud and runs `transformKinectPointCloud_simple` on each with its own voxel side length (`SensorDataProcessing::m_voxel_side_length`, mm). Per point: sensor to world with the pose, metres to mm, divide by the side length, truncate to uint (negative coordinates only `assert`ed), Morton code; sort; equal voxels are combined. Object voxel occupancy = `estimateVoxelProbability(points in the voxel)` = `clamp(initial + update * count, MIN_PROBABILITY, MAX_PROBABILITY)` (sensor model of the object data). If object and free-space settings are equal the free-space voxels are a copy of the object voxels.
2. **The ray cast runs on the free-space grid (inferred, high).** `free_space_scale = free_res / m_resolution` (a power of two, only `assert`ed), `sensor_origin_scaled = sensor_origin / free_space_scale` (integer division). Dense map over the bounding box of all free-space voxels and the origin, one cell per voxel of the free-space grid.
3. **The ray is a 6-connected line (inferred, high).** `kernel_rayInsert` walks from every free-space voxel to the origin, `n = dx + dy + dz + 1` cells, both end points included (same walk as `ProbVoxelMap` in S3). The cell of the hit itself is therefore free in the map.
4. **Cell value (inferred, high).** Deterministic tree: `ns_FREE`, valid = not equal to the initial 0. Probabilistic tree: `handleRayHit` does `v = v + max(FREE_UPDATE_PROBABILITY (-10), MIN_PROBABILITY (-127) - v)` per ray hit without atomics ("concurrent writes ... a conflict shouldn't happen that often"), start 0 = no ray. So the value after `n` hits is `max(-10 n, -127)` (a clamped count); CUDA loses hits to races, we count exactly.
5. **Packing (inferred, medium).** Level by level, 2x2x2 groups of cells are replaced by one cell of the next coarser map iff all 8 are valid (det) or all 8 are valid **and have the same value** (prob: `max - min == 0`). A group that is not packed emits its valid cells at this level. Result: free space as the biggest possible aligned cubes, sorted by Morton id per level (CUB / thrust sort). The alignment of the map origin (`align()`) was not read; the only meaning that makes node ids right is "groups aligned in absolute coordinates", and that is what the reference will implement.
6. **Insert order (inferred, high).** For level `l` of the packed lists: ids are scaled to the base grid (`Trafo_OctreeVoxelID(free_space_scale)`), then `insertVoxel(ids, data, soft-insert reset data, level = log2(scale) + l)`; the data are the cell values of 4. (`BasicData(value, 0)`; for the deterministic tree only the id matters). After all free levels the object voxels are inserted (hard) with their occupancy, then propagate. Free space first, so occupied voxels win inside a coarse free cube that contains a hit.
7. **CUDA quirks we do not copy (inferred, high):** (a) `kernel_rayInsert` swaps `from`/`to` inside the grid-stride loop but re-assigns only `to`: from the second voxel of a thread on, the ray runs between two free-space voxels, not to the origin (visible only if there are more voxels than threads); (b) the races of 4.; (c) integer division of the origin loses the sub-voxel position; (d) the dense map allocates the whole bounding box in the free-space grid and has no limit; (e) the grid-stride `to` in `kernel_countVoxel` (`from + chunk + (id < overhead) ? 1 : 0`, precedence) is in `transformKinectPointCloud`, which `_processSensorData` does not use.
8. **Already on the Vulkan side (measured, S3 / S6d / S6e):** the 6-connected walk and its CPU reference (`ProbVoxelMap::insert_sensor_data`), `insert_nodes(level, ids, datum, Hard|Soft)` for both kinds, `eBVM_FREE`, the GPU sort / dedupe of ids (S6d-5), `insert_point_cloud` with voxelization on the GPU (S6e), exact and bounded propagate.
9. **Missing:** (a) `insert_nodes` takes a datum (Free / Occupied), not a value per node: the probabilistic free space needs a value per node (-10 .. -127); (b) object occupancy from a count (the `estimateVoxelProbability` rule) when inserting points; (c) the pose; (d) ray cast and packing.

## Design (what we build)
- **Dense count map at the free-space resolution** (as in CUDA, the box is small because the free-space grid is coarse), `uint32` per cell, `atomicAdd` per visited cell: deterministic counts. A budget (default 1 GiB, settable) turns "box too big" into a clear `VulkanError` before allocating; sparse maps are a later step if real data need them. (An earlier idea of this session, a sparse sort-and-dedupe of the ray cells, would need more memory than the dense map for long rays and is not better at coarse free-space grids.)
- **Packing on the GPU** level by level with the same dense maps (each 1/8 the size of the last), emitting per level the ids and values of the cells that stay.
- **Facade** `Octree::insert_point_cloud_with_free_space(points, pose, SensorFreeSpaceOptions{free_res_levels, sensor model, range}, propagate_now)`.
- **Reference:** a plain CPU implementation of findings 1 - 6 (dense byte map, same walk, same packing rule, node inserts into `DynamicOctreeReference`); the Vulkan result must equal the reference tree exactly. This is a reference of *our reading of CUDA*, not of CUDA.

## Slices
| Slice | Content | Check |
|---|---|---|
| S6f-1 | pose + voxelize at a given side length -> sorted unique ids + counts (+ occupancy from count) | CPU reference, random clouds, outside points |
| S6f-2 | ray cast into the count map | reference walk, lines in all octants, origin inside / outside the cloud box, budget error |
| S6f-3 | packing into cubes per level with values | reference packing, prob rule "equal values", alignment |
| S6f-4 | `insert_nodes` with a value per node (prob), facade, whole pipeline against the reference tree, `check_tree`, repeated insertions | tree equality; mutation runner |
| later | depth image to points (camera model, invalid pixels as max range), `free_bounding_box` of `build` | only on request |

## What was NOT verified
Everything about CUDA behaviour (no CUDA here); `MapProperties::align()`; the default sensor settings (`SensorDataProcessing` defaults); that real sensor data give boxes within the budget.

## Recommendation / decision needed
Start with S6f-1. Decisions for the user at the end of S6f-4: the default for the box budget, and whether a sparse map is worth building.

## Promoted knowledge
[vulkan_octree_sensor.md](../knowledge_base/architecture/vulkan_octree_sensor.md) (grows with every slice).
