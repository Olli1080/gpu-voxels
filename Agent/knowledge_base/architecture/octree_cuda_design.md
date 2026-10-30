# CUDA octree (`NTree` / `GvlNTree`) — design and behaviour

Facts about `packages/gpu_voxels/gpu_voxels/octree` on `master` (read 2026-10-29, nothing executed: no CUDA here). Evidence tags: **measured** = counted by command, **inferred** = read in code. Decision material built on this page: [Analysis: octree design review](../../Analysis/2026-10-29_octree_design_review/summary.md). Porting problems: [octree_porting_hazards.md](../vulkan_migration/octree_porting_hazards.md).

## Size (measured, lines)
| Part | Lines |
|---|---|
| `NTree.hpp` (all operations) / `NTree.h` | 3380 / 549 |
| `GvlNTree.{h,hpp}` (façade: `GpuVoxelsMap` + collision interfaces) | 711 |
| kernels `kernels/*` (`kernel_Octree.h` 2633, `kernel_common.h` 1097, point cloud, traverse) | 4145 |
| load balancer (`*.cuh/.h` 1382 + `kernel_config/*` 1900) | 3282 |
| nodes, data types, Morton, voxel, work-item structs | 2230 |
| `src/*.cu` (instantiations, point cloud, sensor, robot/environment nodes) + `Sensor.h`/`PointCloud*.h` | 1403 + 499 |
| **non-test total** | **≈16 200** |
| tests (`test/*`, depend on octomap providers, ICMAKER-only CMake) | 5790 |

## Instantiations and limits (inferred)
`BRANCHING_FACTOR = 8`, `LEVEL_COUNT = 15` (`common_defines.h:199-200`); two node families: deterministic (`Environment::InnerNode/LeafNode`, `GvlNTreeDet`, map type `MT_BITVECTOR_OCTREE`) and probabilistic (`...Prob`, `GvlNTreeProb`, `MT_PROBAB_OCTREE`); an extra 9-level instantiation exists for tests. Voxel ids are Morton codes (`OctreeVoxelID = uint64_t`); `build` asserts `id < 8^(levels-1)` = 2^42, i.e. **< 16384 voxels per axis** (`NTree.hpp:495`, only an `assert`). Resolution is `m_resolution` in mm.

## Node layout (inferred)
- `LeafNode`: 1 byte status. `InnerNode`: status (1) + flags (1) + 48-bit child pointer in three `uint16` = 8 bytes, pointing to a **contiguous block of 8 children** (all 8 exist, unknown ones as `ns_UNKNOWN` fillers). Prob variants add one `int8` occupancy per node (leaf 2 B; inner 9-10 B depending on padding, not measured). Root = one `InnerNode` (`m_root`).
- Status bits (`Nodes.h`): FREE 1, UNKNOWN 2, OCCUPIED 4, PART 8 (children differ), LAST_LEVEL 16 (children are leaves), COLLISION 32, STATIC_MAP 64, DYNAMIC_MAP 128. Flags: `nf_UPDATE_SUBTREE` 64, `nf_NEEDS_UPDATE` 128.
- `getNewStatus(or, and, level)` (`Nodes.h:52`): a parent takes the children's status if all 8 are the same pure state, else `or | PART`; static/dynamic bits are OR-ed. **Uniform nodes keep their children in memory** but are treated as leaves (no PART).
- Prob: parent occupancy = **max** of the children; PART unless all children have the same occupancy and no PART (`kernel_common.h:827`).

## Memory management (inferred)
No device allocator in kernels. `build`: bottom-up per level — Morton codes (`kernel_toMortonCode`), `thrust::sort` (CUB path disabled by `SORT_WITH_CUB false`), per level count new parents (`kernel_countNodes`), inclusive scan, `cudaMalloc` one array per level, `kernel_clearNodes`, `kernel_setNodes` (links parent -> child block by pointer) (`NTree.hpp:392-640`). `insertVoxel` (`NTree.hpp:2047`): sorted ids; `kernel_insert_countNeededNodes` (traverses, records per-thread-block counts per level), scan, **one `cudaMalloc` for all new nodes** (leaf part + 128-byte aligned inner part), init, `kernel_insert_setNodes`; the chunk is pushed on `m_allocation_list`. Nothing is freed until `rebuild()`, which extracts all nodes through the load-balanced `Extract` and builds a new tree (`NTree.hpp:2913`); `needsRebuild()` = allocated bytes >= `m_max_memory_usage` (`:3022`).

## Operations (inferred; GvlNTree names in brackets)
| Operation | Algorithm | Kernels / helpers |
|---|---|---|
| `build(points)` [`insertPointCloud`, `insertCoordinateList`] | see above | `kernel_toMortonCode`, count/clear/set nodes, optional `free_bounding_box` |
| `insertVoxel(voxels, set_free, propagate_up)` | sorted ids + per-voxel occupancy, count / allocate / set; flags `NEEDS_UPDATE` | `kernel_insert_*`, `kernel_split_voxel_vector` |
| `propagate()` | restores the status invariant top-down and bottom-up with the load balancer (`Propagate`) | `kernel_common.h` `bottomUpUpdate(Prob)`, tasks tuned by `linearApprox` for a GTX Titan |
| `intersect_sparse(map/list)` [`collideWith(BitVoxelMap/ProbVoxelMap/BitVoxelList)`] | one thread per **occupied map/list voxel** looks the voxel up in the tree from the root (`kernel_intersect_VoxelMap/VoxelList`); counts collisions, optional collision flag, optional OR of type bit vectors, optional count of collisions with unknown cells | `kernel_Octree.h:541-660` |
| `intersect_morton(list)` [`collideWith(MortonVoxelList)`] | same with Morton ids (`BitVoxelList<..., OctreeVoxelID>`) | `kernel_intersect_MortonVoxelList` |
| `intersect_load_balance(other tree)` [`collideWith(GvlNTree*)`] | simultaneous depth-first traversal of both trees with the load balancer; `min_level` = resolution; collision = both nodes OCCUPIED (`DefaultCollider`); marks `ns_COLLISION` | `load_balancer/Intersect`, `kernel_config/Intersect.cuh` |
| `intersect_load_balance(ProbVoxelMap)` | traversal of the tree, lookup in the map | `IntersectVMap` |
| `extractCubes(...)` | all nodes selected by a status mask become `Cube{id, side, type}`; used by the visualiser and by `rebuild` | load-balanced `Extract` |
| `find`, `checkTree`, `print*`, `clearCollisionFlags`, `serialize/deserialize` (text or binary node dump), `clear`, `rebuild` | | |
- Observed quirks in the façade (`GvlNTree.hpp:160-300`): the map/list overloads of `collideWithResolution` **ignore `resolution_level` (only range-checked) and `coll_threshold`**; tree-vs-tree honours `resolution_level` and rejects offsets with a log message; `merge`, `insertMetaPointCloudWithSelfCollisionCheck`, `collideWithBitcheck` are "not (yet) supported" log-only stubs; `collideWithTypes` exists for BitVectorVoxelMap / BitVectorVoxelList only.
- Sensor path: `Sensor`, `PointCloud*`, `kernel_PointCloud` (Kinect depth image preprocessing, transform, voxelize, free-space voxels); free space is computed by `NTree::computeFreeSpaceViaRayCast` (`NTree.hpp:1935`, called from `insertVoxel(free_space, object, sensor_origin, ...)` at `:2318`; two older variants are commented out at `:1177`, `:1812`); `VisNTree.cu` is commented out of the CMake target.

## Load balancer (inferred)
Host loop `AbstractLoadBalancer::run`: `doWork` (kernel `kernelLBWorkConcept`: one work stack per thread block, `NUM_TASKS` = 2688 blocks of 128 threads; every node is handled by 8 threads (`work_index = tid / 8`, `work_lane = tid % 8`)) -> `doBalance` (`kernelCountElements`, `thrust::exclusive_scan`, `kernelMoveElements`: items redistributed round-robin by level, big items first) until no work remains. A block leaves the kernel when its stack is empty/full (and increments a global idle counter) or when the idle counter reaches 2/3 of the tasks (**read, never waited on: no inter-block spin-waiting**). Work items are structs of node pointers + level/flags (`NTreeData.h`).

## Constants (inferred)
`CommonValues.h`: `NUM_BLOCKS 2688`, `NUM_THREADS_PER_BLOCK 128`, `FREE_UPDATE_PROBABILITY -10`, `OCCUPIED_UPDATE_PROBABILITY 100` (the voxel map sensor model uses **72**), `THRESHOLD_OCCUPANCY 10` but `isVoxelOccupied(ProbabilisticVoxel)` uses **>= 50** (`kernel_common.h:161`), many `#define` switches (`DO_REBUILDS`, `KINECT_*`, `LOAD_BALANCING_PROPAGATE`).
