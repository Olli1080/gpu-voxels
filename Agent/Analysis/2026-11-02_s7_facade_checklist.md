# S7: `GpuVoxels` façade, checklist of every public method (2026-11-02)

**Progress:** S7-1, S7-2 and S7-3 are done (2026-11-02, [vulkan_gpu_voxels_facade.md](../knowledge_base/architecture/vulkan_gpu_voxels_facade.md), [vulkan_gpu_voxels_files.md](../knowledge_base/architecture/vulkan_gpu_voxels_files.md), [vulkan_gpu_voxels_robots.md](../knowledge_base/architecture/vulkan_gpu_voxels_robots.md)); the rows marked S7-1 .. S7-3 below are implemented, S7-4 and S7-5 are open (joint limits do not exist for DH robots, also not in CUDA). The text below is the state before S7-1.

**Question:** what does the Vulkan backend need so that the public API of `gpu_voxels::GpuVoxels` ([GpuVoxels.h](../../packages/gpu_voxels/gpu_voxels/gpu_voxels/include/gpu_voxels/GpuVoxels.h), implementation [GpuVoxels.cu](../../packages/gpu_voxels/gpu_voxels/gpu_voxels/src/GpuVoxels.cu)) and the map interface behind it (`GpuVoxelsMap`, [GpuVoxelsMap.h](../../packages/gpu_voxels/gpu_voxels/core/include/gpu_voxels/core/GpuVoxelsMap.h)) can be offered on Vulkan?

**Method:** both headers and `GpuVoxels.cu` were read in full, plus the `merge`, `clearBitVoxelMeaning`, `needsRebuild`, `getDimensions`, `writeToDisk` and self-collision overrides of the voxel map, voxel list and octree classes (the legacy behaviour column below comes from that reading, nothing ran under CUDA). The Vulkan side was read from its public headers (`vulkan/include/gpu_voxels/vulkan/*.h`). The examples in `packages/examples` were counted to see which calls are really used.

**Finding in one sentence:** nothing of `GpuVoxels` exists on Vulkan yet (no class, no map interface, no singleton), but for most of its 37 public members (counting every overload) the building blocks exist and are tested; what is missing is the layer on top (names, runtime map type, `as<T>()`), the point cloud file readers, `insertBoxIntoMap`, list / prob / distance `merge`, the disk formats of maps and lists, and everything that touches a viewer or ROS.

Legend: **R** = the Vulkan building block exists and is tested, the façade only wraps it. **W** = work beyond a wrapper on the Vulkan side. **N** = not offered (legacy stub that only logs, or no consumer). **O** = out of scope until the user asks (Q6: viewer, ROS, URDF, Kinect).

## 1. `GpuVoxels` members

| # | Member | Legacy behaviour (read) | Vulkan state | Status | Slice |
|---|---|---|---|---|---|
| 1 | `~GpuVoxels` | frees the managed maps, robots, arrays | the owner of the `Context` must outlive all maps | R | S7-1 |
| 2 | `getInstance()` | singleton through a `weak_ptr` (lives while someone holds it) | no equivalent; the instance must create and own the `Context` | W (small) | S7-1 |
| 3 | `initialize(dim, side)` | stores dimensions and side length; `addMap` creates maps with them | the maps take `Dim3ui` and a side length; `Context` creation is lazy and may throw | W (small) | S7-1 |
| 4 | `addMap(MapType, name)` | duplicate name: error log + empty pointer; Morton and prob. list types: throws "not implemented" | one class per type, see section 2; no common base class | W | S7-1 |
| 5 | `delMap` | erases, `false` if unknown | trivial | R | S7-1 |
| 6 | `clearMap(name)` | `map->clearMap()` | every class has `clear_map` / `clear` | R | S7-1 |
| 7 | `clearMap(name, meaning)` | `clearBitVoxelMeaning`, see section 3 | bit map, prob. map: yes; lists: no (legacy: not supported); octree: only occupied | R / N | S7-1 |
| 8 | `getMap(name)` | shared pointer, empty + error log if unknown | needs the common map handle of section 4 | W | S7-1 |
| 9 | `visualizeMap`, `visualizePrimitivesArray`, `getVisualization` | without `GPUVOXEL_VISUALIZE`: `false` / `nullptr` | no viewer | O | later |
| 10 | `addRobot` (DH, files) | builds a `KinematicChain`; duplicate name: error log + `false` | `KinematicChain(context, convention, links)` exists; the link clouds must come from files | W (needs 15) | S7-3 |
| 11 | `addRobot` (DH, `MetaPointCloud`) | same, with ready clouds | `KinematicChain` takes `LinkSpec` with host points; a constructor from a `MetaPointCloud` is missing (download + `LinkSpec` is possible) | W (small) | S7-3 |
| 12 | `addRobot` (URDF) | only with `_BUILD_GVL_WITH_URDF_SUPPORT_` | no URDF parser | O | later |
| 13 | `getRobot(name)` | `RobotInterfaceSharedPtr` | the Vulkan robot is `KinematicChain` (no interface class; one is needed when the URDF robot arrives). `RobotInterface` members: `setConfiguration`, `getConfiguration`, `getTransformedClouds`, `updatePointcloud`, `setBaseTransformation`, `getBaseTransformation`, `getTransform` exist; `getJointNames` is derivable from `configuration()`; `getLowerJointLimits` / `getUpperJointLimits` do not exist (the DH structure has no limits) | R / W (limits) | S7-3 |
| 14 | `setRobotConfiguration`, `setRobotBaseTransformation`, `getRobotTransformation`, `getRobotConfiguration`, `updateRobotPart` | thin wrappers, `false` if the robot is unknown | exist on `KinematicChain` | R | S7-3 |
| 15 | `insertPointCloudFromFile` | `.xyz`, `.pcd`, `.binvox` reader, `shift_to_zero`, `offset`, `scaling`, then `insertPointCloud` | no reader. The legacy readers (`XyzFileReader`, `PcdFileReader`, `BinvoxFileReader`) use `cuda_datatypes.hpp` / `thrust` types in their interface | W | S7-2 |
| 16 | `insertPointCloudIntoMap` (`PointCloud`, `vector`) | `map->insertPointCloud` | all classes have `insert_point_cloud(span<Point3f>)`; the legacy `PointCloud` class is thrust based (Q7: not needed, `span` / `vector`) | R | S7-1 |
| 17 | `insertMetaPointCloudIntoMap` (one meaning, per cloud) | `map->insertMetaPointCloud` | see section 3 | R / W | S7-1 |
| 18 | `insertRobotIntoMap` | `insertMetaPointCloud(*robot->getTransformedClouds(), meaning)` | `KinematicChain::transformed_clouds()` + `insert_meta_point_cloud` | R | S7-3 |
| 19 | `insertRobotIntoMapSelfCollAware` | `map->insertMetaPointCloudWithSelfCollisionCheck` | bit map only (`insert_meta_point_cloud_with_self_collision_check`); the legacy prob. map, distance map, lists and octree log "not supported" and **return true** | R (bit map) / N | S7-3 |
| 20 | `insertBoxIntoMap` | `geometry_generation::createBoxOfPoints(min, max, side / points_per_voxel, side)` gives coordinates, then `insertCoordinateList` | `insert_coordinate_list` exists, the box generator is CPU code inside `GeometryGeneration.cu`; a host implementation must be written and tested against the formula | W (small) | S7-2 |
| 21 | `addPrimitives`, `delPrimitives`, `modifyPrimitives` (4 overloads) | a host container of points / sizes for the viewer (`PrimitiveArray::setPoints`) | none; it has no GPU part, a consumer exists only with a viewer | W (small) / O | S7-5 |
| 22 | `getDimensions` (2 overloads), `getVoxelSideLength` | return what `initialize` stored | trivial | R | S7-1 |
| 23 | Types in the header: `CudaIpcMemHandleSharedPtr`, `ManagedMap(s)`, `ManagedPrimitiveArray(s)`, `ManagedRobots` | CUDA IPC handle (visualizer sharing), bookkeeping maps | IPC handle: not applicable (N); the bookkeeping is internal | N / W | S7-1 |

## 2. `MapType` to Vulkan class

| `MapType` | Legacy class | Vulkan class | Status |
|---|---|---|---|
| `MT_BITVECTOR_VOXELMAP` | `voxelmap::BitVectorVoxelMap` | `vulkan::BitVoxelMap` | R |
| `MT_PROBAB_VOXELMAP` | `voxelmap::ProbVoxelMap` | `vulkan::ProbVoxelMap` | R |
| `MT_DISTANCE_VOXELMAP` | `voxelmap::DistanceVoxelMap` | `vulkan::DistanceVoxelMap` | R |
| `MT_BITVECTOR_VOXELLIST` | `voxellist::BitVectorVoxelList` | `vulkan::BitVoxelList` | R |
| `MT_COUNTING_VOXELLIST` | `voxellist::CountingVoxelList` | `vulkan::CountingVoxelList` | R |
| `MT_BITVECTOR_OCTREE` | `NTree::GvlNTreeDet` | `vulkan::Octree(OctreeKind::Deterministic)` | R |
| `MT_PROBAB_OCTREE` | `NTree::GvlNTreeProb` | `vulkan::Octree(OctreeKind::Probabilistic)` | R |
| `MT_PROBAB_VOXELLIST` | throws "not implemented" | `ProbabilisticVoxelList` not built (S4 open) | N (same as legacy) |
| `MT_BITVECTOR_MORTON_VOXELLIST`, `MT_PROBAB_MORTON_VOXELLIST` | throws "not implemented" | not built (64-bit list ids open); `Octree::collide_with_ids` takes raw ids | N (same as legacy) |

## 3. `GpuVoxelsMap` virtual members by map type

`B` bit map, `P` prob. map, `D` distance map, `L` bit list, `C` counting list, `T` octree (both kinds). Cells: R / W / N as in the legend, `L*` = legacy logs an error and does nothing.

| Member | B | P | D | L | C | T | Remarks |
|---|---|---|---|---|---|---|---|
| `getMapType` | R | R | R | R | R | R | a tag in the wrapper |
| `insertPointCloud` (vector, `PointCloud`) | R | R | R | R | R | R | meaning handling per class as before (octree: occupied / free only) |
| `insertCoordinateList` (host) | R | R | R | R | R | R | |
| `insertCoordinateList` (`thrust::device_vector`) | N | N | N | N | N | N | Q7: no thrust in the new API; a `DeviceBuffer` overload only if a consumer appears |
| `insertMetaPointCloud` (one meaning) | R | R | R | R | R | R | |
| `insertMetaPointCloud` (meaning per cloud) | R | R | W (small) | R | R | R | the distance map has only the one-meaning form |
| `insertMetaPointCloudWithSelfCollisionCheck` | R | N | N | N | N | N | legacy: others log and **return true** |
| `insertPointCloudFromFile` | W | W | W | W | W | W | needs the readers (row 15) |
| `merge` (voxel offset, metric offset) | R (list, offset 0) | W (list, offset 0) | W (list, offset 0) | W | W | N | legacy: voxel maps merge lists (`insertCoordinateList` of the list coordinates as occupied), offset != 0 logs an error and returns `false`; lists merge maps / lists (specialisations for bit and counting lists); octree: stub. Vulkan has `BitVoxelMap::merge_voxel_list`, `BitVoxelList::insert_occupied_voxels`, `append_occupied_voxels`, `DistanceVoxelMap::merge_occupied(ProbVoxelMap)` |
| `writeToDisk`, `readFromDisk` | W | W | W | W | W | R | octree: `OctreeStorage` (stream / file); the others need a native format and a decision whether the CUDA formats are read at all |
| `clearMap` | R | R | R | R | R | R | |
| `clearBitVoxelMeaning` | R | R (occupied only: clears) | N (legacy: no-op) | N (L*) | N (L*) | R (occupied only: clears) | |
| `needsRebuild`, `rebuild`, `rebuildIfNeeded` | N (false) | N (false) | N (false) | N (false) | N (false) | R | legacy logs "not supported" and returns `false`; the wrapper returns `false` without logging |
| `getMemoryUsage` | R | R | R | R | R | R | octree: `memory_usage()` |
| `getDimensions` | R | R | R | R (legacy: `(list size, 1, 1)`) | R (same) | R (2^14 per axis) | |
| `getMetricDimensions` | R | R | R | R | R | R | legacy octree has a typo (`dim.z()` for y) |
| `as<T>()`, `is<T>()` | W | W | W | W | W | W | design question D1 |
| public `m_mutex` | W | W | W | W | W | W | design question D3 |

**Collisions are not in the interface** (`collideWith` is commented out in `GpuVoxelsMap`); the examples call them on the concrete class through `as<T>()` (9 `collideWith`, 4 `collideWithTypes`, 8 `collideWithTypeMask`). On Vulkan the collision pairs exist as methods of the classes (S1 to S6c); **`BitVoxelList::collideWithTypeMask` is missing** (S4 open item) and is the only collision call of the examples without a counterpart.

## 4. Design questions with the default I will use

| # | Question | Default (reason) |
|---|---|---|
| D1 | The Vulkan classes have no common base class. How does `getMap` return "a map"? | A thin `vulkan::Map` handle that owns one of the seven classes (`std::variant` of unique pointers), has the members of section 3 as ordinary methods that dispatch on the type, and `as<T>()` returning `T*` or `nullptr` (checked, not a `static_cast` like the legacy one). Reason: the classes stay independent and tested as they are, and a wrong `as<T>()` becomes `nullptr` instead of undefined behaviour |
| D2 | What happens for an operation the type does not support? | An unknown name keeps the legacy way (log, return `false` / empty handle). An operation the map type does not offer throws `VulkanError` with the type and operation in the message. Reason: legacy returns `true` ("collision") for unsupported self-collision insertion, which hides the problem; this follows the S6e rule that an API that always fails is not an API. Not decided by the user: tell me if you want the legacy return values instead |
| D3 | Thread safety (`m_mutex`) | No locking in the first version; documented as "one thread per façade". `Context` is not shared between threads either. Reason: no consumer, and a mutex per map is easy to add later |
| D4 | Types in the signatures | `std::span` / `std::vector` of the Vulkan value types (`Point3f`, `Coord3ui`, `Matrix4f`), as in S1 to S6 (decision Q7, "ja", reading: source compatibility is not required). No Eigen in the public header yet |
| D5 | Where does the code live? | `vulkan/include/gpu_voxels/vulkan/GpuVoxels.h` + `src/GpuVoxels.cpp`, namespace `gpu_voxels::vulkan`, next to the classes it wraps. The legacy header stays untouched |
| D6 | Examples as smoke tests (plan) | The examples need the viewer. Instead, small headless programs (or test cases) repeat their GpuVoxels call sequences without `visualizeMap`; the call count of the examples tells which calls matter most (section 5) |

## 5. What the examples call (`packages/examples`, 25 programs)

| Calls | Member |
|---|---|
| 136 | `getMap` |
| 73 | `addMap` |
| 58 | `visualizeMap` (O: returns `false` here) |
| 26 | `initialize` |
| 20 | `insertBoxIntoMap` |
| 14 | `insertPointCloudFromFile` |
| 13 / 13 / 13 | `setRobotConfiguration`, `insertRobotIntoMap`, `clearMap` |
| 8 / 8 / 8 | `addPrimitives`, `modifyPrimitives`, `visualizePrimitivesArray` |
| 7, 6, 4 | `addRobot`, `delMap`, `insertPointCloudIntoMap` |
| 2, 2 | `insertRobotIntoMapSelfCollAware`, `getRobot` |
| 1 each | `updateRobotPart`, `setRobotBaseTransformation`, `insertMetaPointCloudIntoMap`, `getVoxelSideLength`, `getVisualization` |

`getRobotTransformation`, `getRobotConfiguration`, `getDimensions`, `delPrimitives` and `clearMap(name, meaning)` are not used by any example (they still count for completeness).

## 6. Slice plan

| Slice | Content | Tests (independent of the implementation) |
|---|---|---|
| **S7-1** | `vulkan::GpuVoxels`: `getInstance`, `initialize`, `addMap` for the seven types (+ the legacy "not implemented" types as errors), `delMap`, `getMap`, `clearMap` (both), the map handle with `as<T>()` / `is<T>()`, `getMapType`, dimensions, side length, `getMemoryUsage`, `needsRebuild` / `rebuild`, point cloud / coordinate / meta point cloud inserts, `clearBitVoxelMeaning` rules, `visualize*` returning `false` | for every map type the same insert sequence through the façade and directly on the class gives identical contents; duplicate names, unknown names, wrong `as<T>()`, `getInstance` lifetime, unsupported operations throw with the documented message; mutation runner for the dispatch |
| **S7-2** | host point cloud readers (`.xyz`, `.pcd`, `.binvox`) with `shift_to_zero`, `offset`, `scaling`; `insertPointCloudFromFile`; `insertBoxIntoMap` | readers against hand-written files (incl. malformed ones); the box against the CUDA formula in double precision; the example sequences |
| **S7-3** | robots in the façade: `addRobot` (DH, clouds from files and from a `MetaPointCloud`), `getRobot`, configuration / base / transform / `updateRobotPart` wrappers, `insertRobotIntoMap`, `insertRobotIntoMapSelfCollAware` (bit map), joint limits if the DH input has them | FK reference of S2; self collision against the S1 reference |
| **S7-4** | `merge` for lists and maps (offset 0, new meaning), disk I/O of maps and lists (native format) | merge against `std::map` references; write / read round trips, corrupted files |
| **S7-5** | `addPrimitives` / `modifyPrimitives` / `delPrimitives` as a host container; `BitVoxelList::collideWithTypeMask`; headless versions of the examples as smoke tests | the container against its inputs; the mask collision against a brute force |
| later (Q6) | URDF robot without ROS, viewer, Kinect | n/a |

## 7. Not verified here
- CUDA parity of anything listed (no CUDA on this machine); the legacy behaviour column is a reading of the sources.
- The legacy `initialize` body and the `PrimitiveArray` implementation were not read in detail (only the header and `addPrimitives` / `modifyPrimitives` in `GpuVoxels.cu`).
- Whether the maps behind `MT_BITVECTOR_VOXELLIST` merge maps with an offset: the legacy list `merge` was seen only to be specialised for bit and counting lists (`TemplateVoxelList.cu` lines 8 and 86), its offset handling was not read.
