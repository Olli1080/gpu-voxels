# Data Model and Public API

## Core vocabulary (`helpers/common_defines.h`)
- `BIT_VECTOR_LENGTH = 256` ⇒ a `BitVectorVoxel` (`BitVoxel<256>`) is **32 bytes**; a 256³ BitVoxelMap = 16.8 M voxels ≈ **512 MB** (matches the earlier measurement).
- `BitVoxelMeaning : uint8_t` — `eBVM_FREE=0, OCCUPIED=1, COLLISION=2, UNKNOWN=3, SWEPT_VOLUME_START=4 … END=254, UNDEFINED=255`; values 4/129/254 double as probability anchors (`MAX_FREE_PROB`, `UNCERTAIN_OCC_PROB`, `MAX_OCC_PROB`) when inserting into probabilistic maps.
- `MapType`: `MT_BITVECTOR_VOXELMAP | _VOXELLIST | _OCTREE | _MORTON_VOXELLIST`, `MT_PROBAB_VOXELMAP | _VOXELLIST | _OCTREE | _MORTON_VOXELLIST`, `MT_COUNTING_VOXELLIST`, `MT_DISTANCE_VOXELMAP`.
- `MapVoxelID = uint32_t` (dense maps; GPU RAM bounds size), `OctreeVoxelID = uint64_t` (Morton code).
- Vector types are **Eigen** (`Vector3ui/3i/3f/4i/4f` = `Eigen::Vector<T,N>`), declared in `cuda_vectors.hpp`; matrices in `cuda_matrices.hpp`. (Upstream used CUDA `float3/uint3`; this fork already modernised it.)

## Voxel types (`voxel/`)
| Type | Payload | Notes |
|---|---|---|
| `BitVoxel<N>` | `BitVector<N>` (N bits, one per meaning) | collision between maps = AND of bit vectors via `DefaultCollider`/`SVCollider` |
| `ProbabilisticVoxel` | `Probability` (int8 occupancy, log-odds style) | `updateOccupancy`, `reduce_op`; float↔int8 conversion helpers |
| `DistanceVoxel` | packed obstacle position (`pba_voxel_t`=uint32, comment says 3×9 bit + 3 "uninitialised" marker bits; the Kompute attempt documented 10-bit packing — **unresolved discrepancy**, check before porting) + distance (`pba_dist_t`=int32) | produced by the Parallel Banding Algorithm (PBA); `DISTANCE_UNINITIALISED = 0` |
| `CountingVoxel` | `int8_t m_count` | noise filter via density threshold |
Voxel classes carry host+device methods and Thrust functors (`reduce_op`, `extract_byte_distance`, …) — porting means flattening these into shader code (see [cuda_coupling_inventory.md](cuda_coupling_inventory.md)).

## Containers
| Class | Storage | Key operations |
|---|---|---|
| `TemplateVoxelMap<Voxel>` → `BitVoxelMap<N>`, `ProbVoxelMap`, `DistanceVoxelMap` | `thrust::device_vector<Voxel>` (dense 3D, index = x + y·dx + z·dx·dy) | `insertPointCloud`, `insertCoordinateList`, `insertMetaPointCloud(+SelfCollisionCheck)`, `collideWith(…)` (templated on Voxel & Collider, optional offset), `merge`, erode/dilate/close, `clearMap`, `clearBitVoxelMeaning`, shift swept-volume IDs, `writeToDisk/readFromDisk`, ray-casting sensor insertion |
| `TemplateVoxelList<Voxel,ID>` → `BitVoxelList`, `CountingVoxelList` | sorted coordinate array + voxel array (device vectors) | insert, `make_unique` (sort + reduce-by-key), `collideWith` map/list/octree, remove-underpopulated, merge |
| `NTree`/`GvlNTree` (`octree/`) | node pools with 48-bit pointers, Morton codes, GPU load balancer | build from point cloud/sensor, intersect with octree/voxelmap/voxellist, propagate, extract |
| `PrimitiveArray` | position/size array | visualization of spheres/cubes |
| `MetaPointCloud` / `PointCloud` | concatenated sub-clouds with offsets | robot links; transform on GPU; sub-cloud → voxel meaning |
| `KinematicChain` (DH) | joint/link params | forward kinematics → MetaPointCloud transform |

`GpuVoxelsMap` (abstract, `core/`) is the common interface: `insertPointCloud`, `insertCoordinateList`, `insertMetaPointCloud*`, `merge`, `clearMap`, `needsRebuild/rebuild`, `getMemoryUsage`, `getDimensions`, disk I/O, plus `thrust::device_vector<Vector3ui>` overloads — **Thrust types appear in the public API**.

## `GpuVoxels` façade (`gpu_voxels/GpuVoxels.h`, 469 lines)
`initialize(dim_x,dim_y,dim_z,voxel_side_length)`; `addMap(MapType,name)/delMap/getMap/clearMap`; `visualizeMap/visualizePrimitivesArray`; robots: `addRobot` (DH / URDF), `setRobotConfiguration`, `updateRobotPart`, `insertRobotIntoMap(+SelfCollAware)`; inserts: `insertPointCloudFromFile/IntoMap`, `insertMetaPointCloudIntoMap`, `insertBoxIntoMap`; primitives: `addPrimitives/modifyPrimitives`; `getDimensions`, `getVoxelSideLength`. Constructor calls `cuTestAndInitDevice()` and `exit(123)` when no CUDA device is found.
Keeping this façade source-compatible is the lowest-risk interface for a backend swap (guardrail 4).
