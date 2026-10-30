# Vulkan MetaPointCloud, meta cloud insertion, self collision, DH robot (slice S2, 2026-10-22)

Classes in `gpu_voxels_vulkan`: `MetaPointCloud`, `KinematicChain` (+ `dh_matrix`, `DHParameters`, `LinkSpec`), and the methods `BitVoxelMap::insert_meta_point_cloud` / `insert_meta_point_cloud_with_self_collision_check`. Plain value types in `vulkan_types.h` (`Point3f`, `Matrix4f` row-major, `BitVector256`, ...); no Eigen dependency yet.
Status: **Tested** against independent CPU references; **not Parity** (no CUDA run, Q5). Not part of S2: URDF robot (needs urdfdom/KDL), `GpuVoxelsMap` integration, `ProbVoxelMap`/`DistanceVoxelMap` targets of `insertMetaPointCloud`, DH dirty-tracking, `PointCloud` file loading, the abstract CUDA/Vulkan interface.

## What was ported
| CUDA | Vulkan | Notes |
|---|---|---|
| `MetaPointCloud` (accumulated device cloud, offsets, `transform*`) | `MetaPointCloud` | one buffer of packed floats + `offsets()`; no host mirror; transformation = ONE dispatch for any set of sub clouds (table of 3x4 matrices, binary search of the sub cloud per point); in-place variant has its own shader (no buffer aliasing) |
| `KernelTransform` (`R*p + t`, last row ignored) | `meta_transform*.comp` | p' = M[0..2][0..2]*p + M[0..2][3] |
| `kernelInsertMetaPointCloud` (+ meanings per sub cloud) | `bit_voxel_insert_meta.comp` | same bounds rule as S1 (outside counted, returned) |
| `kernelInsertMetaPointCloudSelfCollCheck` | `bit_voxel_insert_meta_selfcoll.comp`, one dispatch per sub cloud + barrier | default meanings `4 + k`, default masks "all bits except COLLISION and own"; reads the voxel with an atomic load; colliding bits aggregated with `atomicOr` |
| `KinematicLink` / `DHParameters::convertDHtoM` (CLASSIC, CRAIGS) | `dh_matrix()` | computed on the CPU in float, like CUDA |
| `KinematicChain` | `KinematicChain` | T_0 = base, T_{i+1} = T_i * DH_i; link i's cloud is moved with **T_i** (link 0 sits at the base) - as in CUDA |

## CUDA behaviour found while porting (and what the Vulkan code does)
1. `MetaPointCloud::transform(M, out)` calls `thrust::transform(*m_dev_ptrs_to_addrs.begin(), *m_dev_ptrs_to_addrs.end(), ...)` - it dereferences `end()` of a vector; the intended meaning (transform every point) is implemented here (K22).
2. Size checks of CUDA transformations only log an error and return silently; here a mismatch of sub cloud sizes **throws** `VulkanError`. CUDA compares only the total size / count for the whole-cloud transform, this port requires identical sub cloud sizes (needed for the offsets to match).
3. `kernelInsertMetaPointCloudSelfCollCheck` updates `*colliding_subclouds |= masked` without atomics (lost updates possible) and the result depends on the order of points inside a sub cloud if a mask contains the sub cloud's own bit or `eBVM_COLLISION`. The Vulkan version aggregates with `atomicOr`; for the default masks the result is order-independent and equals the sequential reference. With custom masks that include the own/collision bit the result is order dependent (as in CUDA) and not covered by tests.
4. The CUDA chain tracks "dirty" transforms and clouds incrementally; this port recomputes all transforms when something changed (same results, simpler). `getLowerJointLimits/UpperJointLimits` are stubs in CUDA and not ported. `exit(-1)` on mismatching link/parameter counts is replaced by exceptions.
5. Float precision: the transformation runs in float on the GPU (possible FMA contraction); results differ from a double reference by ~1e-6 for unit-sized coordinates (tests use 2e-5 / 2e-4). Voxel assignment of transformed points is only guaranteed away from voxel boundaries.

## Verification (2026-10-22, Intel UHD, validation layer on, zero validation errors)
- References: `tests/reference/kinematics_reference.h` (double precision; **DH matrices composed from elementary rotations/translations, not copied from the CUDA formulas**: classic `Rz(theta) Tz(d) Tx(a) Rx(alpha)`, Craig `Rx(alpha) Tx(a) Rz(theta) Tz(d)`) and the extended `bit_voxel_map_reference.h` (sequential self collision).
- 19 tests: `test_meta_point_cloud.cpp` (geometry incl. empty sub clouds, transform of all / selected / 700 sub clouds with many empty ones, in-place, invalid arguments, meta insertion with per-cloud meanings, self collision with default and custom masks, no-overlap case, FREE-bit-only voxels, argument errors) and `test_kinematic_chain.cpp` (DH matrices of both conventions with revolute and prismatic joints, chain transforms and clouds for random 6-link robots over several configurations, base transformation, cache behaviour, errors, and a robot-to-voxel-map run with self collision on grid-friendly configurations compared to the reference using double precision FK).
- **Mutation testing, 16 mutations, all killed** (script in the session scratchpad): transposed/dropped matrix terms, wrong sub cloud search, disabled clouds transformed in place, wrong meaning lookup, self-collision variants (own meaning missing, mask not applied, own meaning not reported, collision marker missing, FREE bit counted), DH formula errors for both conventions, prismatic joint on theta, link transformed with the wrong frame, default masks/meanings wrong, translation ignored. The FREE-bit mutation initially **survived**; a test with FREE-only voxels was added and killed it.
- Whole suite: `test_vulkan` 49 cases pass; `ctest` in `build/verify-vk` 57/58 (K18 pre-existing).

## Limits / next
- Everything is bound as whole buffers (point clouds and maps up to `maxStorageBufferRange`); 700 sub clouds / 12 k points per cloud were tested, not millions.
- The pipelines are created per `MetaPointCloud`/`BitVoxelMap` family (shared via `shared_ptr` between a cloud and `make_like()`); a per-context kernel cache is still open.
- Next in the plan: finish S1 items that need decisions (offset collision, disk I/O), then S3 `ProbVoxelMap` (needs the sensor model), S4 lists, S5 distance maps, S6 octree; URDF robot depends on Q6/vcpkg (urdfdom, orocos-kdl).
