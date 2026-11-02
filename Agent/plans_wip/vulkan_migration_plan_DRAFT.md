# Vulkan Migration Plan — DRAFT

**Status:** draft (2026-10-22), nothing started. **P1 (Foundation) implemented 2026-10-22 except the abstract CUDA/Vulkan interface.** **Decided:** raw Vulkan (`vulkan.hpp` + VMA) for Q2; GoogleTest for Q13; CUDA stays for now (Q1); all vendors, feature-gated (Q4); CUDA golden data later (Q5); Vulkan SDK installed (Q9). **Still open:** Q3, Q6, Q7, Q10 in [open_questions.md](../project_state/open_questions.md). Companion: [package_layout_proposal.md](package_layout_proposal.md). Step IDs (`P0.1`…) are stable; do not renumber.
**Rules:** [migration_guardrails.md](../agent_policies/migration_guardrails.md), [definition_of_done.md](../agent_policies/definition_of_done.md). Background: [mistakes_to_avoid.md](../knowledge_base/vulkan_migration/mistakes_to_avoid.md), [design_options.md](../knowledge_base/vulkan_migration/design_options.md).

## Goal
A Vulkan compute backend for GPU-Voxels with the same public behaviour as the CUDA library, vendor-neutral, built from a clean checkout, validated against independent references.
**Non-goals (until later phases):** viewer, ROS/Kinect/OMPL helpers, performance tuning beyond "not pathological".

## Gate model
Every phase ends with a gate. A phase may start only when the previous gate is passed *and* its Exceptions table is empty (or the user accepted the entries). Levels (Compiles / Runs / Tested / Parity / Complete) are defined in the DoD.

## P0 — Decisions and prerequisites (no product code)
| ID | Step | Exit evidence |
|---|---|---|
| P0.1 | User answers Q1–Q9; record as ADR notes in this folder | `open_questions.md` "Answered" |
| P0.2 | Toolchain: **done 2026-10-22** — Vulkan SDK 1.4.363.0 (validation layer, glslang 16.6, glslc, dxc 1.9, slangc). Open: vcpkg source/bootstrap (Q10) | `current_state.md` |
| P0.3 | ~~Snapshot the archive~~ — dropped, user chose to leave it (Q8) | — |
| P0.4 | Branch for the work (Q11) | Branch name in `current_state.md` |
| P0.5 | Spike (time-boxed, throw-away): same kernel (`insert point cloud` + `collide`) on Kompute vs thin runtime | Analysis file with LOC, dispatch overhead, validation result; decision recorded |
**Gate P0:** decisions made, toolchain verified.

## P1 — Foundation (Runtime + build + test harness)
| ID | Step | Exit evidence |
|---|---|---|
| P1.1 | `Context`: instance/device selection, feature+limit query, validation layer, RAII | **Tested 2026-10-22** - `Context` (device + limits query, validation layer, RAII); test prints device and limits; zero validation errors; validation oracle test ([vulkan_runtime.md](../knowledge_base/architecture/vulkan_runtime.md)) |
| P1.2 | `DeviceBuffer<T>`: device-local, staging upload/download, size/alignment checks | **Tested** - `DeviceBuffer` (staging up/download, 160 MiB across chunks, invalid sizes rejected) |
| P1.3 | Pipeline/dispatch/barrier API; batching of several dispatches per submit | **Tested** - `ComputePipeline`/`CommandBatch`: dependent dispatches with barrier, 200 dispatches per submit, int64 shader. Caveat K20: a missing RAW barrier is not detected on this GPU/layer |
| P1.4 | CMake: optional backends **done** ([package_layout_proposal.md](package_layout_proposal.md) L1–L4); remaining: shader → SPIR-V custom commands (`cmake/Shaders.cmake`), no absolute paths | Fresh-clone configure+build log |
| P1.5 | Error handling policy replacing `HANDLE_CUDA_ERROR` for the Vulkan side | Done in code: `VulkanError` hierarchy plus `vk::SystemError`; no macro layer. Document when the abstraction interface is designed |
| P1.6 | Test harness **done** (GoogleTest, `gvl_add_test`); remaining: `tests/reference/` CPU references when the first kernel exists | CTest runs on the dev machine (verified for 7 smoke cases) |
**Gate P1:** *Tested* level for P1.1–P1.3 on this machine's device.

## P2 — Primitives (the Thrust replacement)
fill · copy · transform · reduce · inclusive/exclusive scan · radix sort (+by key) · compaction (`copy_if/remove_if`) · unique (+by key) · binary search · min/max.
Each: GLSL/shader, host wrapper, test vs `std::` on random + edge sizes (0, 1, workgroup±1, > one level of recursion). Reuse ideas from [reusable_assets.md](../knowledge_base/vulkan_migration/reusable_assets.md), re-derive code.
**Gate P2:** *Tested* for all primitives; no CPU fallbacks.
**Status 2026-10-23:** exclusive scan, stable radix sort of (key, value) pairs and iota are **Tested** ([vulkan_primitives.md](../knowledge_base/architecture/vulkan_primitives.md)); compaction, unique-by-key and binary search exist as list kernels ([vulkan_voxel_lists.md](../knowledge_base/architecture/vulkan_voxel_lists.md)). Not needed yet: reduce, min/max, generic copy_if. Limit: 2^24 elements.

## P3+ — Vertical slices (one at a time; each ends *Tested*, then *Parity*)
| Slice | Content | Reference/Parity source |
|---|---|---|
| S1 | `BitVoxelMap`: **done as far as defined 2026-10-22 (Tested, not Parity):** construct, `clearMap`, `insertPointCloud`, `insertCoordinateList`, `clearBitVoxelMeaning`, `collideWith`, `collideWithTypes` - [vulkan_bit_voxel_map.md](../knowledge_base/architecture/vulkan_bit_voxel_map.md). `merge`: CUDA only merges lists (-> S4). **Open:** offset collision (semantics undefined), disk I/O | CPU reference + golden data (Q5) |
| S2 | `MetaPointCloud` (GPU transform), `KinematicChain` (DH, both conventions), `insertMetaPointCloud(+SelfCollision)`: **done 2026-10-22 (Tested, not Parity)** - [vulkan_meta_cloud_and_robot.md](../knowledge_base/architecture/vulkan_meta_cloud_and_robot.md). **Open in S2:** URDF robot (urdfdom/KDL, Q6/vcpkg), `PointCloud` file loading | CPU FK reference; golden |
| S3 | `ProbVoxelMap` (+ sensor ray-casting): **done 2026-10-23 (Tested, not Parity)** - insert (points, coordinates, meta clouds), `insert_sensor_data` with CUDA ray walk and robot cut, `collide_with` (bit/prob maps, both directions), update modes Atomic / Accumulate / Sequential - [vulkan_prob_voxel_map.md](../knowledge_base/architecture/vulkan_prob_voxel_map.md). **Open in S3:** 4 voxels per word packing, self-collision insertion (CUDA: unsupported), per-context kernel cache | CPU reference (events + serial orders) |
| S4 | `BitVoxelList`, `CountingVoxelList`, list↔map collide, `make_unique`, merge: **done 2026-10-23 (Tested, not Parity)** - insert (points, coordinates, meta clouds), `make_unique`, `remove_underpopulated`, `collide_with` (list, counting+threshold, map), `subtract`, `BitVoxelMap::merge_voxel_list`, map->list extraction - [vulkan_voxel_lists.md](../knowledge_base/architecture/vulkan_voxel_lists.md). **Open in S4:** `ProbabilisticVoxelList`, list `collideWithTypes`/bit-mask collision, offsets, disk I/O, `getCenterOfMass`, `extractCubes`, `clone`/`equals`, lists > 2^24 entries | CPU reference (`std::map`) |
| S5 | `DistanceVoxelMap` (PBA): **done 2026-10-23 (Tested, not Parity)** - obstacle insertion, `merge_occupied`, `compute_distances` with two algorithms (exact separable, Parallel Banding with m1/m2/m3 bands), distance queries, `extract_distances`; K14 resolved (10-bit) - [vulkan_distance_map.md](../knowledge_base/architecture/vulkan_distance_map.md). **Open in S5 (user: later):** `DistanceMapConverter` (free space -> `BitVoxelList`, now possible on S4), `exactDistances3D`, `jumpFlood3D`, `differences3D`, `init_floodfill` | Brute-force exact-distance reference |
| S6 | Octree (`NTree/GvlNTree`): **design review done 2026-10-29** ([Analysis](../Analysis/2026-10-29_octree_design_review/summary.md): index-based pool + wavefront traversal, slices S6a-f); decisions taken (pointer-faithful first, read-only core first, all variants, I/O behind an interface); **S6a done 2026-10-29 (Tested, not Parity)**: buffer device addresses, Morton host + GLSL, CPU octree reference ([vulkan_octree_foundation.md](../knowledge_base/architecture/vulkan_octree_foundation.md)); **S6b done 2026-10-29 (Tested, not Parity)**: `Octree` (both kinds) with GPU build, find, extract, host `check_tree`, 64-bit radix sort ([vulkan_octree_build.md](../knowledge_base/architecture/vulkan_octree_build.md)); **S6c collisions done 2026-10-30 (Tested, not Parity)**: tree against bit/prob maps, bit lists, Morton ids and other trees, collision flags ([vulkan_octree_collision.md](../knowledge_base/architecture/vulkan_octree_collision.md)); the persistent load balancer is ported as a second implementation of tree-vs-tree and extract (Q23), the wavefront is faster on the Intel UHD; checkpoint done 2026-10-30 ([report](../Analysis/2026-10-30_octree_checkpoint/summary.md), Q24: continue faithful, K34 fixed), next: S6d dynamic updates | CPU reference of the node semantics (recursive construction from a voxel set), brute-force collision on voxel sets |
| S7 | `GpuVoxels` façade completeness: header-by-header checklist of every public method; URDF robot (ROS-free). **Checklist done 2026-11-02:** [Analysis/2026-11-02_s7_facade_checklist.md](../Analysis/2026-11-02_s7_facade_checklist.md) (nothing of the façade exists on Vulkan yet; slices S7-1 .. S7-5; **S7-1 done 2026-11-02 (Tested, not Parity)**: [vulkan_gpu_voxels_facade.md](../knowledge_base/architecture/vulkan_gpu_voxels_facade.md); **S7-2 done 2026-11-02 (Tested, not Parity)**: [vulkan_gpu_voxels_files.md](../knowledge_base/architecture/vulkan_gpu_voxels_files.md)) | Checklist; example programs as smoke tests |
For every slice: public header audit table (method → status), Exceptions table, and an updated `knowledge_base/` entry.

## P-final — Peripheral and cleanup (separate approvals)
Port ICMAKER CMake for examples/tests; visualizer track (D7); documentation/README; remove CUDA only if Q1 says so and Parity evidence exists; delete cleanup candidates.

## Exceptions table (stub register)
| symbol | file | reason | exit criterion | owner/date |
|---|---|---|---|---|
| _(empty)_ | | | | |

## Step log
_(append `YYYY-MM-DD step-id level evidence-link` lines as work proceeds; levels as in [definition_of_done.md](../agent_policies/definition_of_done.md); nothing here is *Parity*, no CUDA is available)_
- 2026-10-22 P1 Tested [vulkan_runtime.md](../knowledge_base/architecture/vulkan_runtime.md)
- 2026-10-22 S1 BitVoxelMap Tested [vulkan_bit_voxel_map.md](../knowledge_base/architecture/vulkan_bit_voxel_map.md)
- 2026-10-22 S2 MetaPointCloud, KinematicChain Tested [vulkan_meta_cloud_and_robot.md](../knowledge_base/architecture/vulkan_meta_cloud_and_robot.md)
- 2026-10-23 P2 primitives Tested [vulkan_primitives.md](../knowledge_base/architecture/vulkan_primitives.md)
- 2026-10-23 S4 voxel lists Tested [vulkan_voxel_lists.md](../knowledge_base/architecture/vulkan_voxel_lists.md)
- 2026-10-23 S3 ProbVoxelMap Tested [vulkan_prob_voxel_map.md](../knowledge_base/architecture/vulkan_prob_voxel_map.md)
- 2026-10-23 S5 DistanceVoxelMap Tested [vulkan_distance_map.md](../knowledge_base/architecture/vulkan_distance_map.md)
- 2026-10-29 S6 design review (decisions Q19-Q22) [Analysis](../Analysis/2026-10-29_octree_design_review/summary.md)
- 2026-10-29 S6a runtime BDA, Morton, reference Tested [vulkan_octree_foundation.md](../knowledge_base/architecture/vulkan_octree_foundation.md)
- 2026-10-29 S6b build / find / extract / check_tree Tested [vulkan_octree_build.md](../knowledge_base/architecture/vulkan_octree_build.md)
- 2026-10-30 vcpkg bump to d4c0a46cb8 (Q10) [current_state.md](../project_state/current_state.md)
- 2026-10-30 S6c collisions (maps, lists, ids, tree vs tree, flags) Tested [vulkan_octree_collision.md](../knowledge_base/architecture/vulkan_octree_collision.md)
- 2026-10-30 S6c load balancer as second implementation (Q23) Tested [vulkan_octree_collision.md](../knowledge_base/architecture/vulkan_octree_collision.md#load-balancer-second-implementation)
- 2026-10-30 Octree checkpoint (decision Q24: continue faithful) [report](../Analysis/2026-10-30_octree_checkpoint/summary.md)
- 2026-10-30 K34 size limits fixed and tested [known_issues.md](../knowledge_base/known_issues.md)
- 2026-10-30 verification tooling saved into the repository, machine change prepared [tools/README.md](../../tools/README.md), [machine_change_14900k.md](../project_state/machine_change_14900k.md)
