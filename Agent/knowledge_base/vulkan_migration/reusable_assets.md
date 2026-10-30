# Reusable Assets from the Failed Attempts

**Warning — the only copy of some of this is uncommitted** in `C:\Code\gpu-voxels` (branch `kompute`, 303 dirty paths): 14 of the 49 untracked `.comp` shaders exist in no commit (listed below). Do not run `git clean/reset/checkout` there. The user decided (Q8, 2026-10-22) to simply leave the archive as it is; keep it read-only.

Treat everything as **reference to re-derive from, not code to paste**: none of it has Parity evidence (see [definition_of_done.md](../../agent_policies/definition_of_done.md)). Port each piece through the normal vertical-slice process.

## Worth reading / adapting
| Asset | Location | Why useful | Caveat |
|---|---|---|---|
| Core voxelmap shaders (clear, insert point cloud / coordinate tuples, collide + debug variant, erode, sensor insert, shift swept-volume IDs, clear meaning, extract distances) + `VoxelMapOperations.glsl`, `DistanceVoxel.glsl` | `kompute_backend:packages/gpu_voxels/gpu_voxels/voxelmap/src/kernels/*.comp|glsl` (committed: 15 files). Uncommitted extras: `kernelClearBitVoxelMeaning`, `kernelDebugMapValues`, `kernelExtractDistances`, `kernelInsertGlobalPointCloudBitVoxel`, `kernelShiftLeftSweptVolumeIDs` | Direct GLSL translations of the CUDA kernels in `VoxelMapOperations.hpp` | Per-instantiation duplication; BitVoxel size hard-wired |
| PBA distance transform: phase 1 flood-Z, phase 2 (proximate back-pointers, forward pointers, merge bands), transpose XY, phase 3 distances; JFA + exact-distance variants | same dir (`kernelPBA*.comp`, `kernelJumpFlood3D`, `kernelExactDistances3D`) and `gpu-voxels/code/shaders/distance_voxel_map.comp` (408 lines, single-file variant) | Hardest non-octree kernel family already translated; exact-distance kernel is a ready brute-force **independent reference** for tests | 9- vs 10-bit packing question; unverified against CUDA |
| Voxel list shaders: insert point cloud, compact list, collide with map | `voxellist/src/kernels/` (committed 3 + `VoxelListOperations.glsl`) | Stream compaction pattern | — |
| GPU primitives: workgroup scan + add-offset (committed); **radix sort histogram/scan/scatter, pack/unpack (uncommitted)**; recursive dispatch for scan >1 workgroup; 4-bit digit LSD radix sort with key/value | `helpers/src/kernels/`; host orchestration in `KomputeBridge.cpp` (≈ lines 250–393) | Covers Thrust's scan/sort/sort_by_key; independently testable (`test_kompute_sort.cpp`) | Written against Kompute tensors; host side must be redone |
| Octree: flat index-based layout `NTree.glsl`, `Morton.glsl`; kernels count/clear/set nodes, propagate level, voxelize pipeline (points → voxels → Morton → nodes), intersect (octree/octrees/voxellist/voxelmap), depth-image preprocessing | `octree/src/kernels/` (committed 15 incl. glsl; uncommitted: `kernel_intersect_VoxelList`, `kernel_intersect_VoxelMap`, `kernel_intersect_octree_simple`) | Proof that the pointer-based tree can be index-based; simultaneous-descent octree/octree collision | Highest risk; verify layout vs. CUDA `Nodes.h`; load balancer not ported |
| Robot / URDF without ROS: `urdf_to_kdl.cpp` converts `urdf::Model` → `KDL::Tree` using vcpkg `urdfdom` + `orocos-kdl`; `transform_pointcloud.comp` (mat4 per sub-cloud) | `gpu-voxels/code/robot/**`, `code/shaders/transform_pointcloud.comp` | Removes the `kdl_parser`/catkin dependency; master's URDF target is `if(OFF)` | Only tested on simulated robot |
| Self-collision with link mask: mode 41 in `voxel_map.comp` (`insertMetaPointCloudWithSelfCollisionCheck`, mask as `uint data[8]`) | `code/shaders/voxel_map.comp` | Shows how to map `BitVector` collision masks | Unverified vs CUDA |
| Test ideas: elementary kernels vs host, dispatch tests, memory-management tests, benchmark skeleton, cross-backend `TestParity` structure | `gpu-voxels/code/tests/*.cpp`, `packages/gpu_voxels/gpu_voxels/test/test_kompute_*.cpp` | Test structure per kernel is sound | References were self-written; remove the `cuda_runtime_api.h` mock and absolute paths |
| Shader build helper | `gpu-voxels/cmake/ShaderCompile.cmake` (`compile_shaders(target list)`, defines `GVL_SHADER_DIR`) | Pattern for glslang → SPIR-V in CMake | Includes a hard-coded kernels include dir |
| Layout probe | `gpu-voxels/test_sizeof.cpp` | Idea: assert host/GPU layout equality | — |

## Not worth salvaging
`SyclBridge.h`, `oneDPLBridge.h` Thrust mocks, `KomputeBridge` pointer registry, `fix_leaks.py`, the `gvl::migration` adapter layer, `install_sycl_deps.py`, the `.old`/duplicate sources, `test_octree_build_out_*.txt`, `FIX_VISUALIZER_HANG.md`/SHM changes (restate the problem against the new design instead), all phase checkboxes.

## Plan documents worth skimming for requirements (not for status)
`gpu-voxels/plans/PHASE_6_FEATURE_PARITY.md` (feature inventory), `PHASE_7_ROBOT_MODEL_SUPPORT.md`, `REMAINING_TASKS.md` (the unfinished list), `FIX_DESTRUCTIVE_CHANGES.md` (what got broken).
