# Library Modules (master @ dd47249)

Root `CMakeLists.txt` (project `GPU_VOXEL_TL`, languages `CXX CUDA`) → `packages/CMakeLists.txt` → `icl_core` (only with `GVL_BACKEND_CUDA`) + `gpu_voxels`.
Library code is under `packages/gpu_voxels/gpu_voxels/<module>/{include/gpu_voxels/<module>,src}`.

## Built on master (inferred from `add_subdirectory` in `packages/gpu_voxels/CMakeLists.txt`)
| Target | Lang | Role | Links (PUBLIC) |
|---|---|---|---|
| `gpu_voxels_logging` | CXX | icl_core logging streams per module (CUDA modules only; built with `GVL_BACKEND_CUDA`; the Vulkan backend logs with `GVL_LOG_*` from `vulkan/Log.h`) | — |
| `gpu_voxels_helpers` | CXX CUDA | Math/vector types (Eigen-based), `BitVector`, `PointCloud`, `MetaPointCloud`, file readers (binvox/xyz/pcd), Kinect (OpenNI2), geometry generation, CUDA error handling | icl_core*, logging |
| `gpu_voxels_primitive_array` | CXX CUDA | `PrimitiveArray` (spheres/cubes for visualization) | logging, helpers |
| `gpu_voxels_core` | CXX | `GpuVoxelsMap` abstract base, `ManagedMap`, constants | helpers, primitive_array, logging |
| `gpu_voxels_voxel` | CXX CUDA | Voxel types + colliders: `BitVoxel`, `ProbabilisticVoxel`, `DistanceVoxel`, `CountingVoxel`, `DefaultCollider`, `SVCollider` | helpers, core |
| `gpu_voxels_voxelmap` | CXX CUDA | Dense 3D maps: `BitVoxelMap`, `ProbVoxelMap`, `DistanceVoxelMap` (PBA), `TemplateVoxelMap`, kernels | helpers, core, voxel, **voxellist** |
| `gpu_voxels_voxellist` | CXX CUDA | Sparse lists: `BitVoxelList`, `CountingVoxelList`, `TemplateVoxelList` | helpers, core, voxel, **voxelmap, octree** |
| `gpu_voxels_octree` | CXX CUDA | `NTree`/`GvlNTree` octree, sensor model, load balancer, Morton codes | voxel, dh_robot, core, **voxelmap, voxellist** ("REMOVE THIS LATER" in CMake) |
| `gpu_voxels_dh_robot` | CXX CUDA | `KinematicChain/Link` (DH-parameter robots) + `robot_interface.h` | helpers |
| `gpu_voxels_distance_map_converter` | CXX CUDA | `DistanceMapConverter` (helpers_highlevel) | voxelmap, voxellist, voxel, core, helpers |
| `gpu_voxels` | CXX | `GpuVoxels` façade (`GpuVoxels.cu`) | all of the above |

**Circular link dependency:** voxelmap ⇄ voxellist ⇄ octree. It only works because of static-library semantics/CUDA separable compilation; any new backend must break this cycle (shared interface headers) rather than copy it.

## Added 2026-10-22 (uncommitted)
`gpu_voxels_vulkan` (`gpu_voxels/vulkan`, CXX only, `GVL_BACKEND_VULKAN`): Vulkan compute runtime - see [vulkan_runtime.md](vulkan_runtime.md). Not linked by any CUDA module.

## NOT built on master
| Item | State | Evidence |
|---|---|---|
| `vis_interface` (`gpu_voxels_vis_interface`) | `add_subdirectory` commented out; commit `dedcdeb` removed link references. `GpuVoxels.cu` only includes the `Vis*` headers under `#ifdef GPUVOXEL_VISUALIZE`, which master never defines ⇒ visualization is compiled out | `packages/gpu_voxels/CMakeLists.txt`, `GpuVoxels.cu:28` |
| `gpu_visualization` (OpenGL viewer) | Not in `packages/CMakeLists.txt`; its CMake uses `ICMAKER_*` macros that no longer exist | `packages/gpu_visualization/CMakeLists.txt` |
| `packages/examples/*` (≈25 programs incl. a stray `RobotVsEnvironment - Kopie.cpp`, plus `swept_fitter`) | Same ICMAKER problem; not wired in | `packages/examples/CMakeLists.txt` |
| URDF robot (`gpu_voxels_urdf_robot`) | wrapped in `if(OFF)` | `robot/CMakeLists.txt` |
| Unit tests (`gpu_voxels/test`, `voxelmap/tests`, `octree/test`) | Present (`*.cu` + fixtures) but use ICMAKER; `add_subdirectory(tests)` commented out. Root sets `BUILD_TESTS ON` but nothing consumes it | CMake files |
| `doc/` (Doxygen), `gvl_ompl_planning/`, `example_how_to_link/` | Standalone; `doc/CMakeLists.txt` uses ICMAKER | — |

Consequence: **master compiles only the library**; the viewer, examples and tests are not buildable without porting their CMake. Whether `master` builds at all could not be verified here (no CUDA).

## `icl_core` (vendored, `packages/icl_core`, ≈12 k LOC)
FZI infrastructure: `icl_core` (utils), `icl_core_config` (XML/ini config), `icl_core_logging` (stream logging macros `LOGGING_*`), `icl_core_performance_monitor`, `icl_core_testsuite`. Used by every CUDA module, **not by the Vulkan backend** (since 2026-11-05 it is only configured and built with `GVL_BACKEND_CUDA`); GPU-agnostic.

## Other top-level items
`pom.xml` (Maven stub), `changelog.txt` (history 2014 →), `overlay-ports/` (vcpkg), `gvl_ompl_planning/` (OMPL planner example), `example_how_to_link/` (CMake consumer demo).
