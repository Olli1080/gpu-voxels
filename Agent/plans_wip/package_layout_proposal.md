# Package Layout and CMake Modernisation — PROPOSAL

**Status:** L1–L4 **implemented 2026-10-22** (user approved; D1 default = module-local layout kept), uncommitted in the working tree. Evidence per step below. L5/L6 not started. Decisions D2–D5 still open. Model: [reach_reference.md](../knowledge_base/reach_reference.md). Feeds step P1.4 of [vulkan_migration_plan_DRAFT.md](vulkan_migration_plan_DRAFT.md).

## Goals
1. Configure and build **without CUDA** (this machine) and **without Vulkan**, selected by options — enables comparing backends (Q1: CUDA stays "for now").
2. One place for dependency discovery; slim root CMake; reproducible presets.
3. Backend code isolated so CUDA and Vulkan never mix in one source file (guardrail 1).
4. Tests with one wrapper, categories, GoogleTest.

## Current problems (measured, see [build_system.md](../knowledge_base/architecture/build_system.md))
`project(... LANGUAGES CXX CUDA)` is unconditional; `find_package` calls sit inside module CMake files (`helpers`: CUDAToolkit, Stb, PCL); circular link deps voxelmap⇄voxellist⇄octree; dead ICMAKER CMake for examples/tests/viewer; `BUILD_TESTS` hard-set; preset generator pinned to VS 2022.

## Proposed structure
```
CMakeLists.txt                  # slim: include cmake/*.cmake, add_subdirectory(packages), tests
cmake/Options.cmake             # GVL_BACKEND_CUDA / GVL_BACKEND_VULKAN, BUILD_TESTS, … → VCPKG_MANIFEST_FEATURES (before project())
cmake/FindDependencies.cmake    # every find_package, per option
cmake/Shaders.cmake             # glslang/slang → .spv custom commands, no absolute paths
cmake/Tools.cmake  SourceGroups.cmake  ProjectFolders.cmake   # as in REACH
vcpkg.json                      # features: cuda, vulkan, pcl/kinect, tests
CMakePresets.json               # version 7: base + cuda, vulkan, both; VS generator and Ninja Multi-Config
packages/gpu_voxels/gpu_voxels/<module>/…   # unchanged locations (see Decisions D1)
  backend/                      # new: gpu_voxels_compute (abstract buffer/dispatch interface, header-only API)
  backend_cuda/                 #      wraps today's cudaMalloc/Thrust/launch code (moves behind the interface gradually)
  backend_vulkan/               #      runtime on vulkan.hpp + VMA, shaders/*.comp|slang
tests/                          # gvl_add_test(), test_categories.cmake, fixtures/, reference/ (independent CPU references)
```
Module CMake follows the schema already used in the repo (FILE_SET HEADERS) plus REACH rules: no `file(GLOB)`, no `find_package` in modules, targets only linked.

## Steps (each independently reviewable; none changes behaviour)
| ID | Step | Verification |
|---|---|---|
| L1 | Add `cmake/` modules; move `find_package`s out of module CMake files; keep CUDA default ON | **Done.** `cmake/{Options,Backends,FindDependencies,Tests}.cmake`; CUDAToolkit/Stb/PCL discovery moved out of `helpers`. *Compiles*: configure+generate+build with CUDA OFF OK. Not verified: configure with CUDA ON (no nvcc here; error path verified, the success path is unchanged logic) |
| L2 | Make `LANGUAGES CXX` + `enable_language(CUDA)` conditional; guard every `.cu` target with the CUDA option | **Done.** `LANGUAGES CXX` + conditional `enable_language(CUDA)`; CUDA modules skipped when OFF. Verified: CUDA=OFF configure/build/test OK; CUDA=ON without nvcc stops with the intended message; Vulkan=ON (no CUDA) configure OK (VMA warning without vcpkg feature) |
| L3 | Presets v7 (`cuda`, `vulkan`, `both`, Ninja variant); `vcpkg.json` features; decide vcpkg source (Q10) | **Partly done.** Presets v7 + `vcpkg.json` features written and JSON-validated; `cmake --list-presets` (configure/build/test) OK; `vcpkg install --dry-run` resolved all feature sets with a modern vcpkg. **Not run:** `cmake --preset …` itself (the Vulkan feature install was not exercised); vcpkg source decision (Q10) open |
| L4 | Test infrastructure: GoogleTest, `gvl_add_test`, categories, one smoke test | **Done.** GoogleTest + `gvl_add_test` + categories + 2 smoke targets (7 cases): all pass (Debug, CUDA=OFF). See [testing.md](../knowledge_base/architecture/testing.md) |
| L5 | Port ICMAKER CMake of examples/tests/viewer to plain targets (separate task; gated by Q6) | Per target |
| L6 | Break voxelmap/voxellist/octree cycle with interface headers (only when backend interface needs it) | Link order test |

## Decisions needed
| ID | Question | Default |
|---|---|---|
| D1 | Move to REACH-style `include/`+`src/` trees? Large rename diff, breaks downstream include paths? (public include path `gpu_voxels/...` can stay identical either way) | **Don't move** now; keep module-local layout |
| D2 | vcpkg source: current submodule (2025-06-06, bootstrapped), REACH's newer vcpkg checkout, or system SDK + selective vcpkg? (REACH's vcpkg uses private overlays) | Keep submodule untouched for CUDA path; for Vulkan use the installed Vulkan SDK + minimal vcpkg ports until decided |
| D3 | Shader compilation at **build time** (offline SPIR-V, shipped next to binary) or runtime (REACH: glslang library at runtime)? | Build time (`glslangValidator`/`glslc`/`slangc` from the SDK), no runtime compiler dependency |
| D4 | Adopt REACH's `.clang-format`, `.clang-tidy`, `.clangd`? | Ask; don't reformat existing files |
| D5 | Compiler cache / PCH | Optional later |
