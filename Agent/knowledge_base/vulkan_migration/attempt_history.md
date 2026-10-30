# Attempt History

Author of all commits below: Olli1080 (agent sessions ran under their git identity; the code contains a path `C:/Users/olive/.gemini/tmp/…`, i.e. at least one session was driven by a Gemini CLI agent — inferred). Reference repo: `C:\Code\gpu-voxels` (remotes `origin` Olli1080, `upstream` FZI). `master` there == this workspace (`dd47249`, 2025-06-23).

## Overview
| # | Name | Where | Dates | Strategy | End state |
|---|---|---|---|---|---|
| 0 | Baseline | this workspace, `master` | ≤ 2025-06-23 | CUDA 12, CMake 3.23, C++20, Eigen vectors, vcpkg | Library target set builds on a CUDA machine (not verifiable here) |
| 1 | SYCL / AdaptiveCpp | branch `sycl_migration` (5 commits ahead of master) | 2026-03-26 → 04-01 | Macro bridge `SyclBridge.h` (`GVL_*` macros hide `__global__`, `cudaMalloc`, launches) + `oneDPLBridge.h` (Thrust → `parallel::`) | Abandoned after ~1 week. Every verification item in `SYCL_MIGRATION_MASTER_PLAN.md` is unchecked; no evidence it ever ran. Reason for switching: not documented |
| 2 | Kompute via the same bridge | branch `kompute_backend` (16 ahead; = #1 + 11 commits) | 2026-04-02 → 04-08 | Keep the original class structure; compile `.cu` as plain C++; `KomputeBridge` emulates the CUDA runtime (pointer registry, `GVL_LAUNCH_KERNEL(name,…)` stringifies the kernel name and dispatches `<name>.comp`); one GLSL shader per kernel | 35 `.comp` shaders committed; standalone `test_kompute_*` programs pass on Intel UHD (claimed). Library-level API never exercised through `GpuVoxels` |
| 3 | Kompute rewrite + adapters | branch `kompute` (14 ahead of master) | 2026-04-26 → 05-02 | New self-contained `migration_lib` in `code/` (Kompute classes on Eigen types, 9 shaders in `code/shaders`), then **adapter** classes in `packages/` delegating to it; later phases for type unification, CPU parity, re-introduction, validation, visualizer | Last commit `43a68a5` (2026-05-02). Working tree has **303 uncommitted paths** (203 modified, 1 deleted, 99 untracked — incl. the 49 attempt-2 shaders and the bridge headers). Several adapters are stubs |
| – | Snapshot `gpu-voxels__` | `C:\Code\gpu-voxels__` | files dated 2026-04-01…04-25 | Looks like a checkout of attempt 2's layout (root CMake lists `packages/**/kernels/*.comp`, macros `GVL_HANDLE_ERROR`, `parallel::`) | Git state: branch `kompute_backend` **unborn (no commits)**; 159 paths differ from master; `core/src` and `gpu_voxels/src` directories missing; `AGENTS.md` points to a non-existent `agent/summary.md`. Treat as a dead copy |
| – | `kompute_migration` | branch in `C:\Code\gpu-voxels` | – | – | Points at `master` (`dd47249`): empty placeholder |

## Attempt 1 — SYCL (what existed)
Docs: `agent/plans_wip/SYCL_MIGRATION_MASTER_PLAN.md`, `agent/project_state/initial_analysis.md`. Ideas: Thrust→oneDPL, USM for memory, `GVL_KERNEL_CTX` + indexing macros (`GVL_GET_GLOBAL_ID_X`) so kernel bodies stay source-compatible, `GVL_WITH_SYCL` CMake option, `install_sycl_deps.py` (only opens download URLs). The `Modernization` commit (`5cb1bf8`) also **deleted the `eigen3` overlay port** (16 files, −165 lines) — harmless without nvcc, but it ties the build to a CUDA-free assumption.

## Attempt 2 — Kompute bridge (what existed)
- `KomputeBridge.{h,cpp}` (145 + 393 lines): `KomputeManager` singleton (Vulkan instance/device), `malloc/free/memcpy/memset/synchronize` mapped to `kp::Tensor` (`eDeviceAndHost`), pointer→tensor registry, shader registry (`auto_register_shaders(dir)`), `launch_generic(name, grid, block, args…)` that sorts args into pointer args (tensors) and POD args (push constants).
- `oneDPLBridge.h` (829 lines): Thrust-like `device_vector`, `zip_iterator`, `copy/fill/transform/sort_by_key/unique_by_key/scan` with GPU radix sort and scan shaders; `SyclBridge.h` (288 lines) with intrinsic mocks to compile `.cu` as C++.
- Kernels: voxelmap (15 shaders incl. PBA phases 1–3, transpose, jump-flood, exact distances), voxellist (3), octree (15), helpers (2 + pack/unpack/scan/radix).
- Tests: `test_kompute_{minimal,kernels,voxellist,octree_*,sort,shader_reg,all_collisions}` as stand-alone executables registered in the root `CMakeLists.txt`.

## Attempt 3 — Kompute rewrite (what existed)
- Plan documents: `MAIN_PLAN.md`, `plans/PHASE_1…13`, `FIX_DESTRUCTIVE_CHANGES.md`, `FIX_VISUALIZER_HANG.md`, `IN_PROCESS_VISUALIZATION.md`, `REMAINING_TASKS.md`, `agent/**`, `AGENTS.md` (handover), `MIGRATION_PLAN_KOMPUTE.md`.
- `code/`: `VoxelMap`, `VoxelList`, `ProbVoxelMap`, `DistanceVoxelMap` (PBA via `runPBA()` + a JFA variant), `CountingVoxelList`, `NTree`, `PrimitiveArray`, `MetaPointCloud`, `DeviceVector` (host `std::vector` mirror + `kp::Tensor`), plus `robot/` (urdfdom + orocos-kdl, own `urdf_to_kdl`), 19 Boost.Test sources in `code/tests`.
- Adapters in `packages/gpu_voxels/**` (`BitVoxelMap.cpp`, `ProbVoxelMap.cpp`, `BitVoxelList.cpp`, `urdf_robot.cpp`, …) converting between `gpu_voxels::*` and `gvl::migration::*`.
- Leftovers: 17 `test_octree_build_out_v*.txt` (UTF-16 console dumps from debugging the octree test), `fix_leaks.py` (regex rewrite of `record/eval` sequences into `eval<>` one-liners), `test_sizeof.cpp`, `.old` files in `voxelmap/include`.

## How to look at the old code without disturbing it
```bash
git -C C:/Code/gpu-voxels log --oneline master..kompute_backend      # attempt 2 commits
git -C C:/Code/gpu-voxels show kompute_backend:packages/gpu_voxels/gpu_voxels/voxelmap/src/kernels/kernelPBAphase1FloodZ.comp
git -C C:/Code/gpu-voxels ls-tree -r --name-only kompute_backend | grep '\.comp$'
```
Never check out, reset or clean in that repo: its uncommitted work is the only copy (see [reusable_assets.md](reusable_assets.md)).
