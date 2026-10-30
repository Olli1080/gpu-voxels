# Workspace Survey and Post-mortem of Earlier Migration Attempts

- **Date:** 2026-10-22
- **Question:** What is in the three workspaces under `C:\Code`, how does the CUDA library work, and why did the earlier CUDA→Vulkan/SYCL attempts fail?
- **Scope / non-goals:** read-only inspection; no build, no test run, no changes to source or to the sibling folders. Not a code review of correctness.
- **Status:** final (superseded only by newer facts in `knowledge_base/`)

## Method
- Directory/ext/LOC counts with PowerShell/`find`/`wc`; `grep -r` for CUDA/Thrust API usage over `packages/` (exclusions as noted in [cuda_coupling_inventory.md](../knowledge_base/architecture/cuda_coupling_inventory.md)).
- Git inspection in all three folders: `git log`, `git branch -a`, `git status --short`, `git diff --stat`, `git ls-tree`, `git show --stat`.
- Read: root/package `CMakeLists.txt`, presets, `vcpkg*.json`, overlay ports, key headers (`GpuVoxels.h`, `GpuVoxelsMap.h`, `common_defines.h`, voxel headers, vis interface), and in `C:\Code\gpu-voxels`: `AGENTS.md`, `MAIN_PLAN.md`, `MIGRATION_PLAN_KOMPUTE.md`, `agent/**`, `plans/**`, `code/CMakeLists.txt`, `KomputeBridge.*`, `DeviceVector.hpp`, selected adapters/tests, and the old test logs (UTF-16 → UTF-8 via `iconv`).
- Machine probe: `Get-Command` for toolchain, `vulkaninfo --summary`, `Get-CimInstance Win32_VideoController`.

## Findings (see linked files for evidence)
1. Master builds only library targets; viewer/examples/tests/URDF are disabled or on removed ICMAKER CMake. *(inferred, high)* → [library_modules.md](../knowledge_base/architecture/library_modules.md)
2. CUDA coupling is broad but regular: ≈100 kernels, Thrust usage reducible to ~15 primitives, hardest areas are the octree (pointers, 64-bit, warp ops, load balancer), PBA (shared memory) and the visualizer's IPC/GL interop. *(measured counts, medium on kernel count)* → [cuda_coupling_inventory.md](../knowledge_base/architecture/cuda_coupling_inventory.md)
3. Three sequential attempts in ~5 weeks (SYCL → Kompute behind a CUDA-emulation bridge → Kompute rewrite with adapters), each abandoning the previous one's work. *(measured from git)* → [attempt_history.md](../knowledge_base/vulkan_migration/attempt_history.md)
4. The last attempt's checkboxes overstate reality: stub bodies, `return 0/false`, `#if 0`, disabled targets; destructive changes had to be repaired; "parity" tests used self-written references and a mocked CUDA header; no CUDA hardware existed to verify parity. *(measured/inferred, high)* → [mistakes_to_avoid.md](../knowledge_base/vulkan_migration/mistakes_to_avoid.md)
5. Useful artefacts exist (≈50 GLSL shaders, PBA, scan/radix sort, flat octree layout, ROS-free URDF loader) but 14 shaders are uncommitted and unique. → [reusable_assets.md](../knowledge_base/vulkan_migration/reusable_assets.md)
6. This machine cannot run CUDA and has no Vulkan validation layer or shader compiler installed. *(measured)* → [current_state.md](../project_state/current_state.md)

## What was NOT verified
- That `master` builds (no CUDA). That any old test passes (not run; logs read only). Performance numbers of old attempts. Whether old shaders are correct. Kernel counts are estimates (declaration + definition double counting).
- Reasons for abandoning SYCL and for restarting Kompute: not documented anywhere found.
- Contents of `gpu-voxels__`'s git objects (branch unborn) beyond file-level diff against master.

## Recommendation / decision needed
Answer [open_questions.md](../project_state/open_questions.md) Q1–Q9, then adopt and refine [vulkan_migration_plan_DRAFT.md](../plans_wip/vulkan_migration_plan_DRAFT.md).

## Promoted knowledge
Everything above is already distributed into `knowledge_base/`, `agent_policies/` and `project_state/`.
