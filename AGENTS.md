# AGENTS.md — entry point for agents

**Project:** GPU-Voxels (fork by Oliver Zahn) — C++20 library for volumetric collision detection and
voxel maps on the GPU. Today it is **CUDA + Thrust**. The goal of this workspace is a migration to **Vulkan**.

**Read first:** [Agent/summary.md](Agent/summary.md) — index of everything agents need to know.

## Must-know in 10 lines
1. All agent knowledge lives in `Agent/`. Every folder there has a `summary.md` index; keep them current.
2. Rules you must follow: [Agent/agent_policies/summary.md](Agent/agent_policies/summary.md).
3. Two earlier Vulkan/SYCL migration attempts failed (`C:\Code\gpu-voxels`, `C:\Code\gpu-voxels__`).
   Read [Agent/knowledge_base/vulkan_migration/mistakes_to_avoid.md](Agent/knowledge_base/vulkan_migration/mistakes_to_avoid.md) **before** touching GPU code.
4. This workspace is `master` (CUDA, last upstream commit 2025-06-23). The two sibling folders are reference material only — do not build on their working trees.
5. This machine has **no CUDA toolkit and no NVIDIA GPU** (Intel UHD only): CUDA code cannot be built or run here; never claim CUDA parity was verified. Vulkan SDK 1.4.363.0 (validation layer, glslang, dxc, slang) is installed. CUDA is to stay buildable for comparison; `C:\Code\REACH` is the reference for CMake/package conventions.
6. A checkbox in a plan is not evidence. "Done" means built + tested + no stubs; see [definition_of_done.md](Agent/agent_policies/definition_of_done.md).
7. Never delete or stub out functionality to make a build pass.
8. Branch `vulkan-migration` and local commits are allowed (user decision Q11, 2026-10-29); never push, open PRs or touch the `vcpkg` submodule pointer unless the user asks.
9. Keep documents ≤ 400 lines; split by aspect, update the indexes.
10. Open decisions for the user: [Agent/project_state/open_questions.md](Agent/project_state/open_questions.md).

## Current state
See [Agent/project_state/summary.md](Agent/project_state/summary.md).
