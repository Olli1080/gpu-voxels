# AGENTS.md — entry point for agents

**Project:** GPU-Voxels (fork by Oliver Zahn) — C++20 library for volumetric collision detection and
voxel maps on the GPU. Today it is **CUDA + Thrust**. The goal of this workspace is a migration to **Vulkan**.

**Read first:** [Agent/summary.md](Agent/summary.md) — index of everything agents need to know.

## Must-know
1. All agent knowledge lives in `Agent/`. Every folder there has a `summary.md` index; keep them current.
2. Rules you must follow: [Agent/agent_policies/summary.md](Agent/agent_policies/summary.md).
3. Two earlier Vulkan/SYCL migration attempts failed (`C:\Code\gpu-voxels`, `C:\Code\gpu-voxels__`).
   Read [Agent/knowledge_base/vulkan_migration/mistakes_to_avoid.md](Agent/knowledge_base/vulkan_migration/mistakes_to_avoid.md) **before** touching GPU code.
4. This workspace is `master` (CUDA, last upstream commit 2025-06-23). The two sibling folders are reference material only — do not build on their working trees.
5. This machine (i9-14900K, since 2026-10-31) has an **NVIDIA RTX 4090 and CUDA 12.8**, and the CUDA side builds and runs here ([build route](Agent/Analysis/2026-11-03_cuda_parity.md)); **never claim CUDA parity for anything that was not compared against a CUDA run**: compared so far are only the five scenarios of that page (list x map collisions, list merges, map x map collisions, prob. map inserts, sensor data); everything else is "Tested, not Parity". Vulkan SDK 1.4.363.0 (validation layer, glslang, dxc, slang) is installed. CUDA is to stay buildable for comparison; `C:\Code\REACH` is the reference for CMake/package conventions.
6. A checkbox in a plan is not evidence. "Done" means built + tested + no stubs; see [definition_of_done.md](Agent/agent_policies/definition_of_done.md).
7. Never delete or stub out functionality to make a build pass.
8. Branch `vulkan-migration` and local commits are allowed (user decision Q11, 2026-10-29); never push, open PRs or touch the `vcpkg` submodule pointer unless the user asks.
9. Keep documents ≤ 400 lines; split by aspect, update the indexes.
10. Open decisions for the user: [Agent/project_state/open_questions.md](Agent/project_state/open_questions.md).
11. Verification tooling (mutation runners, doc checker `python tools/check_docs.py`): [tools/README.md](tools/README.md), method in [Agent/knowledge_base/mutation_testing.md](Agent/knowledge_base/mutation_testing.md).
12. The move to the i9-14900K machine is done (2026-10-31): [Agent/project_state/machine_change_14900k.md](Agent/project_state/machine_change_14900k.md) (setup, what only exists on one machine) and [Agent/Analysis/2026-10-31_machine_baseline.md](Agent/Analysis/2026-10-31_machine_baseline.md) (baseline). Q5 (CUDA parity) was released by the user on 2026-11-03 ("ja und dann cuda").

## Current state
See [Agent/project_state/summary.md](Agent/project_state/summary.md).
