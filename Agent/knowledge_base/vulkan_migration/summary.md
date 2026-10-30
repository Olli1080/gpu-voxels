# vulkan_migration — Index

Everything known about moving GPU-Voxels from CUDA to Vulkan, including the post-mortem of three earlier attempts (Mar–May 2026) that live in the sibling workspaces `C:\Code\gpu-voxels` and `C:\Code\gpu-voxels__`.

| File | Content |
|---|---|
| [attempt_history.md](attempt_history.md) | Timeline and shape of each attempt (SYCL → Kompute bridge → Kompute rewrite + adapters), branch/commit pointers, what state they ended in |
| [mistakes_to_avoid.md](mistakes_to_avoid.md) | **Read first.** Post-mortem: concrete mistakes with evidence, root causes, and the counter-measure for each |
| [kompute_and_glsl_gotchas.md](kompute_and_glsl_gotchas.md) | Technical lessons that remain valid regardless of strategy: std430 layout, push constants, include directive, Kompute lifetime/leaks, shader build, numeric details |
| [reusable_assets.md](reusable_assets.md) | What is worth salvaging from the old attempts (shaders, algorithms, tests) and exactly where it is; what is *not* worth salvaging |
| [kompute_reevaluation.md](kompute_reevaluation.md) | Is Kompute mature enough now? Release/commit/issue facts (2026-10-22), packaging, alternative in REACH, recommendation |
| [octree_porting_hazards.md](octree_porting_hazards.md) | Octree: device facts (measured), CUDA constructs vs Vulkan, latent CUDA bugs (Windows `INVALID_VOXEL`, warp-synchronous code, constants) |
| [design_options.md](design_options.md) | Open architecture decisions for the new attempt (API binding, shader language, memory model, CUDA coexistence, visualization) with trade-offs — undecided, see [open_questions.md](../../project_state/open_questions.md) |

Related: policies in [migration_guardrails.md](../../agent_policies/migration_guardrails.md); inventory of what must be ported in [cuda_coupling_inventory.md](../architecture/cuda_coupling_inventory.md); draft plan in [plans_wip](../../plans_wip/summary.md).
