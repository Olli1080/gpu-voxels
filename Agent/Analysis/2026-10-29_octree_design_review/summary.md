# Octree design review (slice S6) — index and result

- **Date:** 2026-10-29
- **Question:** How should the CUDA octree (`NTree`/`GvlNTree`, ≈16 200 lines, the last and riskiest module) be brought to Vulkan: which data structure, which traversal scheme, which scope first?
- **Scope / non-goals:** design and scope decision only; no octree code written. Visualiser transport (shared memory, GL interop) and the Kinect/ROS front ends are noted but not designed.
- **Status:** final; decisions Q19-Q22 taken by the user on 2026-10-29 (see [Decisions](#decisions-2026-10-29)).

| File | Content |
|---|---|
| [design_options.md](design_options.md) | three candidate designs, criteria, evaluation, the recommended algorithms per operation |
| [slice_plan.md](slice_plan.md) | proposed slices S6a-S6f, prerequisites in the other modules, test strategy, decisions needed |
| [../../knowledge_base/architecture/octree_cuda_design.md](../../knowledge_base/architecture/octree_cuda_design.md) | facts about the CUDA octree (data model, memory management, operations, load balancer) |
| [../../knowledge_base/vulkan_migration/octree_porting_hazards.md](../../knowledge_base/vulkan_migration/octree_porting_hazards.md) | device facts (measured), CUDA constructs vs Vulkan, latent CUDA problems |

## Method
Read, in `packages/gpu_voxels/gpu_voxels/octree` on `master` (commit `dd47249` plus our uncommitted work): `NTree.h/.hpp`, `NTreeData.h`, `Nodes.h`, `EnvironmentNodes.h`, `EnvNodesProbabilistic.h`, `Morton.h`, `DataTypes.h`, `CommonValues.h`, `GvlNTree.h/.hpp`, `kernels/kernel_common.h`, parts of `kernels/kernel_Octree.h`, `load_balancer/*` (framework, intersect config), `src/Octree.cu`; counted lines and CUDA primitives with `wc`/`grep`; queried the device with `vulkaninfoSDK` (Intel UHD). Nothing could be run (no CUDA). Reuse of earlier analysis: [cuda_coupling_inventory.md](../../knowledge_base/architecture/cuda_coupling_inventory.md), [mistakes_to_avoid.md](../../knowledge_base/vulkan_migration/mistakes_to_avoid.md).

## Findings
1. **The tree is level-structured and bottom-up buildable** — `build` is sort + per-level unique-parent counting + scan + one array per level (inferred, high). The primitives we already have (radix sort, scan, compaction, unique, binary search) cover it, except for **64-bit keys** (Morton ids need 42 bit; our sort is 32-bit keys, 2^24 elements).
2. **Pointers are cheap to remove**: nodes point to contiguous blocks of 8 children; allocation is host-decided from a counting pass, never in kernels (inferred, high). Indices into a growing pool preserve the structure (and make `m_allocation_list` and pointer-width packing unnecessary).
3. **The load balancer does not need forward-progress guarantees**: blocks only read an idle counter, never wait for each other (inferred, high). It is portable, but it is also the biggest, most hardware-tuned piece (3 280 lines, 2688 tasks x 128 threads, tuned for a GTX Titan). A level-by-level wavefront with scan/compaction replaces it (proposed, medium confidence until prototyped).
4. **Warp-level code is the real rewrite**: 8 threads per node, `__ballot`/`__popc`/volatile shared memory assuming 32-wide lockstep warps (`kernel_common.h`). On this device subgroups are 8-32 wide (measured). One invocation per parent looping over its 8 children avoids subgroup assumptions (high).
5. **The semantics are subtle but small** (inferred, high): 8 status bits, `getNewStatus` rule, uniform nodes keep children but are treated as leaves, probabilistic parent = max of children. A CPU reference of exactly this (independent of any implementation) is cheap and allows exact tree-equality tests.
6. **Latent CUDA problems** that a port must not copy: Windows `INVALID_VOXEL = ULONG_MAX` (32 bit) vs 64-bit ids, warp-synchronous code without `__syncwarp`, assert-only range checks, inconsistent probability constants (100 vs 72 update, threshold 10 vs 50), ignored `resolution_level`/`coll_threshold` in the map overloads ([hazards page](../../knowledge_base/vulkan_migration/octree_porting_hazards.md)).
7. **Hard device limits**: a storage buffer binding holds 1 GiB (128 M 8-byte nodes), no 64-bit atomics (measured); `bufferDeviceAddress` exists here but is optional by our own feature policy.
8. **Scope is far from uniform**: build/find/extract/collide are the library-relevant core; insert + propagate are the dynamic core; Kinect/sensor pipeline, free-space ray casting, serialization, visualisation (`VisNTree` is even commented out of the CMake target) are peripheral; several `GvlNTree` operations are log-only stubs in CUDA already.

## What was NOT verified
Nothing was run: no CUDA behaviour, no performance of any design (the wavefront and the sorted-array alternative are paper designs), no real-world tree sizes. Node byte sizes of the probabilistic variants are inferred. The Linux behaviour of `INVALID_VOXEL` is known from the C standard, not from a build. `propagate`, `Extract` and `IntersectVMap` kernels were skimmed, not read in full.

## Recommendation (reviewer)
Build **Option 2** of [design_options.md](design_options.md): CUDA's node semantics and block-of-8 layout, **index-based growing pool**, level-synchronous wavefront kernels instead of the persistent load balancer, one invocation per parent/voxel (no warp tricks), in the slice order of [slice_plan.md](slice_plan.md) — read-only operations first (build, find, extract, collide), dynamic operations (insert, propagate, rebuild) after that, sensor pipeline only on demand. Keep the sorted-level-array design (Option 3) as fallback if insertion/propagation prove too intricate; decide after S6b/S6c have numbers.

## Decisions (2026-10-29)
Taken by the user after this review; also recorded in [open_questions.md](../../project_state/open_questions.md).
- **Q19 representation: pointer-faithful port first** (Option 1; the user's wording: "why not try faithful first and document the other options for later"). Options 2 and 3 stay documented in [design_options.md](design_options.md) as the fallback with explicit revisit triggers ([slice_plan.md](slice_plan.md)). Consequence: `bufferDeviceAddress` becomes a feature the octree needs (optional for the rest of the runtime); the reviewer's concerns (no bounds checking, hardest to debug) are answered by host-side pointer validation in every test and a checkpoint before the dynamic slice.
- **Q20 first milestone: read-only core** (build, find, extract, collisions); dynamic insert/propagate/rebuild after a checkpoint.
- **Q21 variants: all** — deterministic tree, probabilistic tree and the Morton voxel list (so voxel lists need 64-bit ids).
- **Q22 I/O: no promise of CUDA-format compatibility; put an interface where the formats may split** (`OctreeStorage`), native format first, a CUDA-format reader can be added behind it.

## Promoted knowledge
[octree_cuda_design.md](../../knowledge_base/architecture/octree_cuda_design.md), [octree_porting_hazards.md](../../knowledge_base/vulkan_migration/octree_porting_hazards.md); plan row S6 in [vulkan_migration_plan_DRAFT.md](../../plans_wip/vulkan_migration_plan_DRAFT.md).
