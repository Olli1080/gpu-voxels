# Octree checkpoint (after S6a-S6c, before S6d) — index and result

- **Date:** 2026-10-30
- **Question:** The read-only core of the faithful octree (build, find, extract, collisions, two traversal implementations) exists. Do the measurements and the fault experience justify continuing the pointer-faithful route (Q19) for S6d (dynamic insert / propagate / rebuild), or should S6d switch to Option 2 (index pool) or Option 3 (sorted level arrays) of [design_options.md](../2026-10-29_octree_design_review/design_options.md)?
- **Scope / non-goals:** the measurements listed under "Measurements to take before S6d" in `design_options.md`, the balancer-vs-wavefront comparison of Q23, the pointer-fault experience, and the revisit triggers of [slice_plan.md](../2026-10-29_octree_design_review/slice_plan.md). Not measured: CUDA (no toolkit here), any other GPU, Release builds.
- **Status:** final; one decision for the user (Q24 in [open_questions.md](../../project_state/open_questions.md)).

| File | Content |
|---|---|
| [measurements.md](measurements.md) | the raw numbers: 64-bit sort, bytes per voxel for five workloads, frontier sizes, balancer vs wavefront |

## Method
All numbers come from `tests/vulkan/test_octree_checkpoint.cpp` (new, prints `[vulkan] ...` lines) and `tests/vulkan/test_octree_balancer.cpp`, run on 2026-10-30 in `build/verify-vk` (Debug, validation layer on), device Intel UHD Graphics (integrated), one run each (no repetition, so differences below ~15 % are noise). Frontier sizes come from the independent CPU recursion (`collide_tree(...).block_pairs` in `tests/reference/octree_reference.h`, new counter, mutation-tested) and are checked against the GPU volume in the same test. Bytes of the index pool (Option 2) and of the sorted level arrays (Option 3) are **computed** from the node counts of the real tree with the node sizes assumed in `design_options.md` (Option 2: 8 B per inner node, leaves as in the faithful tree; Option 3: 8 B key + 1 B status, 10 B with occupancy), not measured on an implementation.

## Findings
1. **64-bit sort costs 1.7-1.95x the 32-bit sort** (measured, high): 4 M pairs 492 ms vs 252 ms, 15.9998 M pairs (the limit) 1 262 ms vs 759 ms, including transfers. The design review expected about 2x per pass with 6 passes against 4. The octree build of 1 M voxels takes 0.3-0.9 s in total (the share of the sort was not measured separately).
2. **The sort limit is 65 535 x 256 = 16 776 960 elements, not 2^24** (measured, high): `Octree::kMaxVoxels`, `VoxelListBase::kMaxEntries` and `Primitives::kMaxElements` say 2^24 (16 777 216). A build or list insert of 16 776 961 .. 16 777 216 elements passes the class check and then throws `Primitives::record_sort_pairs: too many elements`. A clean error, but the constants are 256 too high: [K34](../../knowledge_base/known_issues.md).
3. **Memory per voxel of the faithful layout depends on the density** (measured nodes, computed alternatives, high for the nodes, medium for the alternatives):

   | Workload (deterministic tree) | faithful B/voxel | index pool | sorted levels |
   |---|---|---|---|
   | dense cube 128^3 | 2.1 | 2.1 | 10.3 |
   | sphere shell r = 300 (surface) | 9.9 | 9.9 | 12.9 |
   | 1 M random in 200^3 (12 %) | 15.1 | 15.1 | 16.4 |
   | 1 M random in 2048^3 (0.01 %) | 204 | 204 | 45.6 |
   | 1 M random in 16384^3 (sparse) | 396 | 396 | 72.5 |

   The probabilistic tree is 1.5-1.7x larger in the faithful layout (12 B inner nodes) and about the same in the sorted layout (see [measurements.md](measurements.md)). The cause of the sparse numbers is the block of 8 siblings: a lone voxel allocates one block per level, 7 of its 8 nodes are unknown fillers (56 M allocated nodes, 8 M known nodes for the sparse case). Sorted levels store only known nodes: 5.5x smaller for the sparsest input, 4.8x **larger** for the dense cube, equal for surfaces. This confirms the estimate "up to ~10x for dense data" of the design review and adds its mirror image: the faithful layout (and Option 2, which keeps the blocks) is the wasteful one for sparse data.
4. **Frontiers are small** (measured with the CPU recursion, high): for two sphere shells of ~280 k voxels the peak is 695 block pairs (level 1), for two clusters of 150 k voxels 7 609 (level 1), for two random sets of 150 k voxels 33 701 (level 5), i.e. 0.2-23 % of the voxel count, total over all levels 1.5 k-69 k. No chunking of the queue is needed at these sizes; the shape differs (clustered data peaks at the leaves, spread data in the middle levels), so a per-level queue sized by the blocks of the level is the safe bound.
5. **The load balancer is not competitive with the wavefront on this GPU** (measured, medium: one GPU, one run): tree vs tree 40-47 ms vs 82-110 ms, extract 690-750 ms vs 770-880 ms ([vulkan_octree_collision.md](../../knowledge_base/architecture/vulkan_octree_collision.md)). It stays selectable (Q23), the wavefront is the default.
6. **Pointer faults: none unexplained** (measured/experience, high): bad device addresses are silent or lose the device and the validation layer says nothing (K28, measured in the disabled experiments); the only real pointer defect found in S6a-S6c was the canonical-address problem on this driver (K29, found by the first build test, fixed with `sign_extend_48`, covered by tests). `check_tree` found the corruptions injected by the tests; no device loss occurred outside the deliberate experiments.

## Revisit triggers (slice_plan.md) against the evidence
| Trigger | Result |
|---|---|
| Two or more unexplainable device faults in S6b/S6c | **not met** — none unexplained (finding 6) |
| A target GPU without `bufferDeviceAddress` | **not met on the only GPU available** (Intel UHD has it); other GPUs unverified (not measured) |
| Balancer port not competitive with a plain wavefront | **met on this GPU** (finding 5), **resolved by Q23**: the wavefront is the default and S6d must not depend on the balancer |
| A requirement for trees above what the faithful allocation scheme provides | **not met** — no such requirement; the sparse-memory cost (finding 3) is the nearest thing |

## Recommendation / decision needed
**Continue the faithful route for S6d** (recommended), with these conditions:
1. S6d's `insert` / `propagate` use host-planned allocation and the wavefront style (one kernel per level), no dependence on the balancer (inferred from the CUDA design; new chunks are new buffers, existing node addresses stay valid when a tree grows).
2. Keep Option 3 documented as the fallback **for sparse data** (finding 3); decide again if a real workload of sparse, memory-bound trees appears. Option 2 does not help there (it keeps the blocks).
3. Fix or document K34 (lower the three constants to 16 776 960, or lift the sort limit with a 2-D dispatch) before S6d adds more callers.

The decision is the user's: **Q24** (continue faithful / switch to Option 2 or 3 / continue faithful but fix K34 first). Until answered nothing in S6d is started.

## What was NOT verified
CUDA behaviour and CUDA timings (no toolkit), any GPU other than the Intel UHD (the balancer ranking may differ on a GPU with many compute units), Release-build timings, repetition and variance of the timings, a real sparse-tree workload, the cost of dynamic updates (that is S6d), Option 2 / Option 3 as implementations (their bytes are computed from node counts), the frontier at 1 M voxels (only 150-283 k voxels, CPU recursion).

## Promoted knowledge
Findings 2 and 6 in [known_issues.md](../../knowledge_base/known_issues.md) (K34), the test in [testing.md](../../knowledge_base/architecture/testing.md), the decision row Q24 in [open_questions.md](../../project_state/open_questions.md).
