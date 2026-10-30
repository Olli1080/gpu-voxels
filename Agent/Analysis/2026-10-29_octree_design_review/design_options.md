# Octree: candidate designs, the recommendation and the decision

**Decision (user, 2026-10-29): start with Option 1 (pointer-faithful), keep Options 2 and 3 documented for later.** The reviewer recommended Option 2 (below); what changes for Option 1 is listed in [slice_plan.md](slice_plan.md) (buffer device addresses in the runtime, host-side pointer validation, load balancer port with the warp parts rewritten, checkpoint before the dynamic slice, revisit triggers). The text of all three options is unchanged so that a later switch does not need a new analysis.

Facts behind the statements: [octree_cuda_design.md](../../knowledge_base/architecture/octree_cuda_design.md), [octree_porting_hazards.md](../../knowledge_base/vulkan_migration/octree_porting_hazards.md). All designs below are **paper designs** (nothing prototyped, no performance data).

## Criteria
Semantic fidelity to the CUDA tree (so later CUDA comparison, Q5, and the existing `GpuVoxelsMap` interfaces stay meaningful) · implementation effort and risk · GPU suitability (irregularity, atomics, divergence) · memory per stored voxel · optional Vulkan features needed · testability against an independent reference · cost of a dynamic update · scaling beyond the 1 GiB binding limit (measured on this device).

## Option 1 — pointer-faithful port with buffer device addresses
Keep `InnerNode` with a 48-bit child address, allocate chunks with VMA as separate buffers, access them through `GL_EXT_buffer_reference`; port kernels and the load balancer 1:1 (warp parts rewritten).
- **+** smallest semantic distance, memory scales past 1 GiB, fastest to a first compiling port.
- **−** needs `bufferDeviceAddress` (optional; exists on this device); no bounds checking and no validation-layer help for node accesses (a wrong child address is a GPU fault); keeps `m_allocation_list`, 128-byte chunk alignment and the whole load balancer (3 280 lines tuned for one GPU); hardest to test and debug — exactly the failure mode of the Kompute attempt ([mistakes_to_avoid.md](../../knowledge_base/vulkan_migration/mistakes_to_avoid.md) C3).

## Option 2 — index-based growing pool, CUDA semantics, wavefront traversal (recommended)
Same nodes, same block-of-8 children, same status/flag rules, but:
- **Layout:** `inner[]` with 2 words per node (`status | flags<<8 | occupancy<<16`, `child block index`), `leaf[]` (status, plus occupancy for the probabilistic tree; packed 2-4 per word with `atomicOr` for flag bits, or one word per leaf — measure), a small header buffer (root, counts, capacities). Child index = index of the first node of the 8-block in `leaf[]` (if the parent has `ns_LAST_LEVEL`) or `inner[]`. Blocks are only **appended**, so indices stay valid when the pool is reallocated (capacity doubling by copy); 32-bit indices, 128 M inner nodes per binding.
- **Build:** identical to CUDA's bottom-up scheme: Morton ids (`uint64`) -> 64-bit-key radix sort -> per level unique parents (head flags + scan = the S4 `make_unique` pattern) -> allocate level array -> set nodes with child-block indices. No atomics.
- **Insert:** CUDA's two-pass scheme without device allocation: pass 1 per voxel finds the deepest existing node on its path and flags the first voxel of every new prefix (sorted input => neighbour comparison), per-level counts by scan; host grows the pool; pass 2 initialises the new blocks and links them. Deterministic indices, no atomics.
- **Propagate:** per-level lists of flagged nodes gathered top-down (breadth-first), then bottom-up kernels one level at a time (`getNewStatus` / max occupancy over the 8 children), then the `UPDATE_SUBTREE` top-down pass. One invocation per parent, looping over its 8 children (no 8-thread groups, no ballots).
- **Intersect tree-vs-map/list/Morton list:** one invocation per occupied map/list voxel, iterative lookup from the root (depth <= 15), exactly as `kernel_intersect_VoxelMap`; collision flag via `atomicOr` on the node word; counts in 32-bit.
- **Intersect tree-vs-tree and tree-vs-prob-map (`IntersectVMap`):** breadth-first wavefront: a queue of work items (a-node, b-node/coordinates, level, active flags) per level; one invocation per (item, child lane); next queue by count + scan + write; `min_level` ends a branch; counts of voxel volume (up to 8^14) summed per work group into 64-bit partials on the host (no 64-bit atomics). Frontier larger than a bound is processed in chunks (queue chunking replaces the load balancer's rebalancing).
- **Extract / rebuild:** same wavefront with output compaction; `rebuild()` = extract + build (CUDA's own approach), or direct compaction.
- **+** keeps semantics, structure and test oracles (`checkTree`, CUDA golden data later); portable (Vulkan 1.3 + `shaderInt64` only); all validation tooling works; replaces the largest piece of CUDA code (load balancer + warp reductions) by simple kernels and the primitives we already tested; mostly straightforward kernels.
- **−** still the biggest slice by far; dynamic insert/propagate stay intricate; the 1 GiB binding cap limits one tree to 128 M inner nodes unless several bindings are introduced later; wavefront memory must be bounded (chunking).

## Option 3 — sorted level arrays (implicit tree)
Per level one sorted array of Morton prefixes (`uint64`) with status (and occupancy); a node's children are the entries with `key >> 3` equal to its key; absent children mean UNKNOWN; uniform nodes carry no children.
- **+** no pointers or pool management, no `rebuild`; canonical form => tests can compare whole trees; every operation is sort/merge/scan/unique/binary-search (all available); tree-vs-tree is a merge join per level; no fillers for sparse regions.
- **−** *not* the CUDA structure: node identity, memory numbers, extraction order and serialization differ; dense regions cost a key per node (about 9 B vs 1 B per leaf in CUDA, possibly up to ~10x more memory, **to be measured**); an incremental insert is a merge, O(tree size) per update unless segmented (heavy for large trees on a weak GPU); inserting into a pruned (uniform) region needs node splitting, and the "children kept but ignored" state of CUDA has no equivalent (observable only through memory/`extract` of low levels).

## Comparison
| Criterion | Option 1 | **Option 2** | Option 3 |
|---|---|---|---|
| Fidelity to CUDA structure and semantics | highest | high (same nodes/rules, different memory plumbing) | medium (same observable statuses, different structure) |
| Effort / lines to write | lowest start, highest to stabilise | high | medium-high |
| Risk | high (memory faults, no oracle) | medium | medium (new semantics derivation, memory unknown) |
| Optional Vulkan features | `bufferDeviceAddress` | none | none |
| Memory vs CUDA | equal | equal (leaf packing to be chosen) | sparse: better, dense: worse |
| Dynamic update cost | like CUDA | like CUDA | O(tree) merge |
| > 1 GiB trees | yes | no (later: several bindings) | no (per-level arrays help: each level is its own binding) |
| Testability | poor | good (tree equality + `checkTree` + reference) | best (canonical form) |
| Needs 64-bit-key sort | yes | yes | yes |

## Recommendation
Option 2, in read-only-first order, with Option 3 as the fallback if the dynamic operations of Option 2 turn out disproportionately hard. Both need the same prerequisite work (64-bit sort keys, Morton encode/decode, a CPU reference of the node semantics), so starting with the common slices S6a-S6c ([slice_plan.md](slice_plan.md)) does not commit to either beyond S6b.

## Measurements to take before S6d (dynamic updates)
Done 2026-10-30, results in [../2026-10-30_octree_checkpoint/measurements.md](../2026-10-30_octree_checkpoint/measurements.md) (the list below is the original request; the frontier was measured with the CPU recursion at 150-283 k voxels).
- Intel UHD: time of the existing sort/scan at 4-16 M elements with 64-bit keys (expected: 2x the 32-bit radix cost per pass, 6 passes for 42 bit).
- Frontier sizes of tree-vs-tree intersect on random trees (decides queue chunking).
- Node counts and bytes per voxel for dense and sparse inputs under Option 2 vs 3 (decides whether Option 3 stays a fallback).
