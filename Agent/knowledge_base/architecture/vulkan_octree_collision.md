# Octree collisions (slice S6c, 2026-10-30)

Status: **Tested, not Parity** (no CUDA toolkit here; the semantics come from reading the CUDA kernels). Extends `Octree` ([vulkan_octree_build.md](vulkan_octree_build.md)): collisions with bit maps, probabilistic maps, bit lists and Morton ids, tree against tree, collision flags. **Deviation from the slice plan:** all traversals are plain GPU kernels (one lookup per probe, one level-synchronous wavefront for tree-vs-tree) instead of a port of the persistent-block load balancer; the balancer port is postponed to the checkpoint, where it can be compared with these numbers (see [Measurements](#measurements)).

## API (`Octree.h`)
| Call | CUDA counterpart | Result |
|---|---|---|
| `collide_with(const BitVoxelMap&, options)` | `intersect_sparse(VoxelMap)` | colliding probes; optional unknown count and OR of the 256 bit vectors |
| `collide_with(const ProbVoxelMap&, options)` | `intersect_sparse(ProbVoxelMap)`, `intersect_load_balance(ProbVoxelMap)` | same without bit vectors (throws if `compute_voxel_flags`) |
| `collide_with(const BitVoxelList&, options)` | `intersect_sparse(VoxelList)` | same as bit maps |
| `collide_with_ids(span<uint64_t>, span<BitVector256>, options)` | `intersect_morton` (list class pending: ids come as a span) | flags optional (one per id); the offset is not applied to ids |
| `collide_with(Octree& other, OctreeTreeCollisionOptions)` | `intersect_load_balance(NTree)` | number of voxels in collision (64 bit) |
| `clear_collision_flags()` | collision clearing of the extract balancer | removes the COLLISION bit from every node |

Options of the probe queries: `min_level`, `offset_{x,y,z}`, `set_collision_flags`, `compute_voxel_flags`, `count_unknown`, `tree_occupancy_threshold` (10), `map_occupancy_threshold` (50).

## Semantics (from `TRAVES_MACRO`, `kernel_intersect_*`, `Intersect.cuh`)
- **Probe:** an occupied voxel of the input: bit maps / bit lists / flagged ids: any bit of the 256 bit vector (CUDA `isVoxelOccupied`; note the S1/S4 map-vs-map collisions ignore bit 0, the tree does not), probabilistic maps: value >= 50, unflagged ids: always. Position = coordinates + offset, Morton id of that.
- **Lookup:** from the root, descend into PART nodes while the node is above `min_level`; the node where the walk stops is tested (so `min_level` > 0 is conservative: a partly occupied box collides). Collision = deterministic: OCCUPIED bit, probabilistic: occupancy known and >= tree threshold. A probe *hits unknown space* when that node has the UNKNOWN bit / occupancy -128, independent of the collision.
- **Marks:** the node where the lookup stopped (leaf or inner) gets the COLLISION bit (atomic OR); `check_tree` ignores that bit when it recomputes statuses, `extract` selections see it (status byte index 32).
- **Tree vs tree:** pairs of nodes of one level; a pair collides when both are occupied; it counts as its whole box (8^level voxels) when both are uniform (no PART) or `level <= min_level`; otherwise it is refined, a uniform side standing for its own children; level 1 compares the leaves one by one (`min_level` 0). One node of the pair is marked (CUDA: the real node of tree A, else of B). The count is a sum over levels weighted by 8^level (counters per level because the device has no 64 bit atomics).

## Deviations from CUDA (all decided like Q16: outside is outside, no undefined behaviour)
1. **Probes that leave the grid are ignored.** CUDA computes `coordinate + offset` as signed 64 bit, casts to `uint32_t` and feeds the Morton code: a negative or too large coordinate wraps around and hits some far voxel ([K31](../known_issues.md)).
2. **Unknown counting only when requested.** CUDA reads the flag `collision_w_unknown` uninitialised when `compute_collsWithUnknown` is off and then increments a shared counter ([K32](../known_issues.md)).
3. **`IntersectVMap` is not a separate algorithm here:** it walks the tree over occupied nodes and tests the map voxels below them; by reading the kernel its observable result (count, voxel type flags, marks, `min_level` rule, offset sign) equals the map-driven lookup of `collide_with(ProbVoxelMap)`. Only the work distribution differs; a tree-driven variant would be a performance experiment.
4. **Roots without children:** CUDA starts tree-vs-tree with `root.getChildPtr()` of both trees (null for an empty tree: crash, [K33](../known_issues.md)). Here a root without children is a uniform node that stands for its children; a forged occupied uniform root against another tree counts the other tree's voxels; two uniform occupied roots count 8^14 voxels and mark the first root.
5. `min_level` >= 14 behaves like 13 (the traversal starts at the level of the root's children).
6. The unfinished/unused CUDA variants are not ported: `kernel_intersect` for host vectors (`voxelFlags` "not implemented"), `intersect_shared`, `*_wo_stack`, `*_smallStack`.

## Measurements
Intel UHD Graphics (integrated), Debug build, validation layer on, single runs (not benchmarks): a 1 M voxel tree (200^3 cube) against a 2 M voxel bit map (300 k occupied probes) takes about 30 ms for either tree kind; two trees of about 1 M voxels (13 736 colliding voxels) take about 52-55 ms. No CUDA numbers exist for comparison.

## Verification
- `test_vulkan`: 10 probe-query cases + 8 tree-vs-tree cases (see [testing.md](testing.md)). The probe tests build trees of 800-1 500 voxels, download the maps / lists with the already tested S1/S3/S4 accessors, derive the probes independently (own Morton code, offset, bounds) and compare counts, unknown counts, OR-ed flags, the marked nodes (`dump()` against the reference with the COLLISION bits), `check_tree()`, a repeated query, `clear_collision_flags()` and `extract` with a COLLISION selection; option variants: `min_level` 0/1/3/6, offsets, extreme tree thresholds (-128, -1, 127), map thresholds, each optional result alone. Further cases: ids outside the tree, duplicates counted twice, probes outside the grid (negative, 16384, offsets), lookups that stop at a missing child pointer, the empty tree, 2 M voxel map against a 1 M voxel tree checked with a `std::set` oracle.
- Tree-vs-tree: the plain recursion `collide_tree` in the reference (independent of the kernel's stand-in addresses) for hand-made and random sets, cubes (uniform nodes with kept children), empty trees, both kinds, `min_level` 0/1/2/4/13/14 with and without marks (both trees' dumps compared, flags cleared afterwards); at `min_level` 0 the result must also equal the size of the **intersection of the voxel sets**; identical / disjoint trees, forged uniform roots, invalid partners (other kind, same object), two trees of 1 M voxels against the set intersection.
- Mutation check: probe queries: 63 mutations (kernel, host code, reference); the first run left three survivors that were real test gaps (grid bound 16384, the two optional results always switched on together, a node with PART but no child pointer), all killed after adding tests. Tree vs tree: 54 mutations, 46 killed at once; six were equivalent (redundant checks in the kernel and the reference, removed: GT8, GT14, GT21, GT22, RT4, RT6) and two were real gaps (occupancy threshold of the leaves, root without child pointer), both killed by new tests; a rerun of all changed code (15 mutations) killed everything.

## Not verified / limits
Equality with CUDA is inferred. The number of work items per level is bounded by the allocated blocks of both trees; very large trees need the 32-bit counters (< 4 G pairs per level). A probe query uses one thread per input element (maps: per voxel, also the empty ones). The marking of one node per pair follows CUDA and means "who collided" is only partly visible on tree B.
