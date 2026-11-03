# Vulkan voxel lists (slice S4, 2026-10-23)

Classes in `gpu_voxels_vulkan` (`VoxelList.h`): `VoxelListBase` (shared behaviour), `BitVoxelList` (BitVoxel<256>), `CountingVoxelList` (int8 count), plus `BitVoxelMap::merge_voxel_list()` and `BitVoxelMap::voxel_buffer()`. Primitives behind them: exclusive scan and stable radix sort ([Primitives.h](../../../packages/gpu_voxels/gpu_voxels/vulkan/include/gpu_voxels/vulkan/Primitives.h), slice P2).
Status: **Tested** against an independent CPU reference; **not Parity** (no CUDA run, Q5). Not part of S4: `ProbabilisticVoxelList`, offset collisions, `collideWithTypes` / bit-mask collision of lists (both S7-5: [vulkan_gpu_voxels_examples.md](vulkan_gpu_voxels_examples.md)), disk I/O, `getCenterOfMass`, `extractCubes`, `clone`/`equals`, the abstract CUDA/Vulkan interface.

## Data model
A list = three parallel device buffers: `ids` (linear voxel index of the *reference map*, `z*dx*dy + y*dx + x`, uint32), `coordinates` (3 x uint32), payload (Bit: 8 words per entry, bit i = meaning i; Counting: one word holding the sign extended int8 count). Invariant after every public operation: sorted by id, ids unique. Buffers never have size 0 (an empty list holds one dummy element), `size()` is the entry count.

## What was ported
| CUDA (`TemplateVoxelList` / `BitVoxelList` / `CountingVoxelList`) | Vulkan | Notes |
|---|---|---|
| `insertPointCloud`, `insertCoordinateList`, `insertMetaPointCloud` (both overloads) | `insert_point_cloud`, `insert_coordinate_list`, `insert_meta_point_cloud` | append + drop outside + `make_unique`; return the number of dropped inputs |
| `make_unique` (`sort_by_key`, reverse `inclusive_scan` with `Merge`, `unique_by_key`) | `make_unique` | different algorithm, same result: radix sort of (id, index), head flags, exclusive scan = group index, one kernel reduces every entry into its group (`atomicOr` / `atomicAdd`), counting sums wrapped to int8 afterwards |
| `remove_out_of_bounds` | folded into the entry kernels (flag 0 = dropped) | outside rule as in S1 |
| `CountingVoxelList::remove_underpopulated` | same name | flag `count >= threshold`, scan, compaction |
| `collideWith(list)` (`binary_search`), `collideWith(CountingVoxelList, BitVoxelList, threshold)` | `collide_with` | one binary search per entry; for counting lists only entries with `count >= threshold` count |
| `subtract`, `subtractFromCountingVoxelList` | `subtract`, `subtract_from_counting_list` | removes every entry whose id occurs in the other list |
| `collideWith(BitVoxelMap)` (`kernelCollideWithVoxelMap`) | `BitVoxelList::collide_with(map)` | **corrected in S7-5:** the list voxel and the map voxel must both be occupied (any bit except FREE; DefaultCollider), S4 only looked at the map voxel; colliding list voxels get `eBVM_COLLISION`; 32-bit counter; now with a three dimensional offset and for prob. maps, see [vulkan_gpu_voxels_examples.md](vulkan_gpu_voxels_examples.md) |
| `TemplateVoxelMap::merge(list)` | `BitVoxelMap::merge_voxel_list` | inserts the list coordinates with `eBVM_OCCUPIED`, payload ignored, no offset (CUDA rejects offsets) |
| - (viewer scans maps in `extractCubes`) | `BitVoxelList::insert_occupied_voxels(map)` | **new**: appends all occupied voxels of a map (flags, scan, gather, merge) |

## CUDA behaviour found while porting (and what the Vulkan code does)
1. `kernelInsertGlobalPointCloud` for lists has no bounds check; `remove_out_of_bounds` later filters on the coordinates, and negative points saturate to voxel 0 first (K21). Vulkan: outside points (negative, too large, NaN, inf) are dropped when the entry is created; the count is returned.
2. `collideVoxellists` with a non-zero offset logs an error and returns 0; `TemplateVoxelMap::merge` with an offset likewise. Vulkan has no offset parameters at all.
3. `CountingVoxelList::collideWith` combines the "count >= threshold" mask with the found mask only for the **returned count**; `subtractFromCountingVoxelList` goes through the same code with an all-false mask (a TODO in the CUDA source wonders about this) and removes **all** found entries regardless of the threshold. Vulkan reproduces this: the threshold only exists in `collide_with`.
4. Per-block collision counters of the list-vs-map kernel are `uint16_t` (wrap at 65 535). Vulkan counts in 32 bits.
5. `clearBitVoxelMeaning`, `insertMetaPointCloudWithSelfCollisionCheck`, `merge` of lists are "not yet supported" in CUDA for lists; not ported.
6. Counting voxels are int8: more than 127 points in one voxel wrap to negative counts (130 -> -126, 256 -> 0, the entry stays). Reproduced on purpose and tested; `remove_underpopulated(0)` therefore deletes wrapped entries.
7. Entry order inside `make_unique` is irrelevant for the result (OR / addition commute), so the parallel group reduction is deterministic.

## Limits
- At most `VoxelListBase::kMaxEntries` = 16 776 960 entries per list (`Primitives::kMaxSortElements`: 65 535 sort tile groups x 256); larger inputs throw `VulkanError`. The merge counts the old and the new entries before it removes duplicates, so a full list rejects every further insert and stays as it was; one insert of more entries than the limit is rejected by the same merge (coordinates and points arrive in chunks of 4 Mi, a meta point cloud in one piece). Tested at the limit and one above, K34. Inputs are uploaded in chunks of 4 Mi points; each chunk is merged into the list before the next one starts.
- Reference maps: every axis <= 2^24, `x*y*z` < 2^32 (ids are 32 bit). `insert_occupied_voxels` needs a map with <= 2^24 voxels (the scan limit of the primitives).
- Memory: every operation allocates new exactly sized buffers (no capacity growth); a million random points need about 0.17 s on the Intel UHD (including upload and CPU-side readbacks of two scan totals per step).
- Pipelines (14) are created per list object; a per-context kernel cache is still open (same as for maps).

## Verification (2026-10-23, Intel UHD, validation layer on, zero validation errors)
- Reference: `tests/reference/voxel_list_reference.h` (`std::map` keyed by id, `std::bitset<256>`, int8 arithmetic; written from the CUDA code, no shared layout).
- 24 tests in `tests/vulkan/test_voxel_list.cpp` (construction, empty list, known voxels, random batches vs reference, outside points incl. NaN/inf/boundary voxels, coordinate lists, meta clouds with an empty sub cloud, counting wrap-around, remove_underpopulated, small duplicate groups) and `test_voxel_list_ops.cpp` (collision list/list, counting with thresholds, subtract variants, dimension mismatches, list vs map with FREE-only voxels, map merge incl. smaller map, extraction from a map, 1 M points, two upload chunks).
- Bug found by the tests: the first version of `list_unique_merge.comp` used the exclusive-scan value as group index for non-head entries of a run (off by one); all duplicate-merging tests failed until fixed.
- **Mutation testing, 48 mutations, all killed** (runner `tools/mutation/mutate5.py`, see [mutation_testing.md](../mutation_testing.md)): group index of merged runs, XOR/OR instead of OR/ADD, missing int8 normalisation, head flags, compaction (coordinate, word and offset errors), entry kernels (outside points kept, bound checks of voxel -1 and of the upper bound for x/y/z, counting start value, meta sub cloud lookup, id formula), membership (upper bound search, threshold, nothing kept), list-vs-map collision (FREE counted, no marker), count filter, gather kernel, sort key bits, dropped count, copy of old coordinates, merge mode, dimension checks, threshold flag, pair merging, map merge meaning. One mutation **survived** first (voxel -1 accepted along y: the test only had a negative neighbour along x); points with voxel -1 along y and z were added and killed it.
- Whole suite: `test_vulkan` 80 cases pass; `ctest` in `build/verify-vk` 88 of 89 pass (K18 pre-existing); the CPU-only build (`build/verify-cpu`) is unchanged: 8 of 9.
