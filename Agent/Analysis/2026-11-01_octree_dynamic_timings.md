# Timings of the dynamic octree (insert, propagate, rebuild), 2026-11-01

- **Date:** 2026-11-01
- **Question:** What do insert, propagate, node insert and rebuild cost on the RTX 4090, what dominates, and does a long series of small inserts slow the queries down?
- **Scope / non-goals:** measured, not tuned. One run set; the minimum of 3 repetitions is printed for every operation, run-to-run differences are about 10 - 20 %. No CUDA comparison (nothing runs under CUDA), no sensor pipeline (S6f). Memory use only as far as the trees report it.
- **Status:** measured 2026-11-01, **updated after the fast insert (S6d-5) the same day** (section "After S6d-5"); code: `tests/vulkan/test_octree_timings.cpp` (disabled case `DISABLED_OctreeDynamicTimings`)

## Method
Device NVIDIA GeForce RTX 4090 (driver Vulkan 1.4), **Release build, no validation layer** (`GVL_ALLOW_NO_VALIDATION=1 VK_LOADER_LAYERS_DISABLE=VK_LAYER_KHRONOS_validation`), base tree of 1 000 000 random voxels (seed 42) in a 200^3 volume ("dense", about 12 % filled) and in a 2048^3 volume ("sparse"). Times are wall clock around the blocking call (host work, uploads, readbacks included).
Build: `cmake --build build/verify-vk --config Release --target test_vulkan` (the Visual Studio build tree holds both configurations). Command: `test_vulkan --gtest_also_run_disabled_tests --gtest_filter=VulkanTest.DISABLED_OctreeDynamicTimings`; `GVL_TIMING_VOXELS` changes the base size. A Debug build with the validation layer is about 3 times slower for small operations (insert of 1 voxel 6.5 - 9.6 ms there against 2.2 - 2.8 ms here, measured with a 20 000 voxel base) and must not be used for timings.

## Results (Release, no validation, 1 M voxel base tree)
| Operation | det dense | det sparse | prob dense | prob sparse |
|---|---|---|---|---|
| build 1 M voxels | 9.8 ms | 15.6 ms | 12.3 ms | 27.0 ms |
| tree after build | 13 MiB | 194 MiB | 22 MiB | 295 MiB |
| insert 1 voxel + propagate | 2.2 + 1.9 ms | 2.5 + 2.1 ms | 2.6 + 3.0 ms | 2.6 + 2.2 ms |
| insert 100 voxels + propagate | 2.2 + 1.8 ms | 2.9 + 2.0 ms | 2.8 + 2.3 ms | 2.7 + 2.1 ms |
| insert 10 000 voxels + propagate | 2.7 + 1.9 ms | 3.4 + 2.1 ms | 3.3 + 2.4 ms | 3.9 + 2.1 ms |
| insert 100 000 voxels + propagate | 7.8 + 1.9 ms | 8.3 + 6.0 ms | 8.6 + 2.3 ms | 8.7 + 2.5 ms |
| insert 1 000 000 voxels + propagate | 63 + 2.1 ms | 72 + 13.6 ms | 64 + 2.4 ms | 73 + 16.0 ms |
| host `std::sort` + `unique` of 1 M ids | 52 ms | 52 ms | 52 ms | 52 ms |
| `insert_nodes` level 3, 10 000 nodes, incl. propagate | 5.3 ms | 5.6 ms | 6.4 ms | 6.0 ms |
| 200 inserts of 1000 voxels with propagate: median / max | 4.6 / 18.4 ms | 5.2 / 6.7 ms | 5.7 / 8.1 ms | 5.4 / 7.9 ms |
| tree / chunks after those 200 inserts | 14 MiB / 242 | 226 MiB / 1015 | 23 MiB / 242 | 344 MiB / 1015 |
| `rebuild()` of that tree | 3.0 ms | 16.1 ms | 4.3 ms | 22.3 ms |
| chunks after the rebuild | 15 | 15 | 15 | 15 |
| `find` of 100 000 ids before -> after the rebuild | 0.89 -> 0.89 ms | 0.95 -> 0.96 ms | 0.84 -> 0.90 ms | 0.80 -> 0.82 ms |
| `extract()` before -> after | 66 -> 67 ms | 368 -> 363 ms | 67 -> 68 ms | 363 -> 365 ms |

Garbage scenario, dense cube 128^3 (2 097 152 voxels, made of uniform nodes), then 100 inserts of 1000 random **free** voxels each (every one replaces the child block of a uniform node):
| | det | prob |
|---|---|---|
| tree after the build | 4 MiB | 7 MiB |
| median time of one such insert (with propagate) | 5.6 ms | 5.9 ms |
| tree afterwards / garbage blocks / chunks | 7 MiB / 299 593 / 253 | 11 MiB / 299 593 / 253 |
| `memory_usage()` (walks the tree on the host) | 31.6 ms | 32.7 ms |
| `needs_rebuild()` with the garbage fraction on | 32.8 ms | 31.5 ms |
| `rebuild()` | 3.4 ms | 3.2 ms |
| tree after the rebuild / chunks | 2 MiB / 15 | 4 MiB / 15 |

## Findings
1. **Small inserts are latency bound (high, measured):** 1 to 10 000 voxels cost the same 2 - 4 ms and the propagate adds about 2 ms, on every workload. The GPU work for such a batch is microseconds; the cost is the number of blocking round trips (read from the code, not profiled: the walk, then per level 0 .. 13 a heads kernel plus scan with a readback of the total, then per level with new blocks heads + scan + link, then the leaf / node kernel: about 30 `submit_and_wait` per insert, plus one per level in propagate). A sensor loop that inserts every frame is limited to a few hundred inserts per second by this, not by the batch size. Candidate for later: record the 14 count passes in one batch and read all totals once.
2. **Large batches are dominated by the host sort (high, measured):** 52 ms of the 63 - 73 ms for 1 M voxels are `std::sort` + `unique` of the ids on the host (CUDA requires sorted ids too; `build` already sorts on the GPU with `Primitives`). Candidate: use the GPU sort in `valid_sorted_ids` for large batches.
3. **Many chunks do not slow the queries (high, measured here):** after 200 inserts the tree has 242 / 1015 chunks and `find` and `extract` take the same time before and after the rebuild (differences within the noise). The pointers are direct, so a chunk list only costs the host side (`check_tree`, `dump`, `memory_usage`, buffer capacity sums). The rebuild is about **memory and garbage**, not query speed.
4. **Rebuild is cheap (high, measured):** 3 - 22 ms for 14 - 344 MiB (about 15 GB/s of copied tree), i.e. less than the 5 ms of an ordinary insert times a handful. The 200 MB default of `needs_rebuild()` therefore costs one such pause per 200 MB of growth.
5. **Garbage is real and the fraction policy is expensive to evaluate (high, measured):** 100 inserts of free voxels into a dense cube leave 299 593 garbage blocks, the tree grows from 4 to 7 MiB and the rebuild brings it to **2 MiB, below the freshly built size** (the build keeps the redundant child blocks of its uniform nodes, the rebuild drops them). Finding the garbage is a host walk over a read-back snapshot: 32 ms for an 11 MiB tree, proportional to the tree size (a 300 MB tree would take about one second per call). `memory_usage()` and `needs_rebuild()` with `max_garbage_fraction` should not be called per frame; the byte limit alone is free. Candidate: count garbage on the GPU (the insert walk already counts the end nodes that own a child block, a lower bound for the replaced blocks).
6. The build is 10 - 27 ms for 1 M voxels on this card (the Intel UHD Debug figures of the checkpoint were 300 - 900 ms; not comparable, different device and configuration).

## After S6d-5 (fewer round trips, GPU sort, one-batch propagate)
Same workloads, device and build as above, run after the changes described in [vulkan_octree_dynamic.md](../knowledge_base/architecture/vulkan_octree_dynamic.md): the new blocks of all levels are counted by one kernel with one read back, numbering and linking run in one batch (two submits per insert), batches above 20 000 ids are sorted on the GPU, and propagate runs all levels in one batch while at most 131 072 ids are pending.
| Operation (before -> after) | det dense | det sparse | prob dense | prob sparse |
|---|---|---|---|---|
| insert 1 voxel | 2.2 -> 0.21 ms | 2.5 -> 0.23 ms | 2.6 -> 0.26 ms | 2.6 -> 0.28 ms |
| propagate after it | 1.9 -> 0.30 ms | 2.1 -> 0.29 ms | 3.0 -> 0.33 ms | 2.2 -> 0.38 ms |
| insert 10 000 voxels | 2.7 -> 0.67 ms | 3.4 -> 0.75 ms | 3.3 -> 0.70 ms | 3.9 -> 0.82 ms |
| insert 100 000 voxels | 7.8 -> 1.4 ms | 8.3 -> 1.5 ms | 8.6 -> 1.5 ms | 8.7 -> 1.8 ms |
| insert 1 000 000 voxels | 63 -> 9.3 ms | 72 -> 19.8 ms | 64 -> 9.3 ms | 73 -> 17.7 ms |
| propagate after the 1 M insert (exact path above the bound) | 2.1 -> 2.1 ms | 13.6 -> 13.4 ms | 2.4 -> 2.1 ms | 16.0 -> 16.2 ms |
| `insert_nodes` level 3, 10 000 nodes, incl. propagate | 5.3 -> 1.1 ms | 5.6 -> 1.0 ms | 6.4 -> 1.3 ms | 6.0 -> 1.2 ms |
| 200 inserts of 1000 voxels with propagate, median / max | 4.6 / 18.4 -> 0.57 / 1.9 ms | 5.2 / 6.7 -> 0.61 / 1.7 ms | 5.7 / 8.1 -> 0.70 / 2.1 ms | 5.4 / 7.9 -> 0.74 / 1.6 ms |
| dense cube: one insert of 1000 free voxels, median | 5.6 -> 0.66 ms | | 5.9 -> 0.72 ms | |

Unchanged: build, `rebuild()` (3.3 / 3.7 ms for the cube trees, 2 and 4 MiB afterwards); `memory_usage()` and `needs_rebuild()` with the garbage fraction were 30 - 32 ms (the host walk, finding 5) and are 0.24 ms after S6d-6 (section below).
1. **Small inserts are 10 times faster** (insert 2.2 - 2.6 ms -> 0.2 - 0.3 ms, propagate 2 - 3 ms -> 0.3 ms, a series of small inserts with propagate 5 ms -> 0.6 ms): the round trips were the cost. What is left is two submits, the read back of the counters and the propagate batch.
2. **The large batch is 7 times faster in the dense volume** (63 -> 9.3 ms): the GPU sort replaces the 52 ms of host sorting. In the sparse volume the remaining 18 - 20 ms are the many new blocks of a 1 M voxel insert (a 2048^3 volume needs about 30 M nodes), and its propagate (13 - 16 ms) takes the exact path because 1 M pending ids are above the bound of 131 072: that path still reads back a counter per level.
3. Findings 3, 4 and 6 above are unchanged; finding 5 (garbage counting is a host walk) is closed by S6d-6, see below.

## After S6d-6 (garbage counted on the GPU)
Same workload, device and build (Release, no validation, 1 M voxel base trees, `build/timings-release-garbage.log`); 100 inserts of 1000 free voxels into the 128^3 cube leave 299 593 garbage blocks (the same count as the host walk gave).
| Operation (before -> after) | deterministic | probabilistic |
|---|---|---|
| `memory_usage()` with 299 593 garbage blocks | 30.0 -> 0.24 ms | 31.1 -> 0.25 ms |
| `needs_rebuild()` with the garbage fraction | 30.1 -> 0.23 ms | 32.1 -> 0.24 ms |
4. **The garbage count now costs less than half of one insert (0.6 ms), 120 times less than before:** one batch and one read back replace the snapshot of the whole tree and the host walk, so the fraction policy can be evaluated after every insert. The cost now depends on the number of allocated blocks, not on the memory read back. Not measured: trees above 350 MiB.

## What was NOT measured
CUDA (no comparison possible), the Debug build beyond the sanity run, trees larger than 1 M voxels / 350 MiB (the 200 MB default was not reached by the series; `GVL_TIMING_VOXELS` allows it), several threads inserting, GPU time alone (no timestamp queries: wall clock includes the round trips on purpose), power or clock state of the card (not controlled; the first repetition is not discarded, the minimum of 3 is shown).

## Promoted knowledge
Findings 1 - 5 are summarised in [vulkan_octree_dynamic.md](../knowledge_base/architecture/vulkan_octree_dynamic.md#timings).
