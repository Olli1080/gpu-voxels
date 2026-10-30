# Octree checkpoint — measurements (2026-10-30)

Device: Intel UHD Graphics (integrated), Vulkan 1.3, validation layer on, Debug build, one run each. Source: `[vulkan] ...` lines of `tests/vulkan/test_octree_checkpoint.cpp` and `tests/vulkan/test_octree_balancer.cpp` (see [summary.md](summary.md) for the method and what the numbers do **not** show). "Computed" columns are derived from the node counts of the real trees with assumed node sizes, they are not measurements of an implementation.

## 64-bit sort (Morton ids: 42 bit, 6 passes of 8 bit) against the 32-bit sort (32 bit, 4 passes)
| Pairs | 64-bit keys | 32-bit keys | Ratio |
|---|---|---|---|
| 4 194 304 (4 M) | 492 ms | 252 ms | 1.95 |
| 16 776 960 (15.9998 M, the sort limit) | 1 262 ms | 759 ms | 1.66 |

Times include the transfers of the blocking helpers (`sort_pairs64` / `sort_pairs`). One more element than the limit throws (tested).

## Nodes and bytes per voxel (faithful tree measured, other columns computed)
`allocated` = nodes in the allocated blocks (blocks of 8, including unknown fillers), `known` = nodes with a known state (the only ones the sorted layout stores). Build time is the GPU build of the voxel set including transfers.

| Workload | Kind | voxels | allocated nodes | known nodes | build | faithful B / B per voxel | index pool B/voxel | sorted levels B/voxel |
|---|---|---|---|---|---|---|---|---|
| dense cube 128^3 | det | 2 097 152 | 2 397 904 | 2 396 889 | 470 ms | 4 503 168 / 2.15 | 2.15 | 10.3 |
| dense cube 128^3 | prob | 2 097 152 | 2 397 904 | 2 396 889 | 483 ms | 7 803 328 / 3.72 | 3.15 | 11.4 |
| sphere shell r=300 | det | 1 133 814 | 3 887 888 | 1 619 799 | 345 ms | 11 254 352 / 9.93 | 9.93 | 12.9 |
| sphere shell r=300 | prob | 1 133 814 | 3 887 888 | 1 619 799 | 338 ms | 18 299 296 / 16.1 | 12.4 | 14.3 |
| 1 M random in 200^3 (12 %) | det | 940 070 | 6 198 528 | 1 714 885 | 308 ms | 14 218 232 / 15.1 | 15.1 | 16.4 |
| 1 M random in 200^3 (12 %) | prob | 940 070 | 6 198 528 | 1 714 885 | 356 ms | 23 853 776 / 25.4 | 20.5 | 18.2 |
| 1 M random in 2048^3 (0.01 %) | det | 999 934 | 32 555 088 | 5 069 319 | 602 ms | 204 467 864 / 204 | 204 | 45.6 |
| 1 M random in 2048^3 (0.01 %) | prob | 999 934 | 32 555 088 | 5 069 319 | 673 ms | 310 699 856 / 311 | 212 | 50.7 |
| 1 M random in 16384^3 (sparse) | det | 1 000 000 | 56 444 744 | 8 055 592 | 844 ms | 395 558 008 / 396 | 396 | 72.5 |
| 1 M random in 16384^3 (sparse) | prob | 1 000 000 | 56 444 744 | 8 055 592 | 869 ms | 597 337 008 / 597 | 404 | 80.6 |

Checks in the test: the byte count of the faithful tree equals blocks x 8 x node size per level (inner 8 / 12 B, leaf 1 / 2 B); every allocated node except the 7 unused nodes of the root block is on a path (`dump()`); `check_tree()` passes. Assumed sizes for the computed columns: index pool 8 B per inner node (also for the probabilistic tree), leaves 1 / 2 B, fillers included; sorted levels 9 B (det) / 10 B (prob) per known node.

## Frontier of tree vs tree (work items = pairs of blocks per level, CPU recursion, `min_level` 0)
| Pair | voxels in collision | peak | total | per level (L13 .. L1) |
|---|---|---|---|---|
| two sphere shells r=150, centres 100 apart (283 k voxels each) | 1 560 | 695 (L1) | 1 479 | 1 1 1 2 3 3 7 23 50 97 195 401 695 |
| two random clusters, 150 k voxels in 200^3 | 356 | 7 609 (L1) | 10 259 | 1 1 1 2 8 8 8 8 18 64 343 2 188 7 609 |
| two random sets, 150 k voxels in 4096^3 | 0 | 33 701 (L5) | 68 944 | 1 1 8 8 64 511 3 368 23 458 33 701 6 812 902 98 12 |

The deterministic and the probabilistic trees give the identical frontier in all three cases. The GPU volume equals the CPU recursion in all three cases (asserted).

## Wavefront against load balancer (default 128 groups x 700 items)
Two trees of ~1 M voxels (deterministic and probabilistic): tree vs tree 40-47 ms (wavefront) vs 82-110 ms (balancer, 12-13 kernel launches and as many redistributions); extract of a 1 M voxel tree (5.4 M nodes) 690-750 ms vs 770-880 ms. Details and mechanism: [vulkan_octree_collision.md](../../knowledge_base/architecture/vulkan_octree_collision.md#load-balancer-second-implementation).
