# Octree façade and storage (slice S6e, 2026-11-01)

Status: **Tested, not Parity** (the CUDA behaviour is inferred from reading `GvlNTree.hpp` / `NTree.hpp`, nothing ran under CUDA). Adds to `Octree` ([vulkan_octree_build.md](vulkan_octree_build.md), [vulkan_octree_dynamic.md](vulkan_octree_dynamic.md)) what CUDA's `GvlNTree` offers on top of `NTree`: the point cloud inserts and the storage of a tree behind an interface (decision Q22). Not part of this slice: the sensor path with free space (S6f, on request only), a reader for the CUDA file format, 64 bit ids in `VoxelList`.

## API
| Call | Contract |
|---|---|
| `Octree(context, kind, voxel_side_length = 0)`, `voxel_side_length()`, `set_voxel_side_length(m)` | CUDA `GvlNTree(voxel_side_length, map_type)` (CUDA keeps whole millimetres). Metres; 0 = not set, then the point cloud inserts throw. Negative, NaN and infinite values throw. The tree itself does not depend on it (ids and coordinates work without) |
| `insert_point_cloud(span<Point3f>, meaning = eBVM_OCCUPIED, propagate_now = true)` | CUDA `insertPointCloud`. The voxel of a point is `floor(p / voxel_side_length)` (float, like the other Vulkan maps), computed on the GPU. Points whose voxel is outside the 2^14 grid (negative, too far, NaN, inf) are not inserted: they are counted in the return value and in `dropped_count()`. Then the same as `insert_ids` (hard insert, duplicates merged, one batch). More than 2^24 points throw before anything happens |
| `insert_coordinate_list(span<Coord3ui>, meaning, propagate_now)` | CUDA `insertCoordinateList`; coordinates >= 2^14 are counted and dropped |
| `insert_meta_point_cloud(MetaPointCloud, meaning, propagate_now)` | CUDA `insertMetaPointCloud`: all sub clouds with one meaning, the points never leave the GPU (the converting kernel reads the cloud's buffer) |
| `insert_meta_point_cloud(MetaPointCloud, span<uint8_t> meanings, propagate_now)` | one meaning per sub cloud (CUDA warns and inserts everything with the first). Runs of sub clouds with the same meaning are one batch (duplicates merged), every run is propagated before the next is inserted, so the result is that of one call per run; `propagate_now` concerns the end. Checked before anything is inserted |
| meanings | `eBVM_OCCUPIED` inserts occupied, `eBVM_FREE` free voxels (CUDA: only occupied; `eBVM_FREE` is an addition because `insert` has the free datum); every other meaning throws, because the tree cannot store it |
| `OctreeStorage::write(tree, ostream)`, `read(tree, istream)`, `write_file`, `read_file` | the interface over `dump()` / `load_nodes()`. `write` propagates pending updates first (the stored statuses must be consistent). `read` replaces the tree and restores the voxel side length; it throws for data that is not a valid tree of the tree's kind and then leaves the tree unchanged. The file variants throw where CUDA's `writeToDisk` / `readFromDisk` log and return false |
| `OctreeNativeStorage` | the first implementation, the Vulkan native format below |

## Native format (version 1, little endian)
| Part | Bytes | Content |
|---|---|---|
| header | 32 | magic `GVLOCT\0\1` (8), version u32 = 1, kind u32 (0 deterministic, 1 probabilistic), voxel side length f32 (0 = not set), reserved u32 = 0, node count u64 |
| node records | 16 each | id u64, level u8, status u8, occupancy i8, has_children u8 (0 / 1), reserved u32 = 0; in the order of `Octree::dump()` (by level, then id: leaves first, root last) |
| trailer | 8 | FNV-1a (64 bit) over header and all records |

It stores exactly the nodes reachable from the root: complete blocks, with status, occupancy and COLLISION bits, so a restored tree equals the original in `dump()`, `extract()`, `find()` and collisions; cut subtrees (garbage) are not stored, flags are clear and `voxel_count()` is 0 after a read. The reader reads in batches of 65 536 records (memory is bounded by the data actually present, not by the stored count), checks magic, version, kind, reserved words, the has_children byte, the side length and the checksum, and only then hands the nodes to `load_nodes`, which checks the structure (all 8 nodes of a block, a block per node with children, a parent per block) before the old tree is touched. Bytes after the trailer are not read.

## Differences from CUDA (inferred from `GvlNTree.hpp`, `NTree.hpp`)
| CUDA | Here |
|---|---|
| `kernel_toVoxels`: `static_cast<uint32_t>(p / side)`: negative values are undefined behaviour in C++ and saturate to voxel 0 on the device (inferred), NaN gives 0, values >= 2^14 hit an `assert` in `build` only | `floor`, everything outside the grid is dropped and counted, like `BitVoxelMap` and `DistanceVoxelMap` |
| `insertPointCloud`, `insertCoordinateList`: another meaning than occupied logs an error and does nothing; `insertMetaPointCloud` logs the error and then **inserts anyway** (no `else`, `GvlNTree.hpp:389`) | free is accepted, everything else throws |
| `insertMetaPointCloud(cloud, meanings)`: "not implemented", warns, uses the first meaning | implemented per sub cloud |
| `insertVoxelData`: an empty tree (`m_has_data` false) is filled by `build`, otherwise sort + unique + `insertVoxel` + `propagate` | always the insert path; into an empty tree it makes the same nodes as the build except the STATIC / DYNAMIC bits (build: STATIC, insert: DYNAMIC) |
| `writeToDisk` / `readFromDisk` open the file in **text mode** and write binary data (`serialize(out)` defaults to `bin_mode`), which corrupts the file on Windows | binary mode |
| `serialize`: the cubes of an extraction restricted by `m_extract_status_selection` plus `numBlocks`, `numThreadsPerBlock`, resolution; `deserialize` rebuilds via `internal_rebuild` (the semantics of `rebuild`, a restricted selection loses nodes) | the whole live tree; a CUDA-format reader can be added behind `OctreeStorage` without touching the tree |
| `merge` (both overloads), `insertMetaPointCloudWithSelfCollisionCheck` (logs, returns true), `collideWithBitcheck`: log-only stubs ("not (yet) supported") | not offered: an API that always fails is not an API. `merge` of a voxel list into an octree would be `insert_ids` / `insert_coordinate_list` of its entries |
| `clearBitVoxelMeaning(m)`: only occupied (clears the tree), others log an error | `clear()` |
| `getDimensions()`: 2^14 voxels per axis (the metric variant has a typo: `dim.z()` for y) | `Octree::kCoordinateLimit` |
| `insertPointCloudWithFreespaceCalculation` (Kinect pipeline, ray casting) | S6f, only on request |

## Verification
- `test_octree_facade.cpp` (12 cases, both kinds, validation layer on): the point cloud insert equals `insert_ids` of a twin tree fed with ids this file computes on the host (side lengths 0.05, 1.0, 0.37; 1 to 60 000 points; duplicates; about a third of the points repeated; negative, too far, NaN, +-inf points; a second cloud over the same voxels), `eBVM_FREE`, `propagate_now = false`, rejected meanings (tree unchanged), empty and all-outside clouds, the side length rules, coordinate lists (also free), meta point clouds (all meanings equal, a meaning per sub cloud with empty sub clouds inside a run, errors checked first, a cloud without points), the 2^24 point limit from a span and from a meta cloud (and exactly the limit), storage round trips of an eventful tree (garbage from cut uniform nodes, higher level nodes, COLLISION bits, pending updates, several chunks; `dump`, `extract`, 2000 `find` probes, `collide_with_ids`, further inserts, a hard node insert and `rebuild` on both trees stay equal), the empty tree, the bytes decoded in the test without the library, every cut and every damaged byte of a stream and forged streams with a valid checksum (target unchanged each time), a tree of more than three read batches, files.
- Mutation check: [tools/mutation/mutate_s6e_facade.py](../../../tools/mutation/mutate_s6e_facade.py) (logs in `tools/mutation/logs`, GPU assisted validation), 51 mutations of the points shader, the insert code and the storage code: 46 killed, 5 equivalent (G14 / G15: the point limit is refused by a second check / the sort primitive as well, G16: without the pending ids propagate takes the exact path, M6: the data ends before a huge node count is read, M19: batch size of the reader). Five survivors of the first run were test gaps (an x beyond the grid, the last voxel of the grid and so the sentinel order, runs of meanings with `propagate_now = false`, a forged magic, a forged has_children byte) and were closed. Details: [mutation_testing.md](../mutation_testing.md).

## Decisions
`eBVM_FREE` is accepted next to `eBVM_OCCUPIED` (user decision 2026-11-01): the tree can store free voxels and the sensor path (S6f) needs them; CUDA refuses it.
