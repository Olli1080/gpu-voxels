# Vulkan data-parallel primitives (slice P2, 2026-10-22/23)

`Primitives` in `gpu_voxels_vulkan` (`Primitives.h`, `src/Primitives.cpp`): the Thrust replacement for what the voxel code needs so far. Status: **Tested** against the C++ standard library; not Parity-relevant (no CUDA counterpart class).

## API
| Function | Contract |
|---|---|
| `record_exclusive_scan(batch, in, out, n, keep_alive)` | `out[i] = in[0] + ... + in[i-1]` (uint32, wraps); `in != out`; n <= `kMaxElements` (2^24). Two-level: blocks of 512 elements (256 threads x 2) -> block totals -> recursive scan of the totals -> `scan_add`. |
| `record_sort_pairs(batch, keys, values, n, key_bits, keep_alive)` | Stable ascending sort of (key, value) pairs by the low `key_bits` (1..32) bits, in place. LSD radix sort, 8-bit digits, tiles of 256 elements: histogram per tile (digit-major) -> exclusive scan of the histograms -> stable scatter (rank = earlier equal digits inside the tile). At most 65 535 tiles (= 16 776 960 elements). An odd number of passes ends in the temporary buffers and is copied back. |
| `record_sort_pairs64(batch, keys, values, n, key_bits, keep_alive)` | (S6b) the same sort for 64-bit keys (2 words per key, low word first = the layout of a little-endian `uint64_t[]`), `key_bits` 1..64; separate `radix_*64` shaders, digits never straddle the two words |
| `record_iota(batch, out, n)` | `out[i] = i` |
| `exclusive_scan(vector)`, `sort_pairs(keys, values, key_bits)`, `sort_pairs64(vector<uint64_t>, values, key_bits)` | blocking wrappers (upload, record, submit, download) for tests and simple callers |

`record_*` append to a `CommandBatch` and put temporaries into a `KeepAlive` list that must outlive `submit_and_wait()`. Compaction, "unique by key", binary search and flag kernels are not generic primitives but small kernels of the list code ([vulkan_voxel_lists.md](vulkan_voxel_lists.md)): `list_compact` (copy flagged entries to `pos[i] + offset`), `list_unique_heads` + `list_unique_merge`, `list_membership`. `reduce`/`min`/`max` are not needed yet.

## Verification
- `tests/vulkan/test_primitives.cpp`, 7 cases: scan vs `std::exclusive_scan` for sizes around every block boundary (1, 2, 3, 255..257, 511..513, 1023..1025, 262 144 +- 1, 1 000 003) with small values and wrapping values; scan of 5 M elements; stable sort vs `std::stable_sort` (the original index as value makes stability observable); `key_bits` 1..32 including values that are no multiple of 8; sort of 2 M pairs; iota (including untouched trailing elements and rejected buffers); invalid arguments.
- Bug found by the tests: the first radix version ignored `key_bits` that are not a multiple of 8 (the last pass sorted by bits above `key_bits`); fixed with the `mask` push constant.
- Measured on Intel UHD (validation on, single run, includes transfers): scan of 5 M elements ~45 ms, sort of 2 M pairs ~127 ms.
- **Mutation testing, 13 mutations, all killed by the primitives tests alone** (scan block total / pair / rounds / block index, histogram mask and layout, scatter rank (two variants), 2-block second level, missing copy-back after an odd pass count, last-pass mask, pass count, iota). Two findings: a mutation that only moved the block-total write to another thread was *equivalent* (same value) and was replaced by one writing the wrong value; `record_iota` had no direct test (only the list tests covered it) and its mutation survived until `IotaWritesTheIndexOfEveryElement` was added.
- Limits / next: `kMaxElements` = 2^24 is a conservative host-side check for scan and sort; the sort is additionally bound by 65 535 work groups of 256 elements (the histogram kernel uses one group per tile), so larger sorts need a different scheme (ranges or bigger tiles). A per-context kernel cache is open; the primitives create their pipelines per `Primitives` object.
