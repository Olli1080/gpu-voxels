# Vulkan BitVoxelMap (slice S1, 2026-10-22)

`gpu_voxels::vulkan::BitVoxelMap` in `gpu_voxels_vulkan` (`include/gpu_voxels/vulkan/BitVoxelMap.h`, `src/BitVoxelMap.cpp`, kernels in `vulkan/shaders/*.comp`). A dense map of `BitVoxel<256>` (32 bytes, 8 words per voxel, bit i = meaning i; same byte layout as the CUDA `BitVector<256>` on little-endian hosts) in ONE device-local storage buffer, linear index `z*dx*dy + y*dx + x`.
Status: **Tested** against an independent CPU reference; **not Parity** (no CUDA run possible here, Q5) and not yet wired into `GpuVoxels`/`GpuVoxelsMap` (own types `Point3f`, `Coord3ui`, `Dim3ui`, own `BitVoxelMeaning` enum mirror).

## Scope of S1a
| Function | CUDA counterpart | Notes |
|---|---|---|
| constructor / `clear_map()` | `TemplateVoxelMap<BitVectorVoxel>::clearMap` | zeroes all words (state "eBVM_FREE", not UNKNOWN - as in CUDA) via `vkCmdFillBuffer` |
| `clear_bit_voxel_meaning(m)` | `kernelClearVoxelMap(..., bit_index)` | one thread per voxel |
| `insert_point_cloud(points, m)` | `kernelInsertGlobalPointCloud` | returns the number of points outside (CUDA: only a warning flag + log) |
| `insert_coordinate_list(coords, m)` | `kernelInsertCoordinateTuples` | same |
| `collide_with(other)` | `collideWith` = `collisionCheckWithCounterRelativeTransform` / `kernelCollideVoxelMapsDebug` with `DefaultCollider` | counts colliding voxels, sets `eBVM_COLLISION` in *this* |
| `collide_with_types(other)` | `collideWithTypes` = `collisionCheckBitvector` with `SVCollider` (window 0, offset 0) | colliding bits of a voxel pair = `(a & b)` without bits 0..3; count + OR of all colliding bits; marks `eBVM_COLLISION` in *this*; the CUDA "last colliding voxel per thread" quirk is harmless because the grid is large enough for one voxel per thread |
| `download_words(first, count)` | - | test/inspection access |
**Not in S1** (open): collision with offset (semantics undefined, K21), SVCollider window size > 0 and `sv_offset` (not reachable through the CUDA public API), disk I/O, dilated/closed/eroded insertion, `ProbVoxelMap` as collision partner, sensor insertion. **`merge`:** in CUDA `TemplateVoxelMap::merge` only accepts voxel *lists* (it inserts their coordinates as `eBVM_OCCUPIED`, offset must be zero) and rejects maps; there is no map-to-map merge to port. The list case becomes `insert_coordinate_list` once `VoxelList` exists (slice S4). Meta cloud insertion and self collision: [vulkan_meta_cloud_and_robot.md](vulkan_meta_cloud_and_robot.md).

## Deviations from CUDA (decisions to confirm: Q16)
1. **Negative coordinates:** CUDA computes `static_cast<uint32_t>(floor(p/side))`; on the device a negative float converts with saturation to 0, so a point at x = -0.3 voxels is inserted into voxel x = 0 (inferred from PTX conversion rules, not observed here). Vulkan version treats negative, too large and NaN coordinates as **outside** and counts them.
2. **Collision counter:** CUDA accumulates per-block `uint16_t` counts (wraps at 65 535 colliding voxels per block); here a 32-bit total.
3. **Float division:** `floor(p / side)` uses the device's float division (Vulkan allows a few ULP error). Points within ~1 ULP of a voxel boundary may land in the neighbouring voxel on different hardware. Tests avoid boundaries (fractions 0.1..0.9 of a voxel); a power-of-two voxel size is exact.
4. **Self collision** (`a.collide_with(a)`) is rejected (aliasing); dimensions must be equal.
5. Limits: whole map in one buffer => `dimensions.voxel_count() * 32 <= maxStorageBufferRange` (1 GiB on the test GPU = 33 M voxels, e.g. 256x256x512); each axis <= 2^24.
CUDA behaviour that is **not** reproduced because it looks like a bug: `collideWith(..., offset)` shifts the base pointer by a *linear* signed index and then iterates over the full size (reads out of bounds for positive offsets; wraps across rows) - an offset variant needs a defined meaning first.

## Verification (2026-10-22, Intel UHD, validation layer on, zero validation errors)
- Reference: `tests/reference/bit_voxel_map_reference.h` - sparse `unordered_map<index, std::bitset<256>>`, written from the CUDA sources, shares no code or layout with the implementation.
- 16 tests (`tests/vulkan/test_bit_voxel_map.cpp`, 3 of them for `collide_with_types`): geometry/empty, invalid construction, known voxels, meanings at word boundaries (0, 1, 31, 32, 63, 64, 127, 128, 224, 255), 150 k random points with 8 invalid ones (inf, NaN, negative, too large) for voxel sizes 0.0625 and 0.1 on a 37x29x41 map, 1 M atomic-OR contention on one voxel, coordinate list incl. out-of-range, empty inputs, clear / clear-meaning, collision incl. FREE-bit-only voxels and stability on repeat, argument errors, and a **256^3 map (512 MiB) x 2** with full comparison in 1 M-voxel slices (collide 45.9 ms, single measurement).
- **Mutation testing** of the shaders (script kept in the session scratchpad, results here): 8 mutations - FREE bit counted as occupied, `round` instead of `floor`, accepting x >= -1, clearing the whole word, wrong word index, swapped y/z in the coordinate index, collision bit not written, collision counted twice - **all killed** by at least one test; 5 further mutations of the types collider (low bits not masked, types not aggregated, wrong result word, collision bit missing, count missing) were killed as well (the first version of mutation M3 was invalid, `> -1.0` equals the correct behaviour for floor results; corrected to `>= -1.0` and killed).

## Implementation notes
- Kernels are grid-stride loops (64 threads), counters are `atomicAdd` on a scratch buffer (`collide`: one `atomicAdd` per work group via shared memory); the batch is `fill(counter) -> dispatch -> submit -> download(counter)`.
- Shaders are compiled with `glslangValidator` at build time and **embedded** into the library (`gvl_embed_shaders`, `cmake/EmbedSpirv.cmake`); `src/Shaders.h` declares the arrays.
- Each `BitVoxelMap` creates its 4 pipelines (shared per-Context kernel cache is a later optimisation); point clouds are uploaded through a temporary device buffer per call (<= 4 M items per chunk).
