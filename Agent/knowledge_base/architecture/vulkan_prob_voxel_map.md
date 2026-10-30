# Vulkan ProbVoxelMap and sensor data (slice S3, 2026-10-23)

Classes in `gpu_voxels_vulkan` (`ProbVoxelMap.h`): `ProbVoxelMap`, `SensorModel`, `UpdateMode`, `UpdateOptions`; plus `BitVoxelMap::collide_with(ProbVoxelMap)`, `BitVoxelMap::context()`, `eBVM_UNCERTAIN_OCC_PROB` in `vulkan_types.h`.
Status: **Tested** against an independent CPU reference; **not Parity** (no CUDA run, Q5). Not part of S3: `insertMetaPointCloudWithSelfCollisionCheck` (CUDA: "not supported" for prob maps), offset collisions, disk I/O, packing four int8 voxels per word, sensor frustum / Kinect models of the octree (`octree/SensorModel.h` belongs to S6).

## Data model
One 32-bit word per voxel holding the sign extended int8 log-odds occupancy; **-128 = unknown** (initial value), updates saturate at -127 / +127. 4 bytes per voxel (CUDA: 1) - packing needs CAS loops on sub-word fields and is an open optimisation. Update rule (`ProbabilisticVoxel::updateOccupancy`): unknown voxel takes the increment, known voxel adds and clamps. Increments by meaning (`ProbabilisticVoxel::insert`): FREE -> `model.free_update` (-10), OCCUPIED/COLLISION -> `model.occupied_update` (+72), UNKNOWN/UNDEFINED -> set unknown, 4..254 -> `meaning - 129`.

## User decisions (2026-10-23)
1. Concurrency semantics should be selectable "like `std::atomic` memory orders" -> `UpdateMode` (below).
2. Sensor model = struct with the CUDA defaults as an optional argument (`SensorModel{-10, 72}`, values must lie in [-127, 127]).
3. Ray traversal: replicate the CUDA walk. 4. Outside rule: keep CUDA's "drop the measurement if hit or sensor voxel is outside" but process all points (CUDA bug, see below).

## UpdateMode
| Mode | Result | Cost |
|---|---|---|
| `Atomic` | every update is one CAS loop (no lost updates); with saturation the result is that of SOME serial order (tests check this against all permutations) | O(updates) |
| `Accumulate` (default) | per voxel the sum of all updates of the call, applied once: `clamp(old + sum)`, unknown counts as 0; deterministic; equals the sequential result unless a voxel saturates inside the call | O(updates) + one pass over the map; 5 extra bytes per voxel (allocated on first use, reset by the apply pass) |
| `Sequential` | exactly the single-threaded CUDA result: events in measurement order (free voxels in ray order, then the hit), saturating after each | count -> scan -> write events -> radix sort by voxel (stable) -> one thread per voxel replays its run; chunks of at most `UpdateOptions::max_sequential_events` (default 2^22) updates; 100 k rays into 128^3: 444 ms vs 47 / 67 ms (Atomic / Accumulate) on the Intel UHD |
"Racy like CUDA" (plain read-modify-write) is deliberately not offered: lost updates, nondeterministic, untestable.

## What was ported
| CUDA | Vulkan | Notes |
|---|---|---|
| `insertPointCloud` / `insertCoordinateList` / `insertMetaPointCloud` (TemplateVoxelMap, `voxel->insert(meaning)`) | `insert_point_cloud`, `insert_coordinate_list`, `insert_meta_point_cloud` | meta clouds: sub clouds in order; neighbours with equal updates are applied as one batch (a pure speed-up: equal updates have the same sign, so the result cannot differ) |
| `insertSensorData` (+ `RayCaster`, `DummyRayCaster`, `cut_real_robot`) | `insert_sensor_data(points, pose, raycasting, robot_map, robot_bit, options)` | returns `{outside, cut_by_robot}`; robot map = `BitVoxelMap` with the same dimensions |
| `clearBitVoxelMeaning` | `clear_bit_voxel_meaning` | only `eBVM_OCCUPIED` (clears the map); other meanings throw (CUDA logs an error) |
| `collideWith(BitVoxelMap / ProbVoxelMap, threshold)` (`DefaultCollider(float)`) | `collide_with(...)` | threshold -> `probability_from_float` (default 1.0 -> 127); side effect: colliding voxels of *this* get `insert(eBVM_COLLISION)` = one update with `model.occupied_update` |
| `BitVoxelMap::collideWith(ProbVoxelMap)` | `BitVoxelMap::collide_with(ProbVoxelMap, threshold)` | marks the bit map's `eBVM_COLLISION` bit |
| `floatToProbability` / `probabilityToFloat` | `probability_from_float` / `probability_to_float` | NaN -> 1.0 -> 127 as in CUDA (`std::min(1, NaN)`) |

## CUDA behaviour found while porting (and what the Vulkan code does)
1. `ProbabilisticVoxel::updateOccupancy` is a non-atomic read-modify-write called by thousands of threads (many rays share voxels): updates are lost and the result is nondeterministic. Vulkan: the three modes above.
2. `kernelInsertSensorData` loops `i < voxelmap_size && i < num_points`: only the first `min(voxels, points)` measurements are processed. Vulkan processes all (tested with 600 points in a 64 voxel map). Reported as K24.
3. The hit voxel gets a free update from the ray walk (the walk includes both end points) and then the occupied update: net +62 from unknown. The sensor voxel itself is updated as free too. Without ray casting CUDA still drops measurements whose *sensor* voxel is outside.
4. The ray walk (`RayCaster::rayCast`) is a 6-connected Bresenham-like state machine without per-axis termination. Checked exhaustively for all pairs in 0..6^3 and 3000 random pairs up to 400 (test `RayWalk.ReferenceWalkHasTheClaimedGeometricProperties`) and earlier by a script: it visits exactly `1 + dx + dy + dz` voxels, starts at `from`, ends at `to`, moves one axis by one voxel towards the target and never leaves the box of its end points. The shader keeps a bounds guard anyway.
5. `insert(eBVM_COLLISION)` on a probabilistic voxel adds +72 (the CUDA comment says "sets MAX_PROBABILITY" - stale). Ported as the code behaves.
6. `isOccupied(threshold)` of the probabilistic voxel compares the raw int8 with the float threshold, while the collider converts the threshold with `floatToProbability`; only the collider path is ported.

## Limits
- One shader invocation walks a whole ray: a single ray of millions of voxels can exceed the GPU watchdog. Observed: a 16.7 M voxel ray (one measurement across a 2^24 x 1 x 1 map) made the Intel UHD report `ErrorDeviceLost` (K25); rays up to the usual map sizes (hundreds to a few thousand voxels) are no problem.
- Sequential mode rejects a single measurement with more than `max_sequential_events` updates; 2^24 is the scan limit of the primitives.
- Map memory: `voxels * 4` bytes (+ `voxels * 4 + voxels / 8` for Accumulate); the buffer must fit `maxStorageBufferRange`.
- Pipelines (6 + 1 per bit map) are created per map object (per-context kernel cache still open).

## Verification (2026-10-23, Intel UHD, validation layer on, zero validation errors)
- Reference: `tests/reference/prob_voxel_map_reference.h` - sparse int map; updates generated as explicit event lists (ray walk translated from the CUDA code, free of the GPU code), folded sequentially or accumulated; `serializations()` enumerates all serial orders of a voxel's updates.
- 29 tests: `test_prob_voxel_map.cpp` (16: geometry, conversions, every meaning in every mode, invalid models/meanings, Sequential exact with saturation, Accumulate with resets between calls, Accumulate == Sequential without saturation, no lost updates under contention (20 000 updates on 200 voxels, all modes), Atomic result is a serial order, coordinates, outside points incl. NaN/inf, meta clouds, collisions with thresholds and custom marker, bit map vs prob map, mismatching maps) and `test_prob_sensor.cpp` (13: ray walk properties, rays vs reference in every mode, random scans, Atomic where no order effect is possible, no ray casting, dropped measurements, robot cut, more points than voxels, 80 000 rays split into two default chunks, small chunk limits, over-long rays, custom model, performance report).
- Bug found by the first test run: none in the shaders; the device-lost finding (K25) came from a too ambitious test and led to the configurable chunk limit.
- **Mutation testing, 63 mutations (shaders and C++): 61 killed, 2 equivalent.** Covered: update rule (unknown base, both clamps), bounds/NaN rules, robot bit word/inversion/cut count, origin requirement, every line of the ray walk (length, branch conditions, error updates, free vs hit increment, lost hit), assign-unknown, accumulate (store instead of sum, missing touched flag, never selected, base, no reset, no clamp), plain read-modify-write instead of the CAS loop, fold (first update only, every entry replays, order reversed), event writer (base, sign, id), event counter, collision shaders (threshold, FREE bit, marker, fixed increment), conversions, meanings, flags in pushes, robot dimension check, clear semantics, download sign, sort key bits, chunk base / first measurement, chunk limit (total, single ray), model validation, bit-map threshold conversion. The two equivalent mutants cannot change any result: (a) not grouping neighbouring sub clouds with equal updates (same-sign increments: `clamp(clamp(a+s1)+s2) == clamp(a+s1+s2)`), (b) ignoring the chunk limit when searching the chunk end (chunk size is invisible in the result; the limit for single rays and the chunk-start/base arithmetic are killed by other mutants). Harness: ~20 s per mutation (5 s build + 6-14 s tests); the 7 s 80 000-ray test is excluded from the mutation filter because the small-limit chunk test covers the same code.
- Whole suite: `test_vulkan` 109 cases pass; `ctest` in `build/verify-vk` 117 of 118 pass (K18 pre-existing), CPU-only build unchanged (8 of 9).
