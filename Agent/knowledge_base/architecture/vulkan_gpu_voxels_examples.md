# List x map collisions, primitive arrays and example smoke tests (slice S7-5, 2026-11-03)

Status: **Tested, not Parity** (the CUDA behaviour is a reading of `BitVoxelList.hpp`, `TemplateVoxelList.hpp`, `VoxelListOperations.hpp`, `DefaultCollider.hpp`, `BitVector.cuhpp`, `PrimitiveArray.cpp` and `GpuVoxels.cu`; nothing ran under CUDA). Builds on [vulkan_gpu_voxels_facade.md](vulkan_gpu_voxels_facade.md). Plan: [S7 checklist](../../Analysis/2026-11-02_s7_facade_checklist.md), rows `addPrimitives` / `delPrimitives` / `modifyPrimitives` (row 21) and the slice plan entry S7-5. Code: `vulkan/shaders/list_collide.comp`, `BitVoxelList::collide_with*` in `vulkan/src/VoxelList.cpp`, `vulkan/include/gpu_voxels/vulkan/PrimitiveArray.h` / `src/PrimitiveArray.cpp`, the primitives members of `GpuVoxels`.

Reading the examples showed that the checklist was wrong in one sentence: `BitVoxelList::collideWithTypeMask` was **not** the only collision call of the examples without a Vulkan counterpart. The examples also call `collideWith(ProbVoxelMap, threshold, offset)` and `collideWithTypes(ProbVoxelMap, ...)` of a bit list, which did not exist either. All of them are now one family.

## List against map: three forms
A bit list (`BitVoxelList`) against a `BitVoxelMap` or a `ProbVoxelMap` with the reference dimensions. Entry i looks at the map voxel at its coordinates plus an `Offset3i`. All forms mark a colliding entry with `eBVM_COLLISION` (after its bits were read) and count in 32 bits (CUDA: `uint16_t` per block).

| Vulkan | CUDA | A collision is |
|---|---|---|
| `collide_with(map, offset)` / `collide_with(prob, threshold, offset)` | `collideWith` (kernel with `DefaultCollider`) | the **list voxel and the map voxel** both occupied. Bit voxel: any bit besides `eBVM_FREE` (`!noneButEmpty`). Prob. voxel: occupancy >= `ProbVoxelMap::probability_from_float(threshold)` (default 1.0 -> 127) |
| `collide_with_types(map or prob, threshold, offset)` -> `{count, types}` | `collideWithTypes` | the **map voxel** occupied, the list voxel is not looked at. Bit voxel: any bit besides FREE (threshold ignored). Prob. voxel: `float(occupancy) >= threshold` **raw** (default 1.0 means occupancy >= 1, not 127). `types` = OR of the bits of the colliding entries before the marker was added |
| `collide_with_type_mask(map or prob, mask, threshold, offset)` | `collideWithTypeMask` | as `collide_with_types`, and the list voxel shares a bit with `mask` (all 256 bits, FREE included; an empty mask collides with nothing) |

The two threshold rules are CUDA's (`DefaultCollider(float)` converts through `floatToProbability`, `isOccupied(float)` compares the int8 occupancy as a float); they are kept so that code ported from CUDA gives the same counts. Unknown prob. voxels are -128, so only a threshold of -128 or less (not NaN) lets them collide in the raw forms.

**Correction of S4.** `collide_with(BitVoxelMap)` of S4 only looked at the map voxel ("only the map voxel is inspected", reference and test said so). That was read from the wrong kernel overload: `collideWith` goes through `collisionCheckWithCollider` and the Collider kernel, which calls `DefaultCollider::collide(list voxel, map voxel)`, and that needs both voxels occupied. The difference shows for entries that only carry `eBVM_FREE` (inserted with meaning 0, or created by `merge` of a counting list): CUDA does not count them, S4 did. Fixed; reference (`tests/reference/voxel_list_reference.h`), the S4 test comment and [vulkan_voxel_lists.md](vulkan_voxel_lists.md) are corrected.

Differences to CUDA, all by design:
- **Offset:** three dimensional. An entry whose moved coordinates leave the map never collides. CUDA adds the offset to the *linear* index and bounds-checks that, so an x offset wraps into the next row and a y offset into the next slice (same decision as for the octree, K31, and the bit map's offset, [vulkan_bit_voxel_map.md](vulkan_bit_voxel_map.md)). Results are equal wherever the moved voxel stays inside the map on every axis. An offset of a whole axis or more returns 0 without a dispatch.
- Counting lists do not collide with maps (CUDA neither); `collide_with*` exist on `BitVoxelList` only. A map of another context or with other dimensions throws (CUDA: error log and `size_t(-1)`).
- Counters are 32 bit.

## Primitive arrays
`PrimitiveArray` (host only; the CUDA class copies to device memory for the viewer and nothing else reads it) stores per primitive `{x, y, z, radius}`: the centre in voxel units and half the size in voxel units, computed in float as CUDA: metric `floor(p / side) + 0.5` and `(w / side) / 2`, voxel positions `p + 0.5` and `w / 2`. Four `set_points` forms (`Point4f` metric with size, `Coord4i` voxels with size, `Point3f` metric with one diameter, `Coord3i` voxels with one diameter); `memory_usage_bytes()` = 16 per primitive.

`GpuVoxels` gets `add_primitives(type, name)` (false for a used name; names are separate from map names), `del_primitives`, four `modify_primitives`, `get_primitives(name)` (new: without a viewer the only way to look at the content; handles outlive `del_primitives`) and `visualize_primitives_array` (always false, Q6). Unknown names return false with an error log as in CUDA.

Differences: `set_points` throws `VulkanError` (and changes nothing) for a negative size or a position / size that is not finite in voxel units (NaN, infinity, overflow after the division); CUDA stores whatever it gets. The type must be sphere or cuboid (`ePRIM_INITIAL_VALUE` is an internal marker in CUDA and not offered); `getDiameter` is not offered (CUDA never sets that member). `add_primitives` before `initialize` throws like `add_map`. Positions outside the map are stored (as in CUDA).

## Examples as headless smoke tests
`tests/vulkan/test_gpu_voxels_examples.cpp` repeats the call sequences of six examples without the viewer and the endless loop. Expectations are not taken from the calls under test: the box and primitive formulas in float / double, hand-counted numbers, and the host model `tests/vulkan/list_collision_model.h` (rules of the CUDA kernels applied to the downloaded states, also used by the collision tests).

| Example | Test | Differences to the program |
|---|---|---|
| `Maps` | `ExampleTest.Maps` | 4 frames of the moving box |
| `PrimitiveArrayTest` | `ExampleTest.PrimitiveArrayTest` | none (list points, four arrays, hand-calculated values) |
| `Sandbox` | `ExampleTest.Sandbox` | the program uses `corner3` before it is set in the first frame; the boxes only touch at t = 0 .. 0.52 (octree / bit map) and 1.05 .. 2.09 (prob. map / bit map), so the frames t = 0, 0.03, 1.5, 1.53 are used instead of 50 frames |
| `Collisions` | `ExampleTest.Collisions` | 3 frames; robot cloud from `robot4cmRes.pcd`, 50 swept volume boxes |
| `VoxelListSandbox`, `VoxelListCountingSandbox` | `ExampleTest.VoxelListSandbox` | the collision calls of both are the same; masks 34, 66, 102, 63 of the comments added |
| `CountingVoxelList` | `ExampleTest.CountingVoxelList` | `subtract` / `subtractFromCountingVoxelList` take a metric offset in CUDA (the program passes zero), here none; thresholds 0 .. 3 instead of keyboard input |

Not translated: `VoxelMapSandbox` (offset collisions between voxel maps: not offered, see above), `GeometryGeneration` (`createSphereOfPoints`, `createCylinderOfPoints` are not ported, only the box), the URDF, ROS and Kinect examples (out of scope, Q6), `RobotVsEnvironment` / `SweptVolumeVsEnvironment` (robot description files and octree collisions with swept volumes: not looked at in S7-5). `ProbVoxels`, `ProbMap2DistMap(Example)`, `HeightmapLoader`, `RaycastingSensorExample`, `DistanceVoxelTest` only use calls that earlier slices test at class level (`insert_point_cloud`, sensor data, distance maps); a translation was not done, so their call sequences have **not** been run through the façade.

## Verification
- `tests/vulkan/test_list_map_collisions.cpp` (7 cases): every form, both map kinds, 15 offsets (inside, beyond a whole axis, boundary), thresholds (1.0, 0.5, 0.1, 0 for the converted rule; 1, 0.5, 0, 72, 73, 127, 128, -10, -11, -128, -1000, NaN for the raw rule), 12 masks (empty, single bits, FREE bit, bits across words, all bits) against the host model including the marked list state; the FREE-only entries (default form refuses them, the types forms do not); the offset that would wrap in CUDA; argument errors, a foreign context, an empty list, offsets of INT_MIN / INT_MAX. The S4 test of `collide_with(BitVoxelMap)` now has a reference that inspects the list voxel.
- `tests/vulkan/test_gpu_voxels_primitives.cpp` (7 cases): the four forms against hand-calculated numbers and the formula in double precision (random values, three side lengths), replacement, empty input, every refused input, the façade with the numbers of `PrimitiveArrayTest`, names, handles after `del_primitives`.
- `tests/vulkan/test_gpu_voxels_examples.cpp` (6 cases, above). `test_vulkan`: 329 of 329.
- Mutation runner `tools/mutation/mutate_s75_collide_prims.py` (the shader, the host side of the list, `PrimitiveArray.cpp`, the façade members): **85 mutations: 83 killed, 2 equivalent** (log `tools/mutation/logs/s75_collide_prims.log`). Equivalent: C25 (the mask form also collects the types, but the API returns only the count) and H17 (the host short cut for an offset of a whole axis; the shader skips those entries anyway). One mutant did not build (a removed statement left an `if` without a body) and was repeated with `(void)0;`. The tests found nothing to add: no survivor besides the two equivalent ones
- **Not verified:** CUDA parity of any count (no CUDA here; the linear-offset difference means offset results can differ at the map border by design); lists with more than 2^24 entries; the example programs themselves (only their call sequences, translated by hand).
