# Octree foundation (slice S6a, 2026-10-29)

Status: **Tested, not Parity.** Prepares the pointer-faithful octree port decided in [Q19-Q22](../../project_state/open_questions.md): buffer device addresses in the runtime, Morton codes on host and GPU, and the independent CPU reference of the node semantics. No octree class exists yet (next: S6b). Background: [octree_cuda_design.md](octree_cuda_design.md), [slice plan](../../Analysis/2026-10-29_octree_design_review/slice_plan.md).

## What was added
| Item | Where | Contract |
|---|---|---|
| Buffer device addresses | `Context`, `DeviceBuffer` (`vulkan/`) | `DeviceInfo::buffer_device_address` is true when the device has `bufferDeviceAddress`; the feature is enabled then (optional, nothing else depends on it), the VMA allocator gets `BUFFER_DEVICE_ADDRESS`, every buffer gets `eShaderDeviceAddress`. `DeviceBuffer::device_address()` returns the address of the first byte (stable across moves, 0 after a move) and **throws** `VulkanError` when the feature is missing. `ContextCreateInfo::enable_buffer_device_address = false` switches it off (used by the fallback test). |
| Morton codes, host | `include/gpu_voxels/vulkan/Morton.h`, namespace `gpu_voxels::vulkan::morton` | `code60`/`decode60` (20 bits per axis, x = bit 0, y = bit 1, z = bit 2 of each triple, **higher input bits are ignored**), `z_order_prefix(id, level)` (= `id >> 3(level+1)`), `z_order_node_id(id, level)` (= `(id >> 3 level) & 7`), `common_level(a, b)` (levels to climb until two paths meet, 0 for equal ids). |
| Morton codes, GLSL | `shaders/morton.glsl` | Same functions on `uint64_t` (`GL_ARB_gpu_shader_int64`; the includer enables the extension). Bit-identical to the host header. |
| Octree reference | `tests/reference/octree_reference.h` | `OctreeReference(Kind, voxels)` builds the canonical post-build/propagate tree recursively; `dump()` = all nodes `(level, id, status, occupancy, has_children)` ordered by (level, id), `find(id)`, `node_count([level])`, `dropped()`. The rules are listed in the header comment (all **inferred** from CUDA code reading). |
| Test shaders | `tests/vulkan/shaders/{bda_write,bda_chase,bda_read,morton_check}.comp` | write through an unbound address, follow pointers across buffers, read one word (fault experiments), exercise `morton.glsl` |

## Differences from the CUDA Morton code
- Inputs above 20 bits per axis are ignored on host and GPU (tested with random 32-bit values). By reading, CUDA's `MORTON_TRAFO` behaves the same way (low part masked with 1023, higher bits fall off the masks of `morton_code`); the helper `spread10` masks itself so that it is safe to call on its own.
- `getCommonLevel` exists only for the device and uses `__clzll`; the host fallback uses `countr_zero` (trailing zeros), which is a different function ([K27](../known_issues.md)). `common_level` here is one definition for both sides, tested against a loop that shifts both ids until they are equal.

## Reference semantics in short (inferred)
Levels 0..14 (root = 14, coordinates < 2^14, others are dropped and counted); only nodes whose box holds an occupied voxel own a child block of 8 ("on-path"); the other siblings are childless fillers. Deterministic: occupied leaf `OCCUPIED|STATIC` (68), filler `UNKNOWN|STATIC` (66, `+LAST_LEVEL` at level 1 = 82), on-path node = `getNewStatus(OR, AND)` with STATIC OR-ed (single voxel: level 1 = 94, levels 2..14 = 78, 113 nodes; full 2x2x2 block: level 1 = 84 without PART). Probabilistic: occupied leaf occupancy 127, filler -128, on-path occupancy = max of the children, PART unless all children are equal and none is PART, status always `UNKNOWN|STATIC` (+`LAST_LEVEL` at level 1). Empty tree = root alone. The recorded values come from `Nodes.h` `getNewStatus` and `kernel_common.h` (`_bottomUpUpdateProb`, `_updateStatus`); the leaf/filler status values of the probabilistic tree are the least certain part. CUDA golden data (Q5) must confirm them.

## Measured behaviour of this driver (Intel UHD, `DISABLED_` experiments in `test_device_address_faults.cpp`, one process each)
| Access through a device address | Result |
|---|---|
| 64 bytes past the end of a 64-byte buffer | completes silently; the data read is that of the neighbouring allocation (no error, no validation message) |
| 64 MiB past the end | completes silently (value 0) |
| address 0 | **device lost** (`ErrorDeviceLost`); the shared context is unusable afterwards |
| a freed buffer's address | completes silently (the memory was reused by the next allocation) |
Consequence: with the pointer-faithful route only host-side checks (`check_tree`, registered chunks) can find wrong child pointers; the validation layer does not see them. GPU-assisted validation of buffer references was **not** tried.

## Tests
- `test_octree_reference` (CPU, no device): 16 cases - a `getNewStatus` table (pure states, mixed states, STATIC/DYNAMIC OR, LAST_LEVEL, bits that must not be inherited), host Morton against a bit-loop reference (edges, 20 000 random values, bit positions, masking, spread/compact), prefix/node-index against arithmetic (division), common level against the shift loop incl. the 32/33-bit word boundary; the reference against hand-derived trees (single voxel 113 nodes, full 2x2x2 / 4x4x4 blocks, two voxels meeting at the root, far corner, `find` of uncovered voxels, empty / dropped / duplicate input, probabilistic single voxel and full block) and against a brute-force oracle that counts voxels per node box (random sparse / clustered / corner sets and holey cubes, both kinds).
- `test_vulkan` (+6 cases): address stable / distinct / non-overlapping / moved, shader writes at offset addresses (sentinel guards), shader follows 12 pointer chains over 5 buffers incl. a self loop and an empty chain, feature switched off (second context: `device_address()` throws), `morton.glsl` against host and reference on 1 728 boundary cases + 52 000 random cases.
- Mutation check: 83 mutations (24 host Morton, 34 octree reference, 16 GLSL Morton, 9 device address runtime), all killed. The first pass found three redundant masks in `code60`/`decode60` (equivalent mutants; removed, `spread10`/`compact10` mask themselves), a redundant `address_ = 0` in `DeviceBuffer::release()` (removed), a missing direct test of `new_status` (table test added), a missing moved-from check after the move constructor, and a redundant initial value of the probabilistic max/min (now taken from the first child).
