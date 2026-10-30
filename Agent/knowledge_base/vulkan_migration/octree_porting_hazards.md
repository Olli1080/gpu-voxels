# Octree: what blocks a 1:1 Vulkan port, and latent CUDA problems

Companion of [octree_cuda_design.md](../architecture/octree_cuda_design.md). Device facts **measured 2026-10-29** with `vulkaninfoSDK` on the only device here (Intel UHD, driver API 1.3).

## Device facts that shape the design (measured)
| Fact | Value | Consequence |
|---|---|---|
| `maxStorageBufferRange` / `maxBufferSize` | 1 073 741 820 B (1 GiB) / 4 GiB | one binding holds at most 128 M 8-byte nodes; larger trees need several bindings or buffer device addresses |
| `shaderInt64` | true | Morton ids (up to 42 bit) and counters can be `uint64` in shaders |
| `shaderBufferInt64Atomics`, `shaderSharedInt64Atomics` | **false** | no 64-bit atomics: ids must be plain data, counters 32-bit |
| `bufferDeviceAddress`, `descriptorIndexing`, `runtimeDescriptorArray`, `vulkanMemoryModel` | true | possible but optional features (policy Q4: only Vulkan 1.3 + synchronization2 + shaderInt64 are required today); `bufferDeviceAddress` is enabled when present since S6a. Behaviour of bad addresses measured: [vulkan_octree_foundation.md](../architecture/vulkan_octree_foundation.md) |
| subgroup size | 8..32 (`minSubgroupSize` 8, `maxSubgroupSize` 32) | any code that assumes 32-wide warps is wrong here |
| `maxComputeSharedMemorySize` | 32 768 B | the CUDA work-item caches fit |
| `maxPerStageDescriptorStorageBuffers` | 200 | many bindings are possible but each pipeline in the runtime is limited to 16 |

## CUDA constructs and their Vulkan answer
| CUDA construct (where) | Problem | Answer |
|---|---|---|
| 48-bit device pointers in nodes, `void*` work items (`Nodes.h`/`EnvironmentNodes.h`, `NTreeData.h`) | GLSL has no pointers | node indices into a pool (child block index), or `GL_EXT_buffer_reference` (optional BDA) |
| `cudaMalloc` of new chunks during insertion, `m_allocation_list` | chunks in separate buffers cannot be indexed | one pool buffer with capacity growth (copy), or one buffer per level; host decides sizes from the counting pass (as CUDA already does) |
| 8 threads per node + `warp_reduction`/`BALLOT`/`__popc` on 32-bit warps, volatile shared memory (`kernel_common.h:86-150, 319-560`) | subgroup size 8-32; implicit lockstep is not guaranteed | one invocation per parent looping over its 8 children, or shared-memory reductions with barriers |
| persistent block stacks + host rebalancing (load balancer) | complex, GTX Titan tuned, but it **does not need forward progress guarantees** (only reads an idle counter) | portable in principle; a wavefront (breadth-first) formulation with scan/compaction removes it |
| `thrust::sort` of 64-bit keys, `thrust::reduce/scan/unique`, optional CUB radix sort | the Vulkan sort primitive is 32-bit keys / 24-bit limit 2^24 elements | extend sort to 64-bit keys (two 32-bit halves) and larger arrays |
| `__constant__` tables (`const_voxel_at_level`, `const_extract_selection`) | | push constants / small uniform buffer |
| `cudaPitchedPtr`, 3D `MapProperties` (voxel-map packing in `build` from maps) | | plain linear buffers |
| `printf`/`assert` for error reporting inside kernels | | status word in a buffer read back by the host |
| `cub::Log2<>` (compile-time) | | constexpr |
| recursion | none found in kernels (stacks) | fine (GLSL forbids recursion) |

## Latent problems in the CUDA code (inferred, not run)
1. **Warp-synchronous reductions without `__syncwarp`** (`warp_reduction`, `thread_prefix`): valid on pre-Volta lockstep warps, a data race with independent thread scheduling (compute capability >= 7.0). Not a Vulkan requirement but means the CUDA behaviour on current GPUs is itself unverified.
2. **`INVALID_VOXEL` is `ULONG_MAX`** (`DataTypes.h:53`) while ids are `uint64_t`: on Windows/MSVC `unsigned long` has 32 bits, so the "invalid id" sentinel is 0xFFFFFFFF, a valid Morton code (kernels compare ids against it, e.g. `kernel_Octree.h:1345,2217`). Port: use the 64-bit all-ones value or a separate validity flag.
3. Range checking of input points is an `assert` only (`NTree.hpp:495`); out-of-range ids silently corrupt the tree in release builds.
4. Two inconsistent probabilistic constants: tree update +100/-10 (`CommonValues.h`) versus voxel map sensor model +72/-10; occupancy threshold 10 (`THRESHOLD_OCCUPANCY`) versus 50 (`isVoxelOccupied`).
5. Collision parameters `resolution_level` and `coll_threshold` are ignored by the map and list overloads ([octree_cuda_design.md](../architecture/octree_cuda_design.md)).
6. Memory only grows between rebuilds; `rebuild()` needs a full extract of the tree into a temporary array (`INITIAL_REBUILD_BUFFER_SIZE` 4 M nodes, grows on demand).
7. **Device addresses are not small integers**: on Intel the upper 16 bits are set (canonical form), so a 48-bit pointer field must be sign extended ([K29](../known_issues.md)); the CUDA code (`assert(ptr < 2^48)` in `setChildPtr`) would reject such addresses.
8. **`getCommonLevel` differs between host and device** (`Morton.h:240-255`): the device build counts leading zeros (`__clzll`), the host fallback `std::countr_zero` counts trailing zeros, so the same ids give different levels ([K27](../known_issues.md)). The port has one definition.
9. Task counts and thread counts are tuned for one GPU via `linearApprox(blocks, nodes)`; no evidence the numbers matter elsewhere.

## Earlier attempts (from the post-mortem)
The Kompute attempt rewrote the octree as flat index-based buffers (`NTree.glsl`), used `uvec2` for 64-bit values, 4-bit digit radix sort and one sequence submit per step (377 submits in one test); its octree work was done mid-stream, was debugged through 17 console dumps (`test_octree_build_out_v2..v17`, `v2` still printing a failure) and is recorded as the highest-risk component (`attempt_history.md`, `mistakes_to_avoid.md` B4, C3); we did not re-run it. Lesson already recorded: port the octree last, index-based from the start, review before building.
