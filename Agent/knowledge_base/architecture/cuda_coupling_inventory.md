# CUDA / Thrust Coupling Inventory (measured 2026-10-22 on `master`)

Counts come from `grep` over `packages/` (excluding `icl_core` for LOC). They include declarations and uses, so treat as orders of magnitude.

## Size
| Kind | LOC |
|---|---|
| `.cu` | 11.5 k |
| `.cuh` + `.cuhpp` | 2.8 k + 0.5 k |
| `.h` / `.hpp` | 21.7 k / 11.0 k |
| `.cpp` | 17.4 k |
| `icl_core` (GPU-agnostic) | 12.2 k |
Files: 701 under `packages/` (gpu_voxels 469, icl_core 116, gpu_visualization 61, examples 54); repo has 736 tracked files, 135 of them `.binvox` models.

## Kernels
- `__global__` occurrences: **189** (declarations + definitions); `<<<…>>>` launches: **93**. Real distinct kernels ≈ 100 (estimate).
- Distribution (occurrences): octree `kernel_Octree.h` 29 · voxelmap `VoxelMapOperations.{h,hpp,cu}` 23/15/13 · voxellist `VoxelListOperations.{h,hpp,cu}` 21/18/4 · octree `kernel_PointCloud.{h,cu}` 13/12 · PBA `VoxelMapOperationsPBA.{h,hpp}` 8/9 · visualizer `VoxelMapVisualizerOperations` 6/6 · `MetaPointCloudOperations` 3 · octree `LoadBalance.cuh` 3 · `DistanceMapConverter.cu` 1.
- Streams: only 2 `cudaStreamCreate` uses; effectively everything runs on the default stream with `cudaDeviceSynchronize` after launches (**294** calls).

## CUDA runtime API (top)
`cudaDeviceSynchronize` 294 · `cudaMemcpy` 82 (+ kind constants) · `cudaMalloc` 62 · `cudaFree` 45 · `cudaMemset` 24 · `cudaIpcMemHandle*` ≈ 56 (visualizer path) · `cudaGraphics*` ≈ 34 (GL interop) · `cudaEvent*` ≈ 25 (timing) · `cudaMemcpyToSymbol` 5 (constant memory) · `cudaDeviceSetCacheConfig` 4 · `cudaProfiler*` 7 · texture objects: 2 mentions (`VoxelMapOperationsPBA.hpp`, `DistanceVoxelMap.cu`).

## Device-code features that constrain a port
| Feature | Where | Vulkan/GLSL consequence |
|---|---|---|
| `__shared__` memory | 8 files (octree kernels, load balancer, voxellist/voxelmap ops, PBA) | `shared` vars; size must be compile-time constant or spec constant |
| Atomics: `atomicAdd` 30, `atomicOr/And` 3 each, `atomicInc` 2, `atomicMin/Max` 1 each | voxel maps/lists, octree | Core atomics on `uint/int` are fine; **float atomicAdd needs `VK_EXT_shader_atomic_float`**; 64-bit atomics need `shaderInt64`+`shaderBufferInt64Atomics` |
| Warp/bit intrinsics (`__shfl*`, `__ballot`, `__popc`, `__clz`, `__ffs`, `__threadfence`, `warpSize`) | octree kernels + load balancer, `Morton.h`, `cuda_datatypes.hpp` | `GL_KHR_shader_subgroup_*` (subgroup size is *not* 32 everywhere: Intel 8/16/32, AMD 32/64) or rewrite without warp assumptions; `bitCount`, `findMSB/LSB` exist |
| 64-bit integers (Morton codes up to 60 bit, 48-bit node pointers) | octree | `GL_ARB_gpu_shader_int64` / `uint64_t` need `shaderInt64`; earlier attempt used `uvec2` pairs and index-based nodes |
| Raw pointers, pointer arithmetic, device-side `new`-like pools | octree node pools, `TemplateVoxelMap` kernels | No pointers in GLSL: use buffer + index (or buffer-device-address, `VK_KHR_buffer_device_address`) |
| C++ templates on Voxel/Collider/functor types in kernels | `TemplateVoxelMap.hpp`, `VoxelMapOperations.hpp` | Needs one shader per instantiation or code generation (macros / Slang generics / HLSL templates) |
| Dynamic block/grid sizing, 1024 threads/block, 65535 blocks (`cMAX_THREADS_PER_BLOCK`, `cMAX_NR_OF_BLOCKS`) | everywhere | Vulkan limits: `maxComputeWorkGroupInvocations` (≥128 guaranteed; 1024 typical), `maxComputeWorkGroupCount` (≥65535 per dim) — query at runtime |
| `__constant__` / `cudaMemcpyToSymbol` | 5 sites | push constants (≥128 B guaranteed) or UBO |
| Texture objects (PBA phase 3) | 2 sites | raw buffer works functionally; images/samplers optional |
| CUDA–OpenGL interop, CUDA IPC | visualizer | See [visualization_and_ipc.md](visualization_and_ipc.md) |

## Thrust (69 files)
`device_vector` 377 · `host_vector` 143 · `raw_pointer_cast` 82 · `zip_iterator` 72 · `tuple/make_tuple` 91 · `device_ptr` 56 · `transform` 47 · `reduce` 28 · `sort` 21 (+`sort_by_key` 4) · `fill` 16 · `counting_iterator` 16 · `copy` 14 · `exclusive_scan` 12 / `inclusive_scan` 5 · `minmax_element` 11 · `copy_if` 11 · `count_if` 8 · `remove_if` 6 · `unique` 5 · `binary_search` 6 · `set_intersection` 2.
Heaviest files: `octree/NTree.hpp` (256 hits) · `voxellist/TemplateVoxelList.hpp` (107) · `voxelmap/DistanceVoxelMap.cu` (83) · `voxellist/BitVoxelList.hpp` (72) · `voxellist/TemplateVoxelList.cu` (52) · octree `Tests.cu` (48) · `octree/PointCloud.cu` (35) · `octree/Sensor.cu` (30) · `voxelmap/VoxelMapOperations.cu` (29).
⇒ A **primitive library** (device buffer, fill, copy, transform, reduce, exclusive/inclusive scan, radix sort(+by key), copy_if/remove_if (stream compaction), unique(+by key), binary search, minmax) covers essentially all Thrust usage. Build it once, test it in isolation (the earlier attempt did: scan, radix sort, pack/unpack — see [reusable_assets.md](../vulkan_migration/reusable_assets.md)).

## Suggested porting order (inferred; by dependency and risk)
1. Primitives (buffer, scan, sort, compact, reduce) + runtime/context + error handling.
2. `BitVoxelMap` insert/clear/collide → `ProbVoxelMap` → `MetaPointCloud` transform.
3. `VoxelList` (sort/unique/merge) → list↔map collide.
4. `DistanceVoxelMap` (PBA, 8–9 kernels, shared memory).
5. Octree (largest: pointers, 64-bit, warp intrinsics, load balancer) — last, may need redesign.
6. Robot/URDF, `GpuVoxels` façade features, then visualizer/examples.
