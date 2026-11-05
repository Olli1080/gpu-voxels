# Vulkan Backend Runtime (P1 Foundation, 2026-10-22)

Module `packages/gpu_voxels/gpu_voxels/vulkan/` -> target `gpu_voxels_vulkan`, namespace `gpu_voxels::vulkan`, built only with `GVL_BACKEND_VULKAN=ON`. Stack: `vulkan.hpp` (RAII `vk::Unique*`, exceptions) with the statically linked loader (`Vulkan::Vulkan`), VMA (vcpkg `vulkan-memory-allocator`, implementation in `src/vma_impl.cpp`, not in public headers), logging via `GVL_LOG_*` (`vulkan/Log.h`; level, handler, default output stderr; no icl_core since 2026-11-05). Decision basis: [kompute_reevaluation.md](../vulkan_migration/kompute_reevaluation.md), [reach_reference.md](../reach_reference.md).
Status: **Tested** on Intel UHD (see results); not yet behind a CUDA/Vulkan abstraction interface; nothing of the voxel library uses it yet.

## API (all in `include/gpu_voxels/vulkan/`)
| Class | Purpose / contract |
|---|---|
| `Context` | Instance + device + compute queue + VMA allocator. Selects the best device (discrete > integrated > virtual > cpu; override `ContextCreateInfo::device_index` or env `GVL_VULKAN_DEVICE`). **Requires** Vulkan 1.3, `synchronization2`, `shaderInt64`, a compute queue (family without graphics preferred) - nothing else (policy Q4). Enables `VK_LAYER_KHRONOS_validation` when installed (`require_validation` throws `ValidationUnavailableError`); a debug-utils messenger counts errors/warnings (`validation_error_count()`, `validation_messages()`). `execute(fn)` = one-time command buffer + submit + wait; queue access is serialised by a mutex. Exposes `info()` (`DeviceInfo`) and `limits()`. Not copyable. |
| `DeviceBuffer` | Device-local storage buffer (VMA `AUTO_PREFER_DEVICE`), no host mirror. `upload/download` via staging buffers in 64 MiB chunks (blocking), `fill(uint32)`, typed helpers. Rejects size 0, size > `maxStorageBufferRange` (whole-buffer binding only for now) and out-of-range transfers. Movable. |
| `ComputePipeline` | SPIR-V + N storage buffers at set 0 / bindings 0..N-1 (max 16) + push constants (multiple of 4, within the device limit); entry point `main`; `load_spirv(path)`. |
| `BitVoxelMap` | Slice S1, see [vulkan_bit_voxel_map.md](vulkan_bit_voxel_map.md). |
| `CommandBatch` | One command buffer: `dispatch(pipeline, {&buf...}, gx,gy,gz, push)`, `fill(buffer, value)` (+ barrier), `barrier()` (compute write -> compute read/write via `VkMemoryBarrier2`), `submit_and_wait()`. Single use. Descriptor sets come from per-batch pools (64 sets each, grows). Validates buffer count, push size and group counts against `maxComputeWorkGroupCount`. **Dispatches are unordered unless `barrier()` is called.** |
| errors | `VulkanError` (runtime), `NoDeviceError`, `ValidationUnavailableError`; vulkan.hpp throws `vk::SystemError`. |

Since S6a (octree foundation, [vulkan_octree_foundation.md](vulkan_octree_foundation.md)): `bufferDeviceAddress` is enabled when the device has it (`DeviceInfo::buffer_device_address`, switch `ContextCreateInfo::enable_buffer_device_address`), `DeviceBuffer::device_address()` returns the shader address of a buffer and throws without the feature.

Design choices: descriptor pools instead of `VK_KHR_push_descriptor` (stay within core 1.3); shaders compiled offline to SPIR-V (`cmake/Shaders.cmake`: `gvl_add_shaders` for test shaders loaded from files, `gvl_embed_shaders` for library kernels embedded as C++ arrays; glslang from the Vulkan SDK, `--target-env vulkan1.3`); no pointer or "unified memory" emulation (guardrail 5); sync validation requested through `VK_EXT_layer_settings` (`validate_sync`) with `VK_EXT_validation_features` as fallback.

## Tests (`tests/vulkan/`, target `test_vulkan`, 14 cases)
Context (device policy, **validation oracle**: a deliberately invalid `vkCreateBuffer` must be counted), buffers (round trip, offsets, fill, invalid sizes/ranges, move, **160 MiB across staging chunks**), dispatch (1 000 003-element kernel, two dependent dispatches with barrier in one submit, 200 dispatches in one batch -> pool growth, pipeline reuse, **int64 arithmetic in a shader**, misuse errors). The fixture creates one `Context` per process and fails every test that produced a validation error; env `GVL_ALLOW_NO_VALIDATION=1` permits running without the layer; no device -> tests skip.

## Verified results (2026-10-22, Intel UHD Graphics, API 1.3, validation layer 1.4.363, Debug)
- 14/14 pass, zero validation errors. 160 MiB: upload 137 ms, download 177 ms. `add_one` over 1 M elements: 1.5 ms (single measurement incl. submit/wait; not a benchmark).
- Mutation checks of the oracle: removing the barrier in the 200-dispatch test makes it **fail** (lost updates are observable); removing it between `add_one` and `scale_copy` still passes on this GPU **and no synchronization-validation message appeared** (K20) - that one test is a weak oracle for missing barriers.

## Open / next
- Abstract buffer/dispatch interface shared with the CUDA backend; range binding for buffers larger than `maxStorageBufferRange`; pipeline cache; specialization constants; timeline semaphores / async submit; multi-queue.
- Port kernels per [reusable_assets.md](../vulkan_migration/reusable_assets.md), starting with slice S1 (`BitVoxelMap`) of the plan.
