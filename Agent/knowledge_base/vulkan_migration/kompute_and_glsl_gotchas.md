# Technical Gotchas (Vulkan, GLSL, Kompute, tooling)

Tags: **[old-doc]** = stated in the earlier attempts' notes (`kompute_migration_learnings.md`, `kompute_development.md`, plans), not re-run by us; **[log]** = measured from the old Kompute test log (`test_octree_build_out_v17.txt`, 2026-04-12, Intel UHD); **[general]** = general Vulkan knowledge, not tested in this repo.

## Shaders / GLSL
- **std430 layout:** `vec3`/`uvec3` have 16-byte alignment (arrays of them have stride 16) ⇒ host `Vector3ui` (12 B, Eigen) does not match. Use raw `uint`/`float` arrays with manual indexing (`i*3+k`), or `VK_EXT_scalar_block_layout`/`GL_EXT_scalar_block_layout` (advertised by the Intel UHD driver **[log]**). Add a host-side `static_assert(sizeof/offsetof)` test per shared struct. **[old-doc]** + **[general]**
- **Push constants:** use scalar `uint`/`float` members, no `uvec3`. Guaranteed minimum size is 128 bytes **[general]**. Kompute packs push constants as `std::vector<float>` (bit-cast) **[old-doc]**.
- **BitVoxel<256>:** treat as `uint data[8]` + `voxel_size_uints` passed to the shader; every shader hard-wires 256 bits ⇒ changing `BIT_VECTOR_LENGTH` needs a shader build variant. **[old-doc]**
- **Includes:** `#extension GL_GOOGLE_include_directive : enable` must follow `#version` immediately; pass `-I<dir>` to glslang. Use absolute source/output paths in CMake. **[old-doc]**
- **Integer `dot()`** does not exist for `ivec3`; write the sum of products. **[old-doc]**
- **Distance packing:** shader code used `(x&0x3FF)|((y&0x3FF)<<10)|((z&0x3FF)<<20)` (10 bit, range 0–1023) **[old-doc]**, while the CUDA header comment says 3×9 bit + marker bits (`DistanceVoxel.h`). Resolve against the CUDA `PBA_UNINITIALISED_COORD` definition before reusing; it bounds the maximum map edge (≤ 1023 or ≤ 511 voxels for distance maps).
- **Textures → buffers:** PBA phase 3 used CUDA texture objects; raw storage buffers were functionally sufficient. **[old-doc]**
- **Target env:** shaders compiled with `glslangValidator --target-env vulkan1.2 -V`. Needs `glslang[tools]` via vcpkg (no `glslangValidator` on PATH here).
- **Recursion is illegal in GLSL:** octree `findLeaf` had to be iterative. **[old-doc]**
- **Pointers:** none in GLSL (use indices; or `GL_EXT_buffer_reference` with `VK_KHR_buffer_device_address`, which the Intel UHD driver advertises **[log]**).
- **64-bit:** Morton codes (up to 60 bit) and 48-bit pointers: either `GL_EXT_shader_explicit_arithmetic_types_int64` (needs `shaderInt64`) or `uvec2` pairs (what attempt 2 did). `shaderBufferInt64Atomics` is far less available than 32-bit atomics **[general]**.
- **Float atomics:** the extension `VK_EXT_shader_atomic_float` is *advertised* on this Intel driver **[log]**, but the feature `shaderBufferFloat32AtomicAdd` is **false** (measured 2026-10-22) — check feature bits, not extension names. Prefer integer atomics (fixed-point or `floatBitsToUint` CAS loops).
- **Subgroups:** size varies (Intel 8/16/32, NVIDIA 32, AMD 32/64 **[general]**); `VK_EXT_subgroup_size_control` exists here **[log]**. Never hard-code 32 as CUDA's `warpSize` was.
- **Limits to query, never assume:** `maxComputeWorkGroupInvocations`, `maxComputeWorkGroupSize`, `maxComputeWorkGroupCount` (≥ 65535 per dim), `maxStorageBufferRange` (guaranteed only 128 MiB), `maxMemoryAllocationSize`, `minStorageBufferOffsetAlignment`. A 256³ `BitVoxelMap` is 512 MiB in one buffer; a 512³ one is 4 GiB. `MapVoxelID` is `uint32_t` ⇒ at most 4.29·10⁹ voxels; the earlier plan item "stress test 2048³" (8.6·10⁹ voxels, 256 GiB with 32-byte voxels) was never computable.

## Kompute (v0.9.0 pinned to commit `6e5380de…`) **[old-doc]/[log]**
- Tensor ↔ pointer: only memory created through the bridge's allocator could be passed to kernels (a registry hack, see [mistakes_to_avoid.md](mistakes_to_avoid.md) A2).
- `kp::Memory::dataType<T>()` specialisations for custom element types must be declared **inside `namespace kp`** as `template<> inline constexpr Memory::DataTypes Memory::dataType<T>()`.
- `Manager` asks for debug validation layers; on that machine **no Khronos validation layer was installed** (`no valid layer names found`; only OBS, Steam overlay and Steam fossilize implicit layers) **[log]**. Install the Vulkan SDK or vcpkg `vulkan-validationlayers` before trusting any run. Implicit layers (OBS capture hook, Steam overlay) can be disabled per process via their `DISABLE_…` environment variables **[general]**.
- Each `sequence()->eval<…>()` is submit + wait; one octree test issued 377 sequence submissions **[log]**. Batch operations into one recorded sequence/command buffer.
- spdlog debug output (every tensor/memory create/destroy) is on by default and swamps test output; set the log level to warn in tests.
- Kompute teardown order matters: tensors/algorithms/sequences must be released before the `Manager`; leaks of ~128-byte objects were reported but not resolved *(claimed)*.

## Windows / tooling **[general]/[old-doc]**
- Re-run `cmake --preset …` after editing `vcpkg.json` (manifest install happens at configure).
- Header-only template code guarded by `#ifndef GVL_USE_KOMPUTE` in both `.h` and `.hpp` was needed to compile `.cu` as C++ — a symptom of the bridge strategy, not a technique to repeat.
- PowerShell 5.1 `>` redirection writes UTF-16; see [working_rules.md](../../agent_policies/working_rules.md).
- Boost.Interprocess + Windows: see [visualization_and_ipc.md](../architecture/visualization_and_ipc.md).

## Device facts — Intel(R) UHD Graphics (iGPU, driver 31.0.101.2141) **[measured 2026-10-22, `vulkaninfoSDK.exe`]**
| Item | Value | Implication |
|---|---|---|
| Device API / instance | Vulkan 1.3.215 / loader 1.4.363 | Baseline Vulkan 1.3 core is available |
| `maxStorageBufferRange` | 1 073 741 820 (≈ 1 GiB) | 256³ BitVoxelMap (512 MiB) fits in one binding; 512³ (4 GiB) does not ⇒ split or BDA |
| `maxBufferSize` / `maxMemoryAllocationSize` | 4 GiB / 15.9 GiB (shared system-memory heap) | Single buffers ≤ 4 GiB |
| `maxComputeWorkGroupInvocations` / shared memory | 1024 / 32 KiB | CUDA's 1024 threads/block and typical shared usage fit |
| `maxPushConstantsSize` | 256 B | |
| Subgroup | size 32, range 8–32 (`subgroupSizeControl` = true) | Don't hard-code 32 |
| `shaderInt64` / `shaderFloat64` / `shaderInt16` / `shaderInt8` | true | 64-bit arithmetic OK (Morton codes) |
| `shaderBufferInt64Atomics`, `shaderSharedInt64Atomics` | **false** | No 64-bit atomics ⇒ octree/lists must not rely on them |
| `shaderBufferFloat32AtomicAdd` | **false** | No float atomic add |
| `bufferDeviceAddress`, `scalarBlockLayout`, `timelineSemaphore`, `synchronization2`, `vulkanMemoryModel`, 8-bit storage | true | Usable here; still optional for other vendors ⇒ feature-gate |
Earlier log (2026-04-12) also showed `VK_KHR_push_descriptor`, `VK_KHR_external_memory_win32`, `VK_KHR_shader_subgroup_extended_types`. A discrete NVIDIA/AMD GPU will differ — record limits per device in test logs (policy: required features = Vulkan 1.3 core + `shaderInt64`; the rest optional with fallback, see Q4 in [open_questions.md](../../project_state/open_questions.md)).
