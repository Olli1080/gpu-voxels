# Machine change (planned): moving to an Intel Core i9-14900K machine

**Status:** announced by the user on 2026-10-30 ("we will change the machine soon, at the next phase, to a 14900K"). Not done yet. What is **not known**: the GPU(s) of the new machine (the 14900K has an integrated UHD 770; whether there is a discrete GPU, and whether an NVIDIA one, is open: Q25), the OS and driver versions, whether the old machine stays available.
This page lists what is bound to the current machine, how to rebuild the environment, and what to re-measure first. Rules that do not change with the machine: [AGENTS.md](../../AGENTS.md).

## 1. What exists only on this machine (carry it over, or it is lost)
| Item | Where | Risk | Action |
|---|---|---|---|
| Branch `vulkan-migration`: **9 local commits, never pushed** (`git log master..vulkan-migration`, last `da4d3fd`; `origin/master` = `dd47249`) | `.git` of `C:\Code\2026-10-22_gpu-voxels` | the work exists only in this clone | the user decides how to move it: copy the whole folder including `.git`, or `git bundle create`, or push (agents never push unasked, [working_rules.md](../agent_policies/working_rules.md)) |
| **Uncommitted** changes and shaders of the failed attempts | `C:\Code\gpu-voxels` (303 changed entries; 49 uncommitted `.comp` shaders exist nowhere else), `C:\Code\gpu-voxels__` | read-only reference material the docs link to; **no other copy** | copy both folders completely (with `.git`) before the old machine is wiped |
| Reference project for CMake / package conventions | `C:\Code\REACH` ([reach_reference.md](../knowledge_base/reach_reference.md)) | outside the repository | copy if it is still wanted |
| Build products and dependencies | `build/verify-vk`, `build/verify-cpu`, vcpkg installed trees and `vcpkg.exe` (all git-ignored) | rebuilt in about 10-20 min | recreate (section 2), do not copy |
| Tools: Visual Studio 2026 ("18") and 2022, CMake, Ninja, Git, Python 3.13 (`C:\Python313`), Vulkan SDK 1.4.363.0 (`C:\VulkanSDK\1.4.363.0`, `VULKAN_SDK`) | machine | see [current_state.md](current_state.md) | install again (downloads need the user's go-ahead) |
| Scripts used for verification | were in the session scratchpad (temporary) | lost with the session | **saved on 2026-10-30 into [../../tools/](../../tools/README.md)** (mutation runners, doc checker, their logs) |
| The agent's own memory notes of this tool | the tool's user profile folder | not part of the repository | not needed: everything that matters is in `AGENTS.md` and `Agent/` |

## 2. Setting up the new machine
```
git clone <wherever the branch was moved to> C:\Code\2026-10-22_gpu-voxels      (or copy the folder)
git switch vulkan-migration
git submodule update --init vcpkg                         (pointer d4c0a46cb8, 2026-10-28; do not change it unasked)
vcpkg\bootstrap-vcpkg.bat
```
Install the Vulkan SDK 1.4.363.0 or newer (validation layer, glslang) and Visual Studio with the C++ workload (the build used the generator "Visual Studio 18 2026"; with 2022 use "Visual Studio 17 2022"). Then the two verified configurations (see [build_system.md](../knowledge_base/architecture/build_system.md); the first configure builds Boost, gtest, zlib and others through vcpkg, about 10 minutes):
```
cmake -S . -B build/verify-cpu -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CD%/vcpkg/scripts/buildsystems/vcpkg.cmake -DGVL_BACKEND_CUDA=OFF -DBUILD_TESTS=ON
cmake -S . -B build/verify-vk  -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CD%/vcpkg/scripts/buildsystems/vcpkg.cmake -DGVL_BACKEND_CUDA=OFF -DGVL_BACKEND_VULKAN=ON -DBUILD_TESTS=ON
cmake --build build/verify-vk --config Debug --parallel
ctest --test-dir build/verify-vk -C Debug
```
Pitfalls already known: a build directory that was configured before the vcpkg bump needs `-UZLIB_LIBRARY_DEBUG -UZLIB_LIBRARY_RELEASE -UZLIB_INCLUDE_DIR` and `--clean-first` (K30); shells started before the SDK install lack `VULKAN_SDK` (K16); the repository files are CRLF, Git prints "LF will be replaced by CRLF" warnings (harmless).
**Expected result before anything else is changed:** `test_vulkan` all pass (183 cases, 4 further ones are disabled fault experiments), `ctest` in `build/verify-vk` 213 of 214 run cases (218 listed), in `build/verify-cpu` 30 of 31; the one failure is `ts_icl_core` (K18, not machine dependent). Anything else is a finding about the new machine: write it down before fixing.

## 3. What to measure first on the new machine
The numbers of the checkpoint and the other `[vulkan] ...` timing lines were taken on an **Intel UHD (integrated, driver 31.0.101.2141)** and say nothing about another GPU. Record the new device in a new `Analysis/<date>_machine_baseline.md` (template: [Analysis/_template.md](../Analysis/_template.md)):
1. **Device facts** (`vulkaninfoSDK --summary`, the first lines of `test_vulkan`, `VulkanTest.DeviceInfoMeetsPolicy`): API version, subgroup size range (UHD: 8-32), `maxStorageBufferRange` (UHD: 1 GiB), `bufferDeviceAddress`, `shaderInt64`, 64-bit atomics (UHD: none), max work group counts and invocations. `Context` selects discrete before integrated GPUs; `GVL_VULKAN_DEVICE=<index>` forces one ([vulkan_runtime.md](../knowledge_base/architecture/vulkan_runtime.md)). If the machine has an iGPU **and** a discrete GPU, run the suite on both.
2. **Driver-specific known issues** (each could disappear or change): K29 canonical addresses with the upper 16 bits set (Intel; the octree sign-extends bit 47: confirm the tests still pass, and note the address format of the new driver); K28 bad device addresses (silent / device lost; the four disabled experiments in `test_device_address_faults.cpp`, run one at a time with `--gtest_also_run_disabled_tests`, a device loss ends the process); K25 the watchdog (a 16.7 M voxel ray loses the device on UHD); K20 synchronization validation (does it report the deliberately missing barrier on the new driver? settings file: [tools/vk_layer_settings_sync_validation.txt](../../tools/vk_layer_settings_sync_validation.txt)).
3. **Timings to repeat** (same tests, new numbers next to the old ones): `Checkpoint*` (64-bit sort, bytes per voxel, frontiers), `BalancerOnLargeTrees` (wavefront against load balancer: **the default was chosen on UHD; re-validate it**, the balancer may win on a GPU with many compute units), `OctreeBuildOfOneMillionVoxels`, `OctreeTreeCollisionOfLargeTrees`, the BitVoxelMap 2 x 512 MiB collide, scan 5 M and sort 2 M. Bytes per voxel and frontier sizes do not depend on the GPU and need no repetition. Run with `--gtest_filter` and keep the `[vulkan]` lines.
4. **Mutation runs get faster** with more cores and a faster GPU, but the runners must still run one at a time on one build tree ([mutation_testing.md](../knowledge_base/mutation_testing.md)).

## 4. If the new machine has an NVIDIA GPU (Q25)
CUDA parity (Q5, "later") becomes possible: CUDA toolkit (a download, ask first), `cmake --preset cuda`, the existing CUDA tests, and golden data of the octree and the maps from the original code. Everything the docs tag *inferred* (K27, K31-K33 and the semantics notes of the slices) could then be checked, and the word "Parity" could be used. Suggested order: (1) baseline of the Vulkan side (section 3), (2) CUDA configuration builds and its tests run, (3) golden data for the slices in the order of the plan, (4) update the *inferred* tags. Nothing of this is started or decided.

## 5. Where timings in the docs come from
Timing tables carry the device in the text (`Intel UHD`). New tables should add a device column and keep the old rows labelled "Intel UHD (old machine)" instead of replacing them: [vulkan_octree_collision.md](../knowledge_base/architecture/vulkan_octree_collision.md#measurements), [checkpoint measurements](../Analysis/2026-10-30_octree_checkpoint/measurements.md), [vulkan_runtime.md](../knowledge_base/architecture/vulkan_runtime.md), [vulkan_bit_voxel_map.md](../knowledge_base/architecture/vulkan_bit_voxel_map.md).
