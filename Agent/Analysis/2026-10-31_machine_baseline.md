# Machine baseline: i9-14900K with NVIDIA GeForce RTX 4090

- **Date:** 2026-10-31
- **Question:** Does the Vulkan side (branch `vulkan-migration`, head `cd63312` plus the working-tree changes listed below) build and pass on the new machine, and which timing-based decisions made on the Intel UHD still hold? (Template of [machine_change_14900k.md](../project_state/machine_change_14900k.md), section 3.)
- **Scope / non-goals:** Vulkan only, Debug builds, validation layer on, single runs (not benchmarks). No CUDA build (the toolkit 12.8 is installed, nothing was run with it). Driver-issue experiments K20/K25/K28/K29 and the mutation runners were not repeated.
- **Status:** final for this machine (K35 checked by a hand-made mutation run, not by a runner)

## Method
Clone at `C:\Code\gpu-voxels` (branch `vulkan-migration`), `bootstrap-vcpkg.bat`, Vulkan SDK 1.4.363.0 by winget, generator "Visual Studio 18 2026", the two configurations of the machine-change page (`build/verify-cpu`, `build/verify-vk`), first configure 228 s, second (binary cache) 23 s. `ctest --test-dir build/verify-vk -C Debug` and one `test_vulkan.exe` run in a single process; raw logs in `build/` (git-ignored): `ctest-vk-run1/2/3.log`, `test_vulkan_full_run.log`.

## Evidence
**Device (measured):** Windows 11 Pro 10.0.26200; NVIDIA GeForce RTX 4090, driver 621.01 (Vulkan 1.4.357, discrete), 24 GiB, the only device Vulkan reports (the 14900K iGPU does not show up). Subgroup size 32..32, `maxStorageBufferRange` 4 294 967 295 (UHD: 1 GiB), `maxWorkGroupInvocations` 1024, 64-bit buffer atomics **yes** (UHD: none), `bufferDeviceAddress` yes (octree tests ran, none skipped), CUDA toolkit 12.8 and `nvcc` present.

**Results (measured):**
| Configuration | Result |
|---|---|
| `verify-cpu` ctest | 30 of 31 pass; failure `ts_icl_core` (K18, expected) |
| `verify-vk` ctest, run 3 (with the fix K35) | 225 listed, 221 run, 220 pass, 4 disabled fault experiments, 1 failure `ts_icl_core` (K18): the expectation of the machine-change page |
| `test_vulkan` in one process | 190 of 190 pass, 4 disabled, zero validation errors reported |

**Timings (measured, RTX 4090, next to the old Intel UHD numbers where the docs have them):**
| Measurement | RTX 4090 | Intel UHD (old machine) |
|---|---|---|
| Octree vs octree, 2 x 1 M voxels, wavefront | 5.8-7.8 ms | 52-55 ms |
| same, load balancer | 14-16 ms (12 launches) | 82-110 ms (12-13 launches) |
| Octree extract 1 M voxels (5.4 M nodes), wavefront | 250-255 ms | 690-750 ms |
| same, load balancer | 284-291 ms | 770-880 ms |
| Octree build 1 M voxels incl. transfers (det. / prob.) | 58 / 59 ms | see [checkpoint measurements](2026-10-30_octree_checkpoint/measurements.md) |
| Octree vs 2 M voxel map | 0.96-1.06 ms | about 30 ms (1 M voxel tree vs 2 M map, other workload) |
| BitVoxelMap collide 2 x 512 MiB | 1.16 ms | see [vulkan_bit_voxel_map.md](../knowledge_base/architecture/vulkan_bit_voxel_map.md) |
| Scan 5 M / radix sort 2 M pairs / sort 16 M pairs 64-bit keys | 10.9 ms / 14.8 ms / 282 ms | see [vulkan_primitives.md](../knowledge_base/architecture/vulkan_primitives.md) |
Bytes per voxel and frontier sizes do not depend on the GPU (lines in the log, equal to the checkpoint report). The `Checkpoint*` cases are slow as processes on the 4090 (`CheckpointBytesPerVoxel` 112 s, `CheckpointFrontierOfTreeCollision` 75 s in ctest; cause not investigated).

## Findings
1. **Everything the old machine passed passes here** once K35 is fixed (high; measured).
2. **K35: the load balancer livelocked on the RTX 4090** (high; measured, cause confirmed by an intervention): all groups start together, the working group saw the idle threshold before its first iteration, no progress per launch. Before the fix: 175 s, three timeouts (900 s / 40 s+ / endless), 167 launches for a probabilistic tree collision; after: 0-10 s and 12 launches. Details and the changed shaders: [known_issues.md](../knowledge_base/known_issues.md) K35, [vulkan_octree_collision.md](../knowledge_base/architecture/vulkan_octree_collision.md).
3. **The default (wavefront) is confirmed on this GPU** (high): it is faster than the balancer for tree vs tree (2.4x) and extract (1.1x). The balancer stays selectable.
4. The GPU is 8-10x faster than the UHD on the tree work, but the extract is host- or allocation-bound (250 ms for 5.4 M nodes) (medium: only a ratio, not profiled).

## What was NOT verified
CUDA (no CUDA configuration built or run); the balancer comparison on a second GPU; Release builds; GPU-assisted validation; why the 2.8 s ray survives a default Windows TDR; the two slow `Checkpoint*` cases. The test counts above were taken before the two new experiments below were added (`DISABLED_PrintDeviceAddressFormat`, `DISABLED_LongRayAgainstTheWatchdog`: 227 listed, 6 disabled).

## Driver experiments and mutation check (measured 2026-10-31, same machine)
| Item | RTX 4090 | Intel UHD (docs) |
|---|---|---|
| K28 bad device addresses (4 cases, one process each) | same four outcomes: past the end and freed buffer complete with a neighbour's value, 64 MiB past gives 0, address 0 gives `ErrorDeviceLost` after 2.9 s; validation silent | same |
| K29 address format | upper 16 bits 0 (0x7400000, 0x9600000); sign extension of bit 47 is a no-op | upper 16 bits set |
| K20 sync validation vs missing barrier | not reported (test passes, no message, also with the settings file) | not reported |
| K25 long ray (2^24 x 1 x 1 map) | no device loss: 16 777 215 voxels in 2.8 s, linear (166 ns/voxel) | device lost |
| Mutation check of K35 (`ASSERT_LT(kernel_launches, 60)` and the balancer tests) | 5 mutants: fix removed, pops never counted, pops not reset in either shader: all killed (timeouts or failing tests); check after the second iteration (`> 1u`) survives: equivalent boundary mutant (it changes only when a launch ends) | n/a |
The mutation check was done by hand (one mutation, build, three balancer tests, restore), not with a `tools/mutation` runner; the source files were restored afterwards (`git status` clean for them).

## Recommendation / decision needed
- Keep the K35 fix (it deviates from the order of checks in CUDA). Decision for the user.
- The driver experiments are done (above): nothing in them changes the code. Next: S6d. Q25 is answered in part: NVIDIA RTX 4090, CUDA 12.8 installed, so CUDA parity (Q5) is now possible; the user decides when.

## Promoted knowledge
[known_issues.md](../knowledge_base/known_issues.md) K35; [vulkan_octree_collision.md](../knowledge_base/architecture/vulkan_octree_collision.md) (progress guarantee).
