# Back on the Intel UHD: test run of the finished version, and the unstable ray cast test (K45)

- **Date:** 2026-11-06 / 2026-11-07
- **Question:** does the finished version (built and tested on the RTX 4090 machine) build and pass on the Intel UHD again, and why does `VulkanTest.OctreeRayCastMatchesWalk` fail now and then there?
- **Scope / non-goals:** Vulkan backend only, Debug, validation layer on (unless stated). No CUDA, no timings of the older checkpoint tables (not repeated). No change to library code or tests: every experiment below was reverted.
- **Status:** final for the run; the cause of the flake is **not found** (see K45 in [known_issues.md](../knowledge_base/known_issues.md))

## Method
Machine: Intel Core i5-10210U, Intel UHD Graphics (integrated), driver 31.0.101.2141 (the newest for 6th-10th gen, April 6 2026), Vulkan device API 1.3, Windows 11 Home 10.0.26200, Vulkan SDK 1.4.363.0. Branch `vulkan-migration`, head `e8077dc`.
```
cmake -S . -B build/verify-vk -DGVL_BACKEND_CUDA=OFF -DGVL_BACKEND_VULKAN=ON -DBUILD_TESTS=ON     (reconfigure of the existing tree)
cmake --build build/verify-vk --config Debug --parallel
ctest --test-dir build/verify-vk -C Debug --timeout 900
```
Logs (git-ignored): `build/verify-vk-build.log`, `build/verify-vk-ctest.log`.

## Evidence
1. **measured** Reconfigure 4250 s (vcpkg rebuilt PCL and its dependencies because `vcpkg.json` changed since the last configure here). Build: no errors.
2. **measured** ctest: 456 tests listed, 449 run, 7 disabled (4 device-address fault experiments, `PrintDeviceAddressFormat`, `LongRayAgainstTheWatchdog`, `OctreeDynamicTimings`), **448 passed, 1 failed**, 2130 s. The older numbers in [machine_change_14900k.md](../project_state/machine_change_14900k.md) (218 listed) are from before the later slices.
3. **measured** The one failure, `VulkanTest.OctreeRayCastMatchesWalk` (`tests/vulkan/test_octree_sensor.cpp`): one cell of the ray count map has one ray too few, e.g. `cell 77189 has 5 rays, the reference 6`. Alone it passed, repeated it failed 2 of 5; with the validation layer switched off it failed 3 of 8.
4. **measured** The test has a fixed seed (21): the inputs are identical in every run, the GPU result varies.

Measurements on the flake (temporary test code, 3000 one-ray dispatches per call, long rays, wrong calls per number of calls):

| Setup | Wrong calls |
|---|---|
| 3000 submissions, fence wait after each (the test with chunk size 0) | 3-8 of 100 (varies with machine load) |
| same, command pool / descriptor set created once | 3 of 100 |
| 3000 submissions back to back, one wait at the end | 3 of 100 |
| same, full `AllCommands` memory barrier at the start and the end of every command buffer | 9 of 200 |
| all 3000 dispatches in ONE command buffer (with and without a barrier between them) | 0 of 200 |
| short rays onto few hot cache lines (map 0.5 MB or less), 1 ray per submission | 0 of 150 calls (125 000 submissions) |
| the real test, validation layer off | 3 of 8 runs |

Shape of a deviation: 2-34 cells per call, +1 and -1 mixed, almost all in one 64-byte line (7 of 8 cells in one line in the example that was printed); never a whole lost or duplicated ray, and the extra counts lie on no single ray's walk.

## Findings
1. Not the shader or `atomicAdd` (one lane is active per dispatch; 3000 dispatches in one submission never failed). **High.**
2. Not object churn, not the host wait, not a missing barrier, not the validation layer. **High** (the barrier test needed 200 calls: with a rate of 3-5 % the first test of 40 calls could not decide).
3. It needs many submissions together with long, scattered atomic traffic on a multi-MB buffer. The only explanation left is a cache write-back problem at the submission boundary of this GPU / driver. **Low** (not proven).
4. The default path (one submission per ray cast) was never seen failing. **Medium** (a few hundred calls).
5. Web search (Intel community, Mesa anv, Intel download pages): no report of this behaviour; the release notes of 31.0.101.2141 have no fixed / known issues section; 6th-10th gen is legacy (critical / security fixes only). A newer driver is not to be expected.

## What was NOT verified
Another GPU or driver (the comparison with the RTX 4090 is not wanted for now), the sync validation layer (blind on this driver, K20), shader parameters from a buffer instead of push constants, a plain `+=` instead of `atomicAdd` (planned, dropped because the rate needs very many calls), whether other kernels with many submissions show the same effect.

## Recommendation / decision needed
Keep the number of submissions low (the default does); read a failure of this one test as K45, not as a regression of the ray cast. The tests with chunk sizes 0 and 1000 stay as they are because they are what exposes the effect. No open question for the user.

## Promoted knowledge
[known_issues.md](../knowledge_base/known_issues.md) K45.
