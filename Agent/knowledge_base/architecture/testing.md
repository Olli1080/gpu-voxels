# Testing Infrastructure (since 2026-10-22)

## Layout
- `tests/CMakeLists.txt` — registers test targets with `gvl_add_test()` (defined in `cmake/Tests.cmake`).
- `tests/test_categories.cmake` — registry of categories. Currently only `GVL_TEST_PERFORMANCE_FILTERS` (GoogleTest filter patterns); empty. Performance tests are registered as CTest tests `perf.<Suite>.<Test>` with label `performance` and skipped by the default run.
- `tests/smoke/` — backend-independent tests that run on any machine:
  - `test_build_config.cpp`: the `GVL_BACKEND_CUDA/VULKAN` compile definitions match the CMake options.
  - `test_icl_core_enum_helper.cpp`: 5 cases for the vendored `icl_core::string2Enum`.
- `tests/octree/` (always built, no GPU): `test_octree_reference`, 21 cases - host Morton code, the CPU octree reference ([vulkan_octree_foundation.md](vulkan_octree_foundation.md)) and the node formats ([vulkan_octree_build.md](vulkan_octree_build.md)).
- `tests/vulkan/` (only with `GVL_BACKEND_VULKAN=ON`): `test_vulkan`, 332 cases (190 up to S6c, 4 chunk cases of S6d-0, 11 insert cases of S6d-2 .. S6d-3b, 9 rebuild cases of S6d-4, 5 fast-insert cases of S6d-5, 12 facade and storage cases of S6e, 4 garbage cases of S6d-6, 2 exact-propagate cases of S6d-7, 6 voxelize cases of S6f-1, 5 ray cast cases of S6f-2, 5 packing cases of S6f-3, 5 sensor pipeline cases of S6f-4, 14 façade cases of S7-1, 12 point cloud file cases of S7-2, 9 robot façade cases of S7-3, 16 merge and disk cases of S7-4, 7 list collision, 7 primitive and 9 example smoke cases of S7-5; 7 disabled experiments, among them the timings of S6d-4) (7 size limit cases: [vulkan_primitives.md](vulkan_primitives.md), K34; 3 checkpoint measurement cases: [Analysis/2026-10-30_octree_checkpoint](../../Analysis/2026-10-30_octree_checkpoint/summary.md); 7 octree load balancer cases: [vulkan_octree_collision.md](vulkan_octree_collision.md); 18 octree collision cases: [vulkan_octree_collision.md](vulkan_octree_collision.md); 17 octree build / find / extract / check_tree and 64-bit sort: [vulkan_octree_build.md](vulkan_octree_build.md); 6 device address / Morton: [vulkan_octree_foundation.md](vulkan_octree_foundation.md); 4 further `DISABLED_` fault experiments run only on request, one process each; 14 runtime: [vulkan_runtime.md](vulkan_runtime.md); 16 BitVoxelMap: [vulkan_bit_voxel_map.md](vulkan_bit_voxel_map.md); 19 MetaPointCloud / self collision / DH chain: [vulkan_meta_cloud_and_robot.md](vulkan_meta_cloud_and_robot.md); 7 scan / radix sort / iota: [vulkan_primitives.md](vulkan_primitives.md); 24 voxel lists: [vulkan_voxel_lists.md](vulkan_voxel_lists.md); 29 ProbVoxelMap / sensor data: [vulkan_prob_voxel_map.md](vulkan_prob_voxel_map.md); 23 DistanceVoxelMap / PBA: [vulkan_distance_map.md](vulkan_distance_map.md); all against the CPU references in `tests/reference/`). Needs a Vulkan 1.3 device and the validation layer (env `GVL_ALLOW_NO_VALIDATION=1` relaxes, `GVL_VULKAN_DEVICE=<i>` picks a device).
- CUDA tests: not yet (placeholder comment in `tests/CMakeLists.txt`).

## Rules
1. Always `gvl_add_test(name SOURCES … [LIBS …] [DEFINITIONS …])`; never raw `add_executable` for tests. It links `GTest::gtest_main`, requires C++20, discovers cases with `gtest_discover_tests(... DISCOVERY_MODE PRE_TEST)` (no test executable is run during the build, so DLL lookup problems cannot break the build).
2. GoogleTest is the framework for gpu_voxels tests (user decision, matches REACH). The vendored `icl_core` keeps its own Boost.Test suites (`ts_icl_core`, `ts_icl_core_config`); they are built only when `BUILD_TESTS` is ON.
3. Tests must have an independent reference (see [definition_of_done.md](../../agent_policies/definition_of_done.md)); a regression test's comment names the bug/commit.

## Running (Windows, Visual Studio generator)
```
ctest --test-dir build/<dir> -C Debug -LE performance --output-on-failure
```
or use the test presets (`ctest --preset cuda-debug`, `vulkan-debug`), which exclude the label `performance`.

## Verified results (2026-10-22, CUDA=OFF, MSVC 19.51, Debug, deps via the in-repo vcpkg)
`ctest`: 9 tests, 8 pass (2026-10-22; since 2026-10-29 the CPU-only build also contains `test_octree_reference`: 31 ctest cases, 30 pass, same single failure); `ts_icl_core` fails with "test tree is empty" — **pre-existing** (see K18 in [known_issues.md](../known_issues.md)): `packages/icl_core/icl_core/tests/ts_main.cpp` only defines `BOOST_TEST_MAIN`, no cases.

With `GVL_BACKEND_VULKAN=ON` (build dir `build/verify-vk`; ctest of 2026-11-01 after S6d-6, Debug, 4 in parallel, 116 s): 296 ctest cases listed, 289 run (the 7 disabled experiments are listed but not run), 288 pass (same pre-existing `ts_icl_core` failure, K18).

## Reference implementations
`tests/reference/` holds independent CPU references (rule: written from the CUDA sources, no shared code or data layout with the implementation under test). Currently `bit_voxel_map_reference.h` (sparse bitset map incl. meta-cloud self collision), `kinematics_reference.h` (double precision DH/FK) `voxel_list_reference.h` (`std::map` keyed by voxel id; bit/counting lists, collision, subtract) `prob_voxel_map_reference.h` (sparse int map, explicit update events, ray walk, all serial orders of a voxel's updates) and `distance_map_reference.h` (obstacle set + brute force distances). Validate the tests themselves by mutation testing (modify one shader/kernel line, the tests must fail) - see the S1 page for the procedure and results.
