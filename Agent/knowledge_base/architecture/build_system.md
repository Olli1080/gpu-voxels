# Build System

## Layout of master before 2026-10-22 (historical; changes below)
- Root `CMakeLists.txt`: `cmake_minimum_required(3.23)`, `project(GPU_VOXEL_TL LANGUAGES CXX CUDA)`, cache var `NVCC_THREADS` (default 1), MSVC `/MP`, CUDA flags `--expt-extended-lambda --expt-relaxed-constexpr -Xcudafe --diag_suppress=221 --threads=N`, on Windows `-Xcompiler /bigobj`. `BUILD_TESTS` is hard-set `ON` (only enables `enable_testing()`).
- `packages/CMakeLists.txt` → `icl_core`, `gpu_voxels`. Each module is its own static/shared library target with `FILE_SET HEADERS`, `CUDA_SEPARABLE_COMPILATION ON`, `CUDA_ARCHITECTURES all`, export via `gpu_voxels-targets` → `share/gpu_voxels/gpu_voxels-config.cmake`.
- `CMakePresets.json` (version 2): preset `debug`, generator **"Visual Studio 17 2022"**, binary dir `build/default`, toolchain `./vcpkg/scripts/buildsystems/vcpkg.cmake`, install prefix `install/debug`. Only configure preset; no build/test presets.

## Dependencies (`vcpkg.json`, manifest mode)
`pcl` (feature `openni2`), `boost-interprocess`, `boost-test`, `tinyxml2`, `cppunit`, `cuda`, `glew`, `glm`, `freeglut`, `zlib`, `stb`, `eigen3`.
- `vcpkg-configuration.json`: `overlay-ports: ["./overlay-ports"]`.
- `overlay-ports/eigen3` — Eigen 3.4.0 port-version 3 with `fix-vectorized-reductions-half.patch` and `remove_configure_checks.patch` (nvcc compatibility; README: "NVCC issue but patch available and fixed in Eigen 3.5").
- `overlay-ports/cudnn` — cuDNN 7.6.5 port (reason not documented; possibly pulled by the PCL/`cuda` dependency chain).
- `vcpkg/` submodule: bumped on 2026-10-30 (user decision Q10) from `984f9232b2` (2025-06-06, the pin of the upstream repository) to `d4c0a46cb8` = commit of 2026-10-28; `vcpkg/vcpkg.exe` re-bootstrapped (vcpkg-tool 2026-09-26, git-ignored). No `builtin-baseline` in the manifest ⇒ versions are those of that registry snapshot (Boost 1.92, zlib 1.3.2, gtest 1.18). After a bump see [K30](../known_issues.md): drop the stale zlib cache entries and rebuild with `--clean-first`.
- Missing from the manifest but used: `urdfdom`, `orocos-kdl` (URDF robot, disabled), `octomap`/OMPL/ROS (examples, `gvl_ompl_planning`).

## Legacy: ICMAKER
Original FZI build system. `ICMAKER_SET / ADD_SOURCES / ADD_CUDA_FILES / INTERNAL_DEPENDENCIES …` macros no longer exist in the tree. Still used by: `packages/examples/**`, `packages/gpu_visualization/CMakeLists.txt`, test CMake files (`gpu_voxels/test`, `voxelmap/tests`, `octree/test`), `doc/CMakeLists.txt`, plus commented relics in module CMakes. Porting these to plain CMake is a prerequisite for building examples/viewer/tests (the earlier attempt rewrote some of it; see [attempt_history.md](../vulkan_migration/attempt_history.md)).
Also stale: `packages/gpu_voxels/CMakeLists_old.txt`, `Dependencies.conf`, `package.xml`, `pom.xml`.

## What the previous Vulkan attempts added (for orientation)
- `find_package(kompute|Vulkan|spdlog|GTest)`, `GVL_WITH_KOMPUTE` / `GVL_WITH_SYCL` options, `cmake/ShaderCompile.cmake` (`compile_shaders(target list)` → `glslangValidator --target-env vulkan1.2 -V`), a `kompute` overlay port pinned to GitHub commit `6e5380de…` (version 0.9.0), vcpkg additions `kompute vulkan-loader vulkan-headers gtest glslang[tools]`, preset generator "Visual Studio 18 2026". Not part of master.

## Toolchain facts on this machine (measured 2026-10-22)
Present: CMake (`C:\Program Files\CMake\bin`), Ninja (WinGet), Git, Python 3.13, `vulkaninfo.exe`, Visual Studio under `C:\Program Files\Microsoft Visual Studio\{18,2022}` and x86 `{2017,2019}`.
Absent from PATH/env: `nvcc`, `CUDA_PATH`, `VULKAN_SDK`, `glslangValidator`, `glslc`, `dxc`, `slangc`, `cl` (needs a VS developer prompt). GPU: Intel UHD Graphics only.
⇒ CUDA targets cannot be built or run here; Vulkan work must get its shader compiler via vcpkg (`glslang[tools]`) or the LunarG SDK.

## CUDA build on the RTX 4090 machine (2026-11-03)
The statements above about "no CUDA here" are from the old machine. Now: `nvcc` 12.8 and the driver are present, but Visual Studio 2022 is not (only VS 18 with the toolsets 14.44 and 14.51); CUDA 12.8 rejects 14.51 and has no MSBuild integration for VS 18, so use **Ninja + `vcvarsall.bat x64 -vcvars_ver=14.44`** (full commands: [cuda_parity](../../Analysis/2026-11-03_cuda_parity.md)). New cache variable `GVL_CUDA_ARCHITECTURES` (default `all`; `89` for the RTX 4090) replaces the hard coded `all` of every `gpu_voxels` target. Needed for the parity programs: `-DCMAKE_CUDA_FLAGS=-maxrregcount=64` (K39). Build trees `build/cuda-ninja` (Debug) and `build/cuda-rel` (Release), both git-ignored. `tests/cuda_parity/` (only with `GVL_BACKEND_CUDA`) holds the CUDA dump programs and the golden files.

## After the L1–L4 work (2026-10-22, uncommitted in the working tree)
Plan: [package_layout_proposal.md](../../plans_wip/package_layout_proposal.md). Module-local source layout unchanged (D1 default).
- **Root `CMakeLists.txt`** is slim: `cmake/Options.cmake` (before `project()`), `project(… LANGUAGES CXX)`, `cmake/Backends.cmake`, `cmake/FindDependencies.cmake`, `add_subdirectory(packages)`, then `cmake/Tests.cmake` + `tests/` when `BUILD_TESTS`.
- **Options:** `GVL_BACKEND_CUDA` (default ON; configure fails with a clear message if no `nvcc`), `GVL_BACKEND_VULKAN` (OFF; *only locates dependencies*, no runtime yet), `GVL_WITH_PCL` / `GVL_WITH_KINECT` (ON; replace the old `if(ON) #TODO` blocks in `helpers`), `BUILD_TESTS` (ON, was hard-set). Backends define `GVL_BACKEND_CUDA=1` / `GVL_BACKEND_VULKAN=1`.
- **CUDA is optional:** `enable_language(CUDA)` and the nvcc flags moved into `Backends.cmake`; `packages/gpu_voxels/CMakeLists.txt` adds only `gpu_voxels/logging` without CUDA, all other modules inside `if(GVL_BACKEND_CUDA)`. With CUDA OFF the build contains `icl_core*`, `gpu_voxels_logging`, tests.
- **Central discovery:** `CUDAToolkit`, `Stb`, `PCL` moved from `helpers/CMakeLists.txt` to `FindDependencies.cmake`; Vulkan block finds `Vulkan` (SDK), `VulkanMemoryAllocator` (vcpkg only — the SDK does not ship VMA), `glslangValidator`. `icl_core` (vendored) still does its own discovery.
- **vcpkg manifest:** base = `boost-interprocess tinyxml2 zlib`; features `cuda` (cuda, eigen3, glm, stb), `pcl` (pcl[openni2]), `vulkan` (vulkan-headers, vulkan-memory-allocator; the Vulkan SDK supplies loader, validation layer and glslang), `tests` (boost-test, cppunit, gtest), `viewer` (freeglut, glew, glm; currently unused by CMake). Options append `VCPKG_MANIFEST_FEATURES`. `default-features` is empty. (`vulkan-hpp` is a deprecated port; `vulkan-headers` provides `vulkan.hpp`.) `eigen3` is only in `cuda`, so CUDA-free configurations avoid the eigen3 overlay port (see K17); add it to `vulkan` once shared types need it.
- **Presets (v7):** hidden `base` (toolchain `${sourceDir}/vcpkg/…`), `cuda` (VS 2022), `vulkan` (VS 2026, CUDA off), `both`, legacy `debug` (= cuda, `build/default`); build presets `{cuda,vulkan}-{debug,release}`; test presets exclude label `performance`. Only `cmake --list-presets` was verified, the presets themselves were not run.
- **icl_core tests:** `add_subdirectory(tests|ts|icl_core_testsuite)` now guarded by `if(NOT DEFINED BUILD_TESTS OR BUILD_TESTS)` (minimal vendored-code change).
- `cmake/Shaders.cmake`: `gvl_add_shaders(target OUTPUT_DIR dir SOURCES ... [INCLUDE_DIRS ...])` -> glslangValidator -> `.spv` (loaded from files by tests); `gvl_embed_shaders(target SOURCES ...)` additionally embeds the SPIR-V as generated C++ arrays via `cmake/EmbedSpirv.cmake` (used by the library; no shader files at run time). New module `gpu_voxels/vulkan` (target `gpu_voxels_vulkan`, only with `GVL_BACKEND_VULKAN`): [vulkan_runtime.md](vulkan_runtime.md).
- Tests: [testing.md](testing.md).

### Reproduce the verified configuration (this machine, no CUDA)
```
cmake -S . -B build/verify-cpu -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CD%/vcpkg/scripts/buildsystems/vcpkg.cmake -DGVL_BACKEND_CUDA=OFF -DBUILD_TESTS=ON
cmake --build build/verify-cpu --config Debug --parallel
ctest --test-dir build/verify-cpu -C Debug -LE performance
```
First configure took ≈ 10 min (vcpkg builds Boost, gtest, cppunit, tinyxml2, zlib; cached afterwards). `build/` is git-ignored.
