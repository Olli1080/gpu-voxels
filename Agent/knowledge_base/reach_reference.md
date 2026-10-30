# Reference Project: REACH (`C:\Code\REACH`)

The user's own, more modern C++20/Windows/MSVC project; use it as the model for **package organisation, CMake, tests, agent docs and Vulkan practice**. Read-only reference (separate repo; don't edit). Surveyed 2026-10-22; line references may drift.
Caveat: REACH uses private GitLab submodules (`externals/nexusbot`, `externals/ai3-overlays` at resy-gitlab.inf.uni-bayreuth.de) and a custom triplet `x64-windows-resy`. GPU-Voxels is a public GitHub fork — don't import private dependencies without asking (Q10).

## Repository organisation
- Root: slim `CMakeLists.txt` (≈ 35 lines) that `include()`s modules from `cmake/`: `Options.cmake` (before `project()`), `FindDependencies.cmake`, `Dependencies.cmake`, `Assets.cmake`, `Tools.cmake` (sccache/ccache, linker), `PchProducers.cmake`, `SourceGroups.cmake`, `ProjectFolders.cmake`, `GitInfo.cmake`. Then `add_subdirectory(src)`, optional `docs`, `tests`.
- Sources split into **`include/REACH/<module>/` (public headers)** and **`src/REACH/<module>/` (sources + CMakeLists)**; `src/REACH/CMakeLists.txt` lists modules in dependency order, optional ones behind options (`if(XDL_GUI) …`).
- Other top-level dirs: `assets/`, `docs/` (MkDocs + Doxygen), `examples/`, `externals/` (submodules), `legacy/`, `tests/`, `agent/` (agent docs), `.clang-format/.clang-tidy/.clangd`, `.github/`, `.claude/skills/`.
- GPU-Voxels' current layout (`packages/gpu_voxels/gpu_voxels/<module>/{include,src}`) is *module-local* instead; moving files is a large diff — see [package_layout_proposal.md](../plans_wip/package_layout_proposal.md).

## CMake conventions worth copying
1. **All `find_package` in `cmake/FindDependencies.cmake`**; module CMakeLists only `target_link_libraries(...)` against global targets. (GPU-Voxels violates this: `helpers/CMakeLists.txt` calls `find_package(CUDAToolkit|Stb|PCL)`.) Verify exported target names in the port's `*-config.cmake`; never guess.
2. **Options before `project()`; options drive vcpkg manifest features**: `Options.cmake` appends to `VCPKG_MANIFEST_FEATURES` (`gui`, `xr`, `kinect`, `voice`, …) so optional dependency sets are only installed when enabled. For GPU-Voxels: features `cuda` and `vulkan`, so the project configures on a machine without CUDA.
3. **Module schema:** `add_library(name)` (no forced STATIC/SHARED), `target_include_directories(PUBLIC $<BUILD_INTERFACE:…/include>)`, `target_sources(PRIVATE … PUBLIC FILE_SET HEADERS BASE_DIRS … FILES …)`, `target_compile_features(PUBLIC cxx_std_20)`, `target_link_libraries(PUBLIC …)`, `install(TARGETS … EXPORT …Targets FILE_SET HEADERS …)`. **No `file(GLOB)`**, explicit lists. (GPU-Voxels already uses the FILE_SET form.)
4. **VS niceties:** `ProjectFolders.cmake` (solution folders per target) and `reach_group_sources()` (source filters mirroring directories).
5. **Presets (version 7):** `Main` (VS 18 2026, vcpkg toolchain, triplet, binary cache dir), variants inheriting from it (`Main-XR`), Clang, `Clang-Ninja` (Ninja Multi-Config), no-PCH, IWYU. Build dirs `build/<preset>`.
6. **Compiler cache + PCH:** sccache/ccache auto-detected; shared PCH header sets with `REUSE_FROM`; `USE_PCH` option.
7. **Generated build info** (`GitInfo.cmake`) written only when content changes.
8. Parallel builds always (`cmake --build … --parallel`).

## Tests
- GoogleTest + CTest; **one wrapper `reach_add_test(name src…)`** (no raw `add_executable` for tests); `CONSOLIDATE_TESTS=ON` merges all cases into one `reach_tests_unified` binary (link time!).
- `tests/test_categories.cmake`: single registry of PERFORMANCE and REGRESSION filters; a consistency test fails if a pattern matches nothing. ctest skips performance by default; convention: a regression test's comment names the bug/commit.
- Fixtures under `tests/fixtures`, assets read from the source tree (no copying into every test dir). Baseline test count kept in docs.
- ⇒ For GPU-Voxels: adopt GoogleTest (answers Q13 in favour of GTest) and the same wrapper/category pattern.

## Vulkan practice (render backend) — reusable patterns
- Stack from vcpkg: `vulkan-headers/hpp` via `vulkan.hpp` RAII (`vk::Unique*`), **`vulkan-memory-allocator`** (VMA; adopted 2026-08-13 because per-resource `vkAllocateMemory` risks the ~4096 allocation cap), `vulkan-validationlayers` incl. **synchronization validation** (`VK_EXT_validation_features`, env switch), `vulkan-utility-libraries`, `glslang` (GLSL compiled to SPIR-V at runtime by `ShaderCompiler`; target env/version configurable).
- Device creation chain: `StructureChain<Features2, Vulkan11/12/13Features, …>` — query supported, enable subset, minimum API 1.1+.
- Compute example: `DepthUnprojector` (descriptor set layout with storage buffer, push constants struct, compute pipeline, device-local storage buffer). Also libfreenect2 fork with Vulkan depth processing (`externals/libfreenect2`).
- Rules recorded in `agent/agent_policies/vulkan_standards.md` and `agent/knowledge_base/vulkan_resources.md`: resource registry with explicit `shutdown_resource()` **before** device destruction; dedicated transfer queue + `eConcurrent` sharing; every async upload signals a fence; never share a command pool across threads; **external synchronisation for queue submit/fences (`transfer_mutex`)**; never call `device.waitIdle()` directly; zero-size allocation guards; split prepare (descriptor/UBO updates) from record phases; std140 padding (`vec4` instead of `vec3`).
- Incident history in `agent/knowledge_base/agent_regressions.md` — read it to see which refactors silently broke Vulkan rules (useful to anticipate similar mistakes).

## Agent-documentation conventions in REACH (for consistency)
`AGENTS.md` hub → `agent/{agent_policies,knowledge_base,project_state,plans_wip,plans_inactive,plans_stale,plans_done,analysis}` each with `summary.md`; policies include `cmake_build_standards.md`, `vulkan_standards.md`, `cpp_conventions.md`, `git_diff.md`, `guidelines.md`. `.claude/skills/` holds task recipes (`build-and-test`, `plan-doc-lifecycle`, …). GPU-Voxels' `Agent/` follows the six-folder scheme requested by the user; REACH's extra `plans_inactive/plans_stale` could be added on request.
