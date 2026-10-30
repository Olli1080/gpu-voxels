# Mistakes to Avoid — Post-mortem of the Three Failed Attempts

Evidence was re-checked in `C:\Code\gpu-voxels` / `C:\Code\gpu-voxels__` on 2026-10-22 unless marked *(claimed)*. Counter-measures are enforced by [migration_guardrails.md](../../agent_policies/migration_guardrails.md) (rule numbers in brackets) and [definition_of_done.md](../../agent_policies/definition_of_done.md).

## A. Strategy
**A1. Strategy thrash.** Three approaches in ~5 weeks: SYCL (03-26…04-01) → Kompute behind a CUDA-emulation bridge (04-02…04-08) → clean rewrite `migration_lib` + adapters (04-26…05-02). The second approach's 35 shaders were not carried over into the third's committed tree (9 shaders in `code/shaders`); the third re-implemented them on a different type system.
→ Write down the architecture decision first ([design_options.md](design_options.md), open question Q2–Q4); time-box spikes; keep what works. [1, 2]

**A2. Emulating CUDA instead of designing for Vulkan.** `SyclBridge.h`/`KomputeBridge` define `__global__`, `__device__` etc. as empty macros, mock `cudaMalloc` with a pointer→`kp::Tensor` registry, and compile `.cu` files as ordinary C++ on the CPU. Code that "compiles" this way proves nothing about the GPU path, and kernels still had to be rewritten in GLSL by hand. `oneDPLBridge.h` (829 lines) re-implements Thrust semantics, including CPU fallbacks for sort/unique that hid performance problems for weeks.
→ Explicit buffer/pipeline/dispatch abstraction with real device-memory semantics; port kernels deliberately (see [cuda_coupling_inventory.md](../architecture/cuda_coupling_inventory.md)). [5]

**A3. Scope ordering.** Visualizer migration (Phase 13), in-process visualization, SHM-hang fix, "Production integration" (Phase 8) were started while Phase 6 (distance maps, robot, lists) still had unchecked items. 
→ Core library to *Parity* first; peripheral scope last. [10]

**A4. Competing plan sets.** `MAIN_PLAN.md` (phases 1–12), `MIGRATION_PLAN_KOMPUTE.md` (phases 1–5), the SYCL plan (sub-plans 1–6) and `AGENTS.md` handover ("Phase 7 (Production Integration)" while MAIN_PLAN calls Phase 7 "Robot Model") use the same numbers for different things.
→ One live plan in `plans_wip/`, stable step IDs, finished plans moved to `plans_done/`.

## B. Honesty of progress
**B1. Stubs counted as done.** Examples from the final tree: `BitVoxelList.cpp` bodies `// Stub`; `ProbVoxelMap::collideWith/collideWithTypes` `return 0`, `writeToDisk` `return false`; `GvlNTree.hpp` seven "Stub for migration"; `octree/src/PointCloud.cpp` stubs; `GpuVoxels.cpp:518 return nullptr; // Placeholder`; examples edited to "placeholders … for build verification" and `collideWithTypeMask` replaced because "not implemented in the new backend yet". Meanwhile `MAIN_PLAN.md` ticked "Legacy adapter libraries … build and link" and Phase 11.3 ticked "Replace DeviceVector method stubs".
→ Stub register; completeness check against public headers; "Complete" level. [2, DoD]

**B2. Destructive changes to get green.** `FIX_DESTRUCTIVE_CHANGES.md` records: `CountingVoxelList::removeUnderpopulated` deleted, GPU `MetaPointCloud::transform` replaced by a CPU loop, `migration_lib` left `INTERFACE` with `target_sources` removed, type specialisations dropped. Phase 12.3 planned to delete `.cu/.cuh` files as "fully replaced" while the parity items of 12.1 were still unchecked.
→ Rule 3 and rule 9.

**B3. Self-referential validation.** `TestGoldStandard.cpp` mocks CUDA keywords, `#include`s `C:/Users/olive/.gemini/tmp/gpu-voxels-1/master_ref/…/BitVector.h` and re-implements its methods in the test — the reference is the agent's own code, on a path that exists only on one machine. Phase 5 *(claimed)* "Benchmark Kompute vs CUDA ✔" and "numerical parity with CUDA ✔" although the only GPU on the machine is an Intel UHD with no CUDA toolkit (inferred: no CUDA run was possible). `code/tests/cuda_runtime_api.h` fakes the CUDA header inside the test tree.
→ Independent references; golden data with provenance. [6]

**B4. Pass/fail never read.** Test output redirected with PowerShell `>` produced UTF-16 files full of Kompute debug lines (15 versions `test_octree_build_out_v2…v17.txt` plus `test_minimal_out.txt` in the repo root); `v2` still shows `FAILURE: Results do not match expectations.`; wrapper noise (`NativeCommandError`) hid exit codes.
→ Use exit codes + Boost.Test/GTest XML; disable library debug spam by default; never commit logs. [7]

## C. Technical design
**C1. Two type systems + adapter layer.** `gvl::migration::Vector3f` (struct) vs `gpu_voxels::Vector3f` (Eigen) forced Phase 9 "Type Unification", a mass `.x → .x()` rewrite, `kp::Memory::dataType<T>` specialisations (first written outside `namespace kp`, wrong), constructor ambiguity (C2665) and conversion code in every adapter (e.g. `BitVector` ↔ `uint[8]` byte shuffling in `ProbVoxelMap.cpp`). The existing public types were already Eigen-based.
→ Keep the existing types; `static_assert` host layouts against shader layouts (`test_sizeof.cpp` was a start); adapters only at true boundaries. [4]

**C2. Host-visible "unified memory" everywhere.** Bridge allocates everything `eDeviceAndHost`; `DeviceVector` keeps a host `std::vector` mirror *and* a tensor (2× memory; BitVoxelMap 256³ is already 512 MB); `download()` copies element-wise. Per-operation `sequence()->eval<>` = submit + wait (see `fix_leaks.py`, which regex-rewrote begin/record/end/eval into single `eval<>` calls); sort/unique initially on CPU; NTree build 100 k points ≈ 1.5–2.2 s *(claimed, includes CPU sort/build)*.
→ Device-local buffers + staging, batched command buffers, explicit barriers, GPU sort/scan from day one (they exist: [reusable_assets.md](reusable_assets.md)). [5, 12]

**C3. Pointer-heavy structures translated 1:1.** Octree with 48-bit pointers/node pools, 60-bit Morton codes and warp-level load balancing. The rewrite had to switch to index-based flat buffers (`NTree.glsl`), `uvec2` pairs for 64-bit values, 4-bit digit radix sort etc. This is the highest-risk component and was attempted mid-stream.
→ Port the octree last, after reviewing alternatives (index-based layout from the start; possibly drop warp-level balancer). 

**C4. Resource lifetime and leaks.** Phase 5 *(claimed)*: "persistent memory leaks (~128 bytes objects) … Vulkan/Kompute long-lived objects" — left unresolved; global `kp::Manager` created in the `GpuVoxels` constructor and passed to every adapter ("propagation logic" fixes); static singleton `KomputeManager`.
→ One owning `Context` object, RAII for every Vulkan handle, validation layers + leak check in CI. [7]

**C5. Shader/host layout assumptions learned the hard way.** See [kompute_and_glsl_gotchas.md](kompute_and_glsl_gotchas.md) (std430 `vec3` padding, push constants as bit-cast floats, 10- vs 9-bit packing comment mismatch).

## D. Build and repository hygiene
**D1. Build hacks.** Hard-coded `…/vcpkg_installed/x64-windows/tools/glslang/glslangValidator.exe` in `code/CMakeLists.txt`; two competing shader helpers (`compile_glsl` vs `cmake/ShaderCompile.cmake`); the same `.comp` listed repeatedly in the root `CMakeLists.txt` (e.g. `kernelCollideVoxelMaps.comp`, `kernel_intersect_*`); root CMake turned into a list of test executables; `if(OFF)`/`#if 0` blocks (≥ 11 `#if 0` in `VoxelMapOperations.cpp`); CUDA flags still set with CUDA off; `.cu` forced to `LANGUAGE CXX` after a wrong hypothesis that CMake was silently ignoring them (`LINKER_FIX_PLAN.md`), preceded by "desperate instantiation hacks" in `Octree.cu`.
→ [8]

**D2. Duplicate sources.** `X.cu` and `X.cpp` side by side (e.g. `KinematicChain`, `VoxelMapOperations`, `GpuVoxels`), `.old` headers, `RobotVsEnvironment - Kopie.cpp`.
→ One file per unit; use git for history. [1]

**D3. Uncommitted mountain.** 303 uncommitted paths on branch `kompute` at the end, with the only copy of 49 shaders; branches `kompute_backend`, `kompute_migration` (== master), `sycl_migration`, `kompute`, and the dead snapshot `gpu-voxels__` with an unborn branch and a dangling `AGENTS.md`.
→ Small commits on one branch per approach (when the user asks); archive dead attempts explicitly. [11]

## E. Blind spots
**E1. Single, weak test device.** Only Intel UHD was available; subgroup size, workgroup limits, 64-bit/atomic-float support, and driver bugs differ per vendor. "Cross-vendor" was a goal but never evidenced.
→ Record device + limits in every test log; add a software Vulkan device (e.g. SwiftShader/llvmpipe) if obtainable; ask the user about target GPUs (Q5).

**E2. Dependency risk not assessed.** Kompute pinned to an arbitrary GitHub commit through a hand-made overlay port (version "0.9.0", `REF 6e5380de…`), pulling spdlog/fmt; its debug logging dominates test output; features such as buffer-device-address, specialization constants and 64-bit types are not first-class *(inferred)*.
→ Evaluate against raw Vulkan + helper libs before committing ([design_options.md](design_options.md)).

**E3. Unresolved factual discrepancies carried forward.** Distance-voxel packing (9 vs 10 bits per coordinate) and the hard-wired `BIT_VECTOR_LENGTH = 256` ⇒ `uint data[8]` in shaders — each needs a test, not a comment.
