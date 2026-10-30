# Migration Guardrails (CUDA → Vulkan)

Each rule exists because a previous attempt violated it. Evidence and detail: [mistakes_to_avoid.md](../knowledge_base/vulkan_migration/mistakes_to_avoid.md).

| # | Rule | Failure it prevents |
|---|---|---|
| 1 | **One strategy per branch.** Do not mix a macro-bridge ("compile `.cu` as C++") with a clean rewrite plus adapters in the same tree. | `C:\Code\gpu-voxels` working tree mixes both: duplicate `X.cu` + `X.cpp`, 49 untracked shaders next to 9 committed ones |
| 2 | **Vertical slices.** Take one component (e.g. `BitVoxelMap` + insert + collide) through *Compiles → Runs → Tested → Parity* before starting the next. No horizontal "stub everything so it links" passes. | Phase 8 declared adapters "implemented" while bodies were `// Stub` |
| 3 | **No destructive changes.** Never remove or empty a feature, test or public method to get green. `removeUnderpopulated`, GPU transforms, `STATIC` library target were all removed once and needed a `FIX_DESTRUCTIVE_CHANGES.md` plan to repair. | Silent feature loss |
| 4 | **Keep the public API stable first.** `GpuVoxels`, `GpuVoxelsMap`, `AbstractVoxelMap/List`, voxel types and enums (`MapType`, `BitVoxelMeaning`) keep their signatures until a plan explicitly changes them. Replace `thrust::device_vector` in signatures through a deliberate, documented step. | Two parallel type systems (`gvl::migration::*` vs `gpu_voxels::*`) needing "Phase 9: Type Unification" |
| 5 | **No fake pointer model.** Do not emulate `cudaMalloc`/raw device pointers with a pointer→tensor registry and host-visible "unified" memory and call it a backend. Design explicit buffer objects with explicit upload/download/barrier semantics. | `gvl_kompute_malloc` registry; everything `eDeviceAndHost`; host mirrors doubling memory |
| 6 | **Independent reference.** Parity tests compare against (a) golden data from the real CUDA build, or (b) a straightforward CPU implementation written *from the specification*, kept in `tests/reference/`. Never against the code under test or a mocked CUDA header. | Self-referential `TestGoldStandard.cpp` |
| 7 | **Validation layers on, zero errors.** Every test run on Vulkan uses `VK_LAYER_KHRONOS_validation` when available; errors fail the test. | Debug logs ignored; leaks of ~128-byte objects noted but never resolved |
| 8 | **Build must work from a clean checkout** with documented commands: no absolute user paths, no hard-coded `vcpkg_installed/x64-windows/tools/...`, shaders found via CMake. | `GLSLANG_VALIDATOR` hard-coded path; `C:/Users/olive/.gemini/...` include |
| 9 | **Keep CUDA buildable until the user says otherwise** (user decision 2026-10-22: CUDA stays "for now" to allow comparison). A Vulkan backend is added behind an option; do not delete `.cu` files in the same change that introduces their replacement. | Large deletions (-20 k lines) before parity existed |
| 10 | **Defer peripheral scope.** Visualizer, ROS, Kinect, OMPL, examples, in-process visualization come *after* the core library has Parity. | Phases 13/"in-process visualization"/SHM-hang fixes started with core incomplete |
| 11 | **Small, reviewable commits** (allowed on the work branch since 2026-10-29). Never leave hundreds of files uncommitted across sessions. | 303 uncommitted paths in the last working tree |
| 12 | **Measure before optimising, and measure on the right device.** Record device, driver, build type, input size with every number. | "Clear 256³ ≈ 88 ms", "NTree 100 k points ≈ 1.5–2.2 s" were reported without a comparison baseline (presumably on the Intel UHD, the machine's only GPU) |
| 13 | **Don't trust docs of prior attempts for facts about the code**; re-verify (`grep`, build, run). Treat them as leads. | Plans contradicting code state |
| 14 | **Document each non-obvious finding** in `knowledge_base/` the same day (layout/alignment rules, driver quirks, tool flags). | Re-discovering `std430` padding, `#extension GL_GOOGLE_include_directive` etc. |
