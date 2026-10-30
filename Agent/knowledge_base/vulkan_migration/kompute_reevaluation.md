# Kompute Re-evaluation (2026-10-22)

Question from the user: Kompute was very new when the earlier attempts used it — has it matured enough to reconsider? Sources: GitHub API for `KomputeProject/kompute`, its README, the vcpkg snapshot bundled with REACH (`C:\Code\REACH\externals\vcpkg`, dated 2026-09-04), the in-house Vulkan code in REACH. Web data fetched 2026-10-22; treat as a snapshot.

## Facts (measured / fetched)
| Fact | Value |
|---|---|
| Latest **release** | v0.9.0, **2024-01-20** (Vulkan 1.3 upgrade, build refactor, storage tensors). Before that v0.8.1 (2022-04), v0.8.0 (2021-09). No release in the 21 months since |
| Development on `master` | Alive but slow: ~8 commits between 2026-04-13 and 2026-08-15 (thread-safety mutex option for Sequence/Manager, fmt include fixes, `setPushConstants` fix, docs). Mostly maintenance by few contributors |
| Project | ~2.6 k stars, 79 open issues, Apache-2.0, LF AI & Data sandbox project, not archived |
| API drift | Master differs from v0.9.0 (`kp::Memory` hierarchy, class renamings, issue #438 "example from docs does not compile against 0.9.0"). The old attempts pinned an unreleased commit (`6e5380de…`) and used `kp::Memory` — i.e. they lived on master, not on a release |
| vcpkg | **No `kompute` port** in the 2026-09-04 vcpkg snapshot (`ports/` has vulkan-headers/-hpp/-loader/-validationlayers/-memory-allocator/-utility-libraries, volk, glslang, shaderc, spirv-*, directx-dxc, but no kompute). The earlier attempt needed a hand-written overlay port |
| Open feature/perf issues | #5 only storage buffers (no uniform etc.), #420 "memory copies from device significantly slower than CUDA", #442 request for ShaderModule/Pipeline cache, #443 request to integrate VMA, #463 semaphore-aware async overloads, #475 image samplers. None about buffer-device-address or 64-bit |
| Programming model | Manager / Tensor / Algorithm / Sequence / Ops. Sound for experiments, but pushes "one sequence per operation, eval = submit+wait" unless the code builds sequences deliberately; allocator is per-tensor (no sub-allocation) |

## Assessment
- **Not substantially more mature**: no release since January 2024; master still churns and the API you would build against is the unreleased one. Maintenance activity is real but small.
- **Packaging gap persists**: you would again carry an overlay port pinned to a commit.
- **Capability gap for this project**: needs sub-allocation (many buffers/maps), batched command buffers, barriers between dependent kernels (PBA phases, scan/sort chains), push constants of arbitrary POD, specialization constants, possibly buffer device address / 64-bit — all reachable only by working around or bypassing Kompute's abstractions. Most failures of the earlier attempts were process/design (see [mistakes_to_avoid.md](mistakes_to_avoid.md)), not Kompute defects, but Kompute's convenience API made the "CUDA-emulation" design easy to fall into.
- **In-house alternative exists**: `C:\Code\REACH` already ships a Vulkan stack on current vcpkg ports — `vulkan.hpp` (RAII `vk::Unique*`), `vulkan-memory-allocator` (VMA), `vulkan-validationlayers` with synchronization validation, `glslang`, a compute pipeline (`depth_unprojector.cpp`, `depth_unproject.comp`), staging-buffer pool, transfer-queue fences, and written lessons about host synchronization and resource shutdown ([reach_reference.md](../reach_reference.md)). Reusing these *patterns* is cheaper and better understood than adopting Kompute.

## Recommendation (judgement — user decides, Q2)
**Thin own compute runtime on `vulkan.hpp` + VMA** (+ validation layers), behind the small buffer/dispatch interface described in [design_options.md](design_options.md) D2. Keep Kompute only as a time-boxed comparison in step P0.5 of the draft plan (same kernel, count LOC, dispatch overhead, validation messages), if you want evidence rather than argument.
Revisit if: Kompute publishes a release ≥ 0.10 with a vcpkg port, VMA integration (#443) and pipeline cache (#442) land.
