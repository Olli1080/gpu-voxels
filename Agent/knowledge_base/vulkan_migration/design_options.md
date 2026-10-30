# Design Options for the New Vulkan Attempt

**Status: partly decided.** Choices marked ► are the agent's recommendation (judgement, not measurement); the user decides — see [open_questions.md](../../project_state/open_questions.md). Record decisions as `plans_wip/ADR-<n>.md` (or inside the active plan).
Decided 2026-10-22: **D1 → CUDA stays for now (backend behind interface)**; target GPUs: all vendors (feature policy in the Q4 answer). **D2 decided: own runtime on `vulkan.hpp` + VMA** (user, after [kompute_reevaluation.md](kompute_reevaluation.md)), patterned on REACH ([reach_reference.md](../reach_reference.md)). D6: GoogleTest.

## D1. Relationship to CUDA
| Option | Pros | Cons |
|---|---|---|
| ► Backend behind an interface; CUDA kept until Vulkan has Parity, then removed deliberately | Safe; enables side-by-side golden data (needs an NVIDIA machine) | Interface design effort; two code paths for a while |
| Replace in place (delete `.cu` as you go) | Smaller tree | No reference to compare to; this machine cannot run CUDA anyway; repeats attempt 3's destructive pattern |
| Single-source kernels (e.g. Slang → CUDA + SPIR-V) | One kernel source, both backends | Tooling risk, rewrite of all kernels once; evaluate in a spike |

## D2. Vulkan access layer
| Option | Pros | Cons |
|---|---|---|
| Kompute (what failed before) | Tiny API (Manager/Tensor/Algorithm/Sequence), quick start | Pinned to an unreleased commit via custom overlay port; sequence-per-op model; log spam; `eDeviceAndHost` default encouraged the "unified memory" trap; limited access to BDA/64-bit/subgroup features |
| ► Thin own runtime on raw Vulkan (`vulkan.hpp`/`volk`) + VMA | Full control: device-local buffers + staging, timeline semaphores, barriers, batched command buffers, specialization constants, BDA, validation layers; small surface (est. 1–2 kLOC) | More upfront work; must write error handling and lifetime management carefully |
| Other portable layers (WebGPU/Dawn/wgpu, rendering-hardware-interface libs) | Portable | WebGPU lacks 64-bit atomics, BDA, guaranteed subgroups — conflicts with octree/Morton needs |
Decision driver: a **device-buffer + dispatch + barrier** abstraction of ~10 calls (`create_buffer`, `upload`, `download`, `dispatch(pipeline, bindings, push, groups)`, `barrier`, `submit/wait`, `query_limits`) that tests can use directly. Whichever binding is chosen must sit behind it.

## D3. Shader language / toolchain
| Option | Pros | Cons |
|---|---|---|
| GLSL (`glslangValidator`, vcpkg `glslang[tools]`) | Already used; ~50 existing shaders; mature | No templates/generics ⇒ per-Voxel/Collider variants by macros or codegen; weak modularity |
| HLSL (DXC → SPIR-V) | C++-like, supports templates (HLSL 2021), good tooling | Another toolchain (`directx-dxc` port), Vulkan-specific attributes |
| Slang | Generics/interfaces ≈ the C++ `Voxel`/`Collider` templates; targets SPIR-V and CUDA | Younger ecosystem; not installed here; spike needed |
| ► Start with GLSL for the first slice, decide before the second family of kernels (voxel lists) | Low risk | Possible rework |
Build rule regardless: shaders compiled by CMake custom commands to `.spv`, found via a CMake-provided directory or embedded as byte arrays; no absolute paths (guardrail 8).

## D4. Memory model
- Maps as one storage buffer each (index = x + y·dx + z·dx·dy as in CUDA). Check `maxStorageBufferRange`; above it split the map or use BDA. `uint32` indices cap maps at 4.29·10⁹ voxels.
- Host access only through explicit upload/download/staging; **no** host mirror in the map classes.
- Public API that exposes `thrust::device_vector` (e.g. `insertCoordinateList(const thrust::device_vector<Vector3ui>&)`) needs a replacement type (`DeviceBuffer<T>` handle). Introduce it as a deliberate API change behind an alias on the CUDA side first.
- Voxel payload structs: define once in a shared header usable from C++ and GLSL (via `#include` of a layout-neutral `.glsl` generated or hand-synchronised + `static_assert` tests).

## D5. Primitives first
Implement and unit-test (against `std::` references) before any voxel code: fill, copy, transform, reduce, exclusive/inclusive scan, radix sort (+ by key), stream compaction (`copy_if/remove_if`), unique (+ by key), binary search, min/max. They replace ~all Thrust usage ([cuda_coupling_inventory.md](../architecture/cuda_coupling_inventory.md)). Existing shaders in [reusable_assets.md](reusable_assets.md).

## D6. Tests and references
- Framework: one of Boost.Test (already in master's `vcpkg.json`), GoogleTest (was used in attempts), Catch2. Pick one.
- References: (1) simple CPU implementations under `tests/reference/` derived from the CUDA code; (2) golden files produced by a CUDA run on a user-provided NVIDIA machine, stored with provenance (commit, GPU, driver); (3) analytic cases (known geometry).
- Run on every available Vulkan device incl. a software rasterizer if installable; log device name + limits.

## D7. Visualization (separate track, last)
Options: keep OpenGL viewer and feed it host-side extracted cubes; in-process Vulkan debug renderer; Vulkan external memory sharing with the existing viewer process. See [visualization_and_ipc.md](../architecture/visualization_and_ipc.md). Not needed to prove the compute port.

## D8. Scope cut candidates (ask the user)
ROS/tf helpers, Kinect/OpenNI, OMPL planner example, URDF robot, Doxygen docs, Python helper scripts. Anything not needed by the user's actual use of the library can be deferred or dropped.
