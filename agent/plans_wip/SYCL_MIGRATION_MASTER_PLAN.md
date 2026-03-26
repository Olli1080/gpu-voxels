# SYCL Migration Master Plan (AdaptiveCpp)

## Goal
Migrate the GPU-Voxels library from CUDA to SYCL using AdaptiveCpp to support multi-vendor hardware while maintaining high performance.

## Roadmap Overview
1.  **Preparation & Infrastructure**
2.  **Abstraction Layer for Parallel Algorithms (Thrust to oneDPL/SYCL)**
3.  **Core Data Structures Migration (Unified Shared Memory - USM)**
4.  **Kernel Migration (CUDA to SYCL Kernels)**
5.  **Build System Update (CMake with AdaptiveCpp)**
6.  **Validation & Performance Tuning**

---

## Sub-Plan 1: Preparation & Infrastructure
- [ ] **Infrastructure**: Set up a CI/CD environment with AdaptiveCpp and compatible hardware (Intel/AMD/NVIDIA).
- [ ] **Macro Bridge**: Create a header `SyclBridge.h` to abstract `__host__ __device__`, `__global__`, and CUDA error handling.
- [ ] **Test Harness**: Enhance existing tests to support side-by-side comparison between CUDA and SYCL implementations.

## Sub-Plan 2: Thrust to SYCL/oneDPL Migration
- [ ] **Iterative Step**: Identify a single `thrust::copy` or `thrust::transform` usage (e.g., in `TemplateVoxelList.cu`).
- [ ] **Snippet**: Replace with a SYCL-based parallel algorithm (oneDPL).
- [ ] **Verification**: Run unit tests for `VoxelList` to ensure bit-level parity.
- [ ] **Batch**: Gradually replace all Thrust calls with SYCL-equivalent algorithms.

## Sub-Plan 3: Memory Management (USM)
- [ ] **Abstraction**: Implement a SYCL-based memory manager that uses Unified Shared Memory (USM).
- [ ] **Iterative Step**: Port `GpuVoxelsMap` to use SYCL USM instead of `cudaMalloc`.
- [ ] **Verification**: Ensure host-to-device transfers and pointer accessibility work as expected in isolation.

## Sub-Plan 4: Kernel Migration
- [ ] **Complexity Analysis**: Categorize kernels by complexity (Simple vs. Shared Memory vs. PBA).
- [ ] **Simple Kernels**: Port basic kernels (e.g., `kernelClearVoxelMap`) to `sycl::parallel_for`.
- [ ] **Advanced Kernels**: Port complex kernels (PBA, Collision detection) using SYCL local memory and barriers.
- [ ] **Verification**: Each ported kernel must pass existing functional tests.

## Sub-Plan 5: Build System
- [ ] **CMake Integration**: Update `CMakeLists.txt` to use `find_package(AdaptiveCpp)`.
- [ ] **Conditional Compilation**: Support building both CUDA (original) and SYCL (new) versions for comparison during the transition.

## Sub-Plan 6: Validation
- [ ] **Functional Parity**: Run the full suite of `VoxelMapTests` and `VoxelListTests`.
- [ ] **Performance Benchmarking**: Compare SYCL performance on NVIDIA hardware against the original CUDA implementation.
- [ ] **Hardware Compatibility**: Test on non-NVIDIA hardware using AdaptiveCpp's backends.

---

## Iterative Testing Strategy
- Create a `sycl_port_sandbox` in `packages/gpu_voxels/gpu_voxels/test`.
- For every migrated component, implement a "Dual-Run" test that executes both CUDA and SYCL paths and compares results.
- Use `AdaptiveCpp`'s ability to run on CPU for initial debugging if GPU resources are limited.
