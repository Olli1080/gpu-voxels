# Initial Project Analysis

## Overview
The GPU-Voxels project is a CUDA-based library for high-resolution 3D occupancy mapping and collision detection. It heavily relies on NVIDIA CUDA and the Thrust library for parallel computations on the GPU.

## Key Technologies
- **CUDA**: Used for core computations and memory management.
- **Thrust**: Extensively used for parallel algorithms (copy, transform, fill, count_if, copy_if).
- **C++**: The project uses modern C++ (C++17/20 based on `std::scoped_lock` usage).
- **CMake**: Used for the build system.
- **vcpkg**: Used for dependency management.

## Current Structure (Core)
- **Consistent Naming**: The project follows a consistent naming scheme:
    - `.h` / `.cu`: Non-templated CUDA sources.
    - `.hpp` / `.cuhpp`: Templated headers/implementations, with `.cuhpp` indicating CUDA-specific template code.
- `packages/gpu_voxels`: Contains the main library logic.
- `packages/icl_core`: A core utility library.
- `example_how_to_link`: Demonstrates how to use the library.
- `gvl_ompl_planning`: Integration with OMPL.

## Observations for SYCL Migration
- **Thrust Usage**: Since Thrust is heavily used, migrating to **oneDPL** (part of the oneAPI/SYCL ecosystem) is a natural path.
- **CUDA Kernels**: Extensive use of custom kernels (100+ occurrences of `<<<` and `__global__` identified). Significant logic resides in `VoxelMapOperations.cu`, `DistanceVoxelMap.cu`, and various kernel headers.
- **Memory Management**: Uses `cudaDeviceSynchronize` and `HANDLE_CUDA_ERROR` macros. These will need to be replaced by SYCL queue management and exception handling.
- **Template Instantiations**: The project uses explicit template instantiations in `.cu` files. This pattern needs to be adapted for SYCL (e.g., using `explicit template instantiation` in SYCL source files).

## Obvious Flaws / Improvement Areas
- **Outdated Build Files**: `CMakeLists_old.txt` suggests some legacy configurations are still present.
- **Hardcoded Constants**: Constants like `BIT_VECTOR_LENGTH` seem to be pervasive and might limit flexibility.
- **Complexity in `TemplateVoxelList`**: The file `TemplateVoxelList.cu` is quite complex and handles many cases via switch-cases on map types. This could be refactored to be more generic or use policy-based design.

## Future Steps
- Detailed mapping of CUDA kernels to SYCL equivalent `parallel_for`.
- Evaluate AdaptiveCpp (formerly hipSYCL) compatibility with existing Thrust usage.
- Map CUDA memory management patterns to SYCL Unified Shared Memory (USM).
