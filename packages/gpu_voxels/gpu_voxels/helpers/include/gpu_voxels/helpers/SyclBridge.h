#ifndef GPU_VOXELS_HELPERS_SYCL_BRIDGE_H_INCLUDED
#define GPU_VOXELS_HELPERS_SYCL_BRIDGE_H_INCLUDED

#if defined(__ACPP__) || defined(__HIPSYCL__) || defined(SYCL_LANGUAGE_VERSION)
    #define GVL_USE_SYCL
#endif

#ifdef GVL_USE_SYCL
    #include <sycl/sycl.hpp>
    
    // Attribute abstractions
    #define GVL_HOST_DEVICE
    #define GVL_DEVICE
    #define GVL_HOST
    #define GVL_GLOBAL

    // Memory qualifiers
    #define GVL_SHARED
    #define GVL_CONSTANT

    // Barrier abstraction
    #define GVL_SYNCTHREADS() /* SYCL needs item.barrier() */

    // Error handling abstraction
    #define GVL_HANDLE_ERROR(error) /* SYCL uses exceptions or async handlers */
    #define GVL_CHECK_ERROR()

    // Runtime API abstractions
    #define GVL_MALLOC(ptr, size) (*(ptr) = sycl::malloc_device((size), gpu_voxels::sycl_bridge::get_default_queue()))
    #define GVL_FREE(ptr) sycl::free((ptr), gpu_voxels::sycl_bridge::get_default_queue())
    #define GVL_MEMCPY(dst, src, size, kind) gpu_voxels::sycl_bridge::get_default_queue().memcpy((dst), (src), (size)).wait()
    #define GVL_MEMSET(ptr, value, size) gpu_voxels::sycl_bridge::get_default_queue().memset((ptr), (value), (size)).wait()
    #define GVL_SYNCHRONIZE() gpu_voxels::sycl_bridge::get_default_queue().wait()

    #define GVL_STREAM_CREATE(stream) /* SYCL uses queues */
    #define GVL_EVENT_CREATE(event) /* SYCL uses events from submissions */
    #define GVL_SET_DEVICE(dev) /* SYCL handles this via queue/device selection */

    // Kernel launch abstraction
    // Usage: GVL_LAUNCH_KERNEL(kernel_name, grid, block, shared_mem, stream, ...args)
    #define GVL_LAUNCH_KERNEL(kernel, grid, block, shm, stream, ...) \
        gpu_voxels::sycl_bridge::get_default_queue().parallel_for( \
            sycl::nd_range<3>(sycl::range<3>((grid).z * (block).z, (grid).y * (block).y, (grid).x * (block).x), \
                              sycl::range<3>((block).z, (block).y, (block).x)), \
            [=](sycl::nd_item<3> item) { \
                /* Mapping CUDA blockIdx/threadIdx to SYCL will require adjustments in the kernels themselves */ \
                kernel(__VA_ARGS__); \
            })

    // Memcpy kinds (placeholders for SYCL as it deduces from pointers or uses explicit overloads)
    #define GVL_MEMCPY_HOST_TO_DEVICE 0
    #define GVL_MEMCPY_DEVICE_TO_HOST 1
    #define GVL_MEMCPY_DEVICE_TO_DEVICE 2

    namespace gpu_voxels {
        namespace sycl_bridge {
            // Global queue for simple migration steps
            extern sycl::queue& get_default_queue();
        }
    }

#else
    #include <cuda_runtime.h>
    #include <gpu_voxels/helpers/cuda_handling.h>

    // Attribute abstractions
    #define GVL_HOST_DEVICE __host__ __device__
    #define GVL_DEVICE __device__
    #define GVL_HOST __host__
    #define GVL_GLOBAL __global__

    // Memory qualifiers
    #define GVL_SHARED __shared__
    #define GVL_CONSTANT __constant__

    // Barrier abstraction
    #define GVL_SYNCTHREADS() __syncthreads()

    // Error handling abstraction
    #define GVL_HANDLE_ERROR(error) HANDLE_CUDA_ERROR(error)
    #define GVL_CHECK_ERROR() CHECK_CUDA_ERROR()

    // Runtime API abstractions
    #define GVL_MALLOC(ptr, size) cudaMalloc((ptr), (size))
    #define GVL_FREE(ptr) cudaFree((ptr))
    #define GVL_MEMCPY(dst, src, size, kind) cudaMemcpy((dst), (src), (size), (kind))
    #define GVL_MEMSET(ptr, value, size) cudaMemset((ptr), (value), (size))
    #define GVL_SYNCHRONIZE() cudaDeviceSynchronize()

    #define GVL_STREAM_CREATE(stream) HANDLE_CUDA_ERROR(cudaStreamCreate(stream))
    #define GVL_EVENT_CREATE(event) HANDLE_CUDA_ERROR(cudaEventCreate(event))
    #define GVL_SET_DEVICE(dev) HANDLE_CUDA_ERROR(cudaSetDevice(dev))

    // Kernel launch abstraction
    #define GVL_LAUNCH_KERNEL(kernel, grid, block, shm, stream, ...) \
        kernel<<<grid, block, shm, stream>>>(__VA_ARGS__)

    // Memcpy kinds
    #define GVL_MEMCPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
    #define GVL_MEMCPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
    #define GVL_MEMCPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice

    namespace gpu_voxels {
        namespace sycl_bridge {
            // Mock or empty for CUDA-only builds
        }
    }

#endif

#endif
