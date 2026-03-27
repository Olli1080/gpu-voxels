#ifndef GPU_VOXELS_HELPERS_ONEDPL_BRIDGE_H_INCLUDED
#define GPU_VOXELS_HELPERS_ONEDPL_BRIDGE_H_INCLUDED

#include <gpu_voxels/helpers/SyclBridge.h>

#ifdef GVL_USE_SYCL
    #include <oneapi/dpl/execution>
    #include <oneapi/dpl/algorithm>
    #include <oneapi/dpl/iterator>
    
    namespace gpu_voxels {
        namespace parallel = oneapi::dpl;
        
        // Define an execution policy using the bridge queue
        inline auto get_exec_policy() {
            return oneapi::dpl::execution::make_device_policy(sycl_bridge::get_default_queue());
        }
    }
#else
    #include <thrust/execution_policy.h>
    #include <thrust/copy.h>
    #include <thrust/transform.h>
    #include <thrust/fill.h>
    #include <thrust/count.h>
    #include <thrust/iterator/zip_iterator.h>
    #include <thrust/tuple.h>
    #include <thrust/device_vector.h>

    namespace gpu_voxels {
        namespace parallel = thrust;
        
        // In CUDA mode, we don't strictly need a special policy for most calls,
        // but we can define a placeholder if needed.
    }
#endif

#endif
