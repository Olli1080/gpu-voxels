// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-08-31
 *
 *
 */
//----------------------------------------------------------------------
#include "cuda_handling.h"
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels
{
    bool cuCheckForError(const char* file, int line)
    {
        cudaError_t cuda_error = cudaGetLastError();
        if (cuda_error != cudaSuccess)
        {
            LOGGING_ERROR(CudaLog, 
                cudaGetErrorString(cuda_error) << "(" << cuda_error << ") in " << file << " on line " << line << "." << endl);
            return false;
        }
        return true;
    }

    bool cuHandleError(cudaError_t cuda_error, const char* file, int line)
    {
        if (cuda_error != cudaSuccess)
        {
            LOGGING_ERROR(CudaLog,
                cudaGetErrorString(cuda_error) << " in " << file << " on line " << line << "." << endl);
            return false;
        }
        return true;
    }

    bool cuGetNrOfDevices(int* nr_of_devices)
    {
        return HANDLE_CUDA_ERROR(cudaGetDeviceCount(nr_of_devices));
    }

    bool cuGetDeviceInfo(cudaDeviceProp* device_properties, int nr_of_devices)
    {
        for (int i = 0; i < nr_of_devices; i++)
        {
            if (!HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&device_properties[i], i)))
                return false;
        }
        return true;
    }

    std::string getDeviceInfos()
    {
        int nr_of_devices;
        if (!cuGetNrOfDevices(&nr_of_devices))
            return {};

        if (nr_of_devices <= 0)
            return {};
        
        std::cout << "Found " << nr_of_devices << " devices." << std::endl;
        auto props = std::vector<cudaDeviceProp>(nr_of_devices);

        if (!cuGetDeviceInfo(props.data(), nr_of_devices))
            return {};

        std::stringstream tmp_stream;
        for (int i = 0; i < nr_of_devices; i++)
        {
            const auto& prop = props[i];
            tmp_stream << "Device Information of GPU " << i << std::endl
                << "Model: " << prop.name << std::endl
                << "Multi Processor Count: " << prop.multiProcessorCount << std::endl
                << "Global Memory: " << cBYTE2MBYTE * static_cast<float>(prop.totalGlobalMem) << " MB" << std::endl
                << "Total Constant Memory: " << prop.totalConstMem << std::endl
                << "Shared Memory per Block: " << prop.sharedMemPerBlock << " Shared Memory per Multi Processor: " << prop.sharedMemPerMultiprocessor << std::endl
                << "Max Threads per Block " << prop.maxThreadsPerBlock << " Max Threads per Multi Processor: " << prop.maxThreadsPerMultiProcessor << std::endl
                << "Registers per Block: " << prop.regsPerBlock << " Registers per Multi Processor: " << prop.regsPerMultiprocessor << std::endl
                << "Max grid dimensions: [ " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << " ]" << std::endl
                << "Max Block dimension: [ " << prop.maxThreadsDim[0] << ", " << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << " ]" << std::endl
                << "Warp Size: " << prop.warpSize << std::endl << std::endl;

            std::cout << "Dev " << i << " = " << tmp_stream.str() << std::endl;
        }
        return tmp_stream.str();
    }

    bool cuTestAndInitDevice()
    {
        // The test requires an architecture SM52 or greater (CDP capable).
        int device_count = 0, device = -1;
        cuGetNrOfDevices(&device_count);
        for (int i = 0; i < device_count; ++i)
        {
            cudaDeviceProp properties{};
            HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&properties, i));
            if (properties.major > 5 || (properties.major == 5 && properties.minor >= 2))
            {
                device = i;
                LOGGING_INFO(CudaLog, "Running on GPU " << i << " (" << properties.name << ")" << endl);
                break;
            }
        }
        if (device == -1)
        {
            std::cerr << "No device with SM 5.2 or higher found, which is required for GPU-Voxels.\n"
                << std::endl;
            return false;
        }
        cudaSetDevice(device);
        HANDLE_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
        //HANDLE_CUDA_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
        return true;
    }

    std::string getDeviceMemoryInfo()
    {
        std::stringstream tmp_stream;
        HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
        //unsigned int free, total, used;
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        const size_t used = total - free;

        tmp_stream << "Device memory status:" << std::endl;
        tmp_stream << "-----------------------------------" << std::endl;
        tmp_stream << "total memory (MB)  : " << static_cast<float>(total) * cBYTE2MBYTE << std::endl;
        tmp_stream << "free  memory (MB)  : " << static_cast<float>(free) * cBYTE2MBYTE << std::endl;
        tmp_stream << "used  memory (MB)  : " << static_cast<float>(used) * cBYTE2MBYTE << std::endl;
        tmp_stream << "-----------------------------------" << std::endl;

        return tmp_stream.str();
    }

} // end of namespace