# Backend language / compiler setup. Included after project().

if(GVL_BACKEND_CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
      message(FATAL_ERROR
        "GVL_BACKEND_CUDA is ON but no CUDA compiler (nvcc) was found.\n"
        "Install the CUDA toolkit, or configure without the CUDA backend: -DGVL_BACKEND_CUDA=OFF "
        "(CMake preset 'vulkan').")
    endif()
  endif()
  enable_language(CUDA)

  set(NVCC_THREADS "1" CACHE STRING "Amount of threads for nvcc compilation")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -Xcudafe --diag_suppress=221 --threads=${NVCC_THREADS}")
  if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/bigobj\"")
  endif()
  add_compile_definitions(GVL_BACKEND_CUDA=1)
endif()

if(GVL_BACKEND_VULKAN)
  add_compile_definitions(GVL_BACKEND_VULKAN=1)
endif()

if(NOT GVL_BACKEND_CUDA AND NOT GVL_BACKEND_VULKAN)
  message(STATUS "gpu_voxels: no compute backend selected; only backend-independent targets are built "
                 "(icl_core, gpu_voxels_logging, tests)")
endif()

message(STATUS "gpu_voxels backends: CUDA=${GVL_BACKEND_CUDA} VULKAN=${GVL_BACKEND_VULKAN}")
