# icl_core (vendored FZI logging/config library) is only used by the CUDA modules; the Vulkan backend does not depend on it.
if(GVL_BACKEND_CUDA)
  add_subdirectory(icl_core)
endif()
add_subdirectory(gpu_voxels)
