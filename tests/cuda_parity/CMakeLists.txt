# CUDA side of the parity checks (only with GVL_BACKEND_CUDA). The programs print results that tests/vulkan/test_cuda_parity.cpp compares against
# (golden files in golden/). Run them by hand; they are not CTest cases (they need a CUDA device and are only used to record the golden files).
add_executable(cuda_collisions_dump cuda_collisions_dump.cu)
target_link_libraries(cuda_collisions_dump PRIVATE gpu_voxels)
target_compile_features(cuda_collisions_dump PRIVATE cxx_std_20)
set_property(TARGET cuda_collisions_dump PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_collisions_dump PROPERTY CUDA_ARCHITECTURES ${GVL_CUDA_ARCHITECTURES})
