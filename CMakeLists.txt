cmake_minimum_required(VERSION 3.23)
include(GenerateExportHeader)
include(GNUInstallDirs)

project(GPU_VOXEL_TL
    LANGUAGES CXX CUDA)

set(NVCC_THREADS "1" CACHE STRING "Amount of threads for nvcc compilation")
add_compile_options(
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/MP>
)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -Xcudafe --diag_suppress=221 --threads=${NVCC_THREADS}")
if (WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/bigobj\"")
endif()

set(BUILD_TESTS ON)
if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(packages)