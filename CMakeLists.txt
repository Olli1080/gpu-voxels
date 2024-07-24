cmake_minimum_required(VERSION 3.23)
include(GenerateExportHeader)
include(GNUInstallDirs)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr -rdc=true")
if (WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler \"/bigobj\"")
    #add_compile_options(/bigobj)
endif()


project(GPU_VOXEL_TL
    LANGUAGES CXX CUDA)

set(BUILD_TESTS ON)
if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(packages)