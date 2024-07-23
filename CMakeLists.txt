cmake_minimum_required(VERSION 3.23)
include(GenerateExportHeader)
include(GNUInstallDirs)

project(GPU_VOXEL_TL
    LANGUAGES CXX CUDA)

set(BUILD_TESTS ON)
if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(packages)