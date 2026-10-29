cmake_minimum_required(VERSION 3.23)

# Options must be known before project(): they select the vcpkg manifest features.
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Options.cmake")

# Languages: C++ only here. CUDA is enabled by cmake/Backends.cmake when GVL_BACKEND_CUDA is ON.
project(GPU_VOXEL_TL
    LANGUAGES CXX)

include(GenerateExportHeader)
include(GNUInstallDirs)

include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Backends.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindDependencies.cmake")
if(GVL_BACKEND_VULKAN)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Shaders.cmake")
endif()

add_compile_options(
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<COMPILE_LANGUAGE:CXX>>:/MP>
)

if(BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(packages)

if(BUILD_TESTS)
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Tests.cmake")
    add_subdirectory(tests)
endif()
