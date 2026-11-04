# Window part of the in-process Vulkan viewer (GLFW) and the gvl_viewer program. The viewer core (extraction of cubes, renderer, offscreen target, orbit
# camera) is part of gpu_voxels_vulkan and needs no window system. No separate viewer process, no Boost.Interprocess.
project(gpu_voxels_vulkan_viewer
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/vulkan_viewer)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
  PRIVATE
    src/ViewerWindow.cpp
  PUBLIC FILE_SET HEADERS
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/ViewerWindow.h
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    gpu_voxels_vulkan
  PRIVATE
    glfw
    GPUOpen::VulkanMemoryAllocator
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# The program: a window (or --png: an offscreen image) with a demo scene or a point cloud file.
add_executable(gvl_viewer src/viewer_main.cpp)
target_link_libraries(gvl_viewer PRIVATE ${PROJECT_NAME})
target_compile_features(gvl_viewer PRIVATE cxx_std_20)

install(
  TARGETS
    ${PROJECT_NAME}
    gvl_viewer
  EXPORT
    gpu_voxels-targets
  LIBRARY
    DESTINATION lib
  RUNTIME
    DESTINATION bin
  FILE_SET HEADERS
    DESTINATION ${INCLUDE_DIR}
)
