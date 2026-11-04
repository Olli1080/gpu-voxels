# Viewer core of the Vulkan backend: GPU extraction of cubes from maps, lists, octrees, primitives and robots, the renderer (dynamic rendering, offscreen images,
# PNG files), the orbit camera, the scene with its named layers over a GpuVoxels and the XML style file. No window system, no GLFW: it works headless (tests,
# screenshots). The window is gpu_voxels_vulkan_viewer. gpu_voxels_vulkan itself contains no viewer code.
project(gpu_voxels_vulkan_view
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(INCLUDE_DIR include/gpu_voxels/vulkan_view)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
  PRIVATE
    src/ImageFile.cpp
    src/OrbitCamera.cpp
    src/Viewer.cpp
    src/ViewerConfig.cpp
    src/ViewerScene.cpp
  PUBLIC FILE_SET HEADERS
  BASE_DIRS ${INCLUDE_DIR}
  FILES
    ${INCLUDE_DIR}/ImageFile.h
    ${INCLUDE_DIR}/OrbitCamera.h
    ${INCLUDE_DIR}/Viewer.h
    ${INCLUDE_DIR}/ViewerConfig.h
    ${INCLUDE_DIR}/ViewerScene.h
)

find_package(tinyxml2 CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    gpu_voxels_vulkan
  PRIVATE
    GPUOpen::VulkanMemoryAllocator
    tinyxml2::tinyxml2
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

# Kernels of the viewer: GLSL -> SPIR-V at build time, embedded into the library. viewer_extract_octree.comp includes morton.glsl of the core library.
gvl_embed_shaders(${PROJECT_NAME}
  INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders
    ${CMAKE_CURRENT_SOURCE_DIR}/../vulkan/shaders
  SOURCES
    shaders/viewer_cube.frag
    shaders/viewer_cube.vert
    shaders/viewer_extract_bit.comp
    shaders/viewer_extract_list.comp
    shaders/viewer_extract_octree.comp
    shaders/viewer_extract_points.comp
    shaders/viewer_extract_prob.comp
    shaders/viewer_finalize.comp
    shaders/viewer_sphere.frag)

install(
  TARGETS
    ${PROJECT_NAME}
  EXPORT
    gpu_voxels-targets
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${INCLUDE_DIR}
)
