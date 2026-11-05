# Central dependency discovery for the gpu_voxels packages. Included after Backends.cmake.
#
# Rules (see Agent/knowledge_base/reach_reference.md):
#  * every find_package() for packages/gpu_voxels lives here, module CMakeLists only link targets;
#  * results are created at top-level scope, so they are visible in all subdirectories;
#  * never guess a vcpkg target name: check <port>-config.cmake of the installed port.
# packages/icl_core is vendored third-party code, built only with the CUDA backend, and still performs its own discovery
# (ZLIB, tinyxml2, Boost, CppUnit). The Vulkan backend does not use it.

# ---- CUDA backend -----------------------------------------------------------
if(GVL_BACKEND_CUDA)
  find_package(CUDAToolkit REQUIRED)
  find_package(Stb REQUIRED)
  if(GVL_WITH_PCL)
    find_package(PCL REQUIRED COMPONENTS io)
  endif()
endif()

# ---- Vulkan backend (dependencies only; no runtime yet) -----------------------
if(GVL_BACKEND_VULKAN)
  find_package(Vulkan REQUIRED)

  # Vulkan Memory Allocator: vcpkg port 'vulkan-memory-allocator' (feature "vulkan").
  # NOTE: the Vulkan SDK 1.4.363 does not ship VMA, despite its description.
  find_package(VulkanMemoryAllocator CONFIG)
  if(NOT VulkanMemoryAllocator_FOUND)
    message(WARNING "VulkanMemoryAllocator not found: configure through the vcpkg toolchain (feature 'vulkan')")
  endif()

  # Window of the in-process viewer: GLFW (vcpkg port 'glfw3', feature "vulkan-viewer"). The viewer core needs no window system.
  if(GVL_VULKAN_VIEWER)
    find_package(glfw3 CONFIG REQUIRED)
  endif()

  # Offline GLSL -> SPIR-V compiler (shaders are compiled at build time).
  find_program(GVL_GLSLANG_VALIDATOR glslangValidator
               HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
  if(GVL_GLSLANG_VALIDATOR)
    message(STATUS "glslangValidator: ${GVL_GLSLANG_VALIDATOR}")
  else()
    message(WARNING "glslangValidator not found: install the Vulkan SDK (VULKAN_SDK) or vcpkg glslang[tools]")
  endif()
endif()

# ---- Tests ------------------------------------------------------------------
if(BUILD_TESTS)
  find_package(GTest CONFIG REQUIRED)
endif()
