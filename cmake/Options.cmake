# Project options. Included BEFORE project() so that the options can select the
# vcpkg manifest features (the vcpkg toolchain reads VCPKG_MANIFEST_FEATURES at project()).

# ---- Compute backends -------------------------------------------------------
option(GVL_BACKEND_CUDA "Build the CUDA backend (requires the CUDA toolkit and nvcc)" ON)
option(GVL_BACKEND_VULKAN
       "Vulkan backend: currently only locates the Vulkan dependencies (runtime not implemented yet)" OFF)

# ---- CUDA architectures (default: every architecture of the toolkit, as before; "89" builds only the RTX 4090) ----
set(GVL_CUDA_ARCHITECTURES "all" CACHE STRING "CUDA_ARCHITECTURES of the gpu_voxels targets (all, native or a list such as 89)")

# ---- In-process Vulkan viewer (only with GVL_BACKEND_VULKAN; gpu_voxels_vulkan itself has no viewer code) ----
option(GVL_VULKAN_VIEW "Build the viewer core of the Vulkan backend (gpu_voxels_vulkan_view: cube extraction, renderer, offscreen images, scene, style file; no window system)" ON)
option(GVL_VULKAN_VIEWER "Build the window part of the in-process Vulkan viewer (gpu_voxels_vulkan_viewer, needs GLFW) and the gvl_viewer program; needs GVL_VULKAN_VIEW" ON)
if(GVL_BACKEND_VULKAN AND GVL_VULKAN_VIEWER AND NOT GVL_VULKAN_VIEW)
  message(FATAL_ERROR "GVL_VULKAN_VIEWER (window, gvl_viewer) needs GVL_VULKAN_VIEW (viewer core): switch GVL_VULKAN_VIEW on or GVL_VULKAN_VIEWER off")
endif()

# ---- Optional third-party features of the CUDA backend ---------------------
option(GVL_WITH_PCL "Build PCL point cloud file support (pcd) into gpu_voxels_helpers" ON)
option(GVL_WITH_KINECT "Build OpenNI2 Kinect support into gpu_voxels_helpers (needs GVL_WITH_PCL)" ON)

# ---- Tests ------------------------------------------------------------------
option(BUILD_TESTS "Build the unit tests (GoogleTest, plus the Boost.Test suites of icl_core)" ON)

# ---- vcpkg manifest features -------------------------------------------------
if(GVL_BACKEND_CUDA)
  list(APPEND VCPKG_MANIFEST_FEATURES "cuda")
  if(GVL_WITH_PCL)
    list(APPEND VCPKG_MANIFEST_FEATURES "pcl")
  endif()
endif()
if(GVL_BACKEND_VULKAN)
  list(APPEND VCPKG_MANIFEST_FEATURES "vulkan")
  if(GVL_VULKAN_VIEWER)
    list(APPEND VCPKG_MANIFEST_FEATURES "vulkan-viewer")
  endif()
endif()
if(BUILD_TESTS)
  list(APPEND VCPKG_MANIFEST_FEATURES "tests")
endif()
