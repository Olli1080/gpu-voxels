# Visualization and Inter-Process Communication

## Design (upstream, inferred from code)
Two processes:
- **Provider** = user program linking `gpu_voxels`. For each map visualised it creates a `Vis*` object (`VisVoxelMap`, `VisTemplateVoxelList`, `VisNTree`, `VisPrimitiveArray`; base `VisProvider`) in `vis_interface/`.
- **Viewer** = executable `gpu_voxels_visualizer` (`packages/gpu_visualization`): OpenGL via GLEW + (free)GLUT + GLM, own shaders (`shaders/*.vert/.frag`), XML configuration (`Config.xml`, `examples/VisConfig_*.xml`, colour gradients via Python helpers).

They talk through **Boost.Interprocess managed shared memory** segments named in `VisualizerInterface.h`: `VoxelmapSharedMemory`, `VoxellistSharedMemory`, `OctreeSharedMemory`, `PrimitiveArraysSharedMemory`, `VisualizerSharedMemory`. Each map publishes named variables (`voxel_map_name_N`, `…_dev_pointer_N`, `…_dimensions_N`, `…_data_changed_N`, type, side length, …; analogous for lists/octrees/primitive arrays; plus `target_point`, `toggle_draw_types`).
The payload is **not copied through shared memory**: the provider stores a `cudaIpcMemHandle_t` for the device buffer; the viewer opens it (`cudaIpcOpenMemHandle`), runs CUDA kernels (`VoxelMapVisualizerOperations.cu`, `VisualizerUtil.cu`) that write cube positions straight into OpenGL VBOs registered with `cudaGraphicsGLRegisterBuffer` (CUDA–GL interop).

## State on master
- Not built: `gpu_visualization` and `vis_interface` are not in the CMake tree (see [library_modules.md](library_modules.md)); `GpuVoxels.cu` includes the `Vis*` headers only under `#ifdef GPUVOXEL_VISUALIZE` (undefined).
- The viewer sources still use `cudaIpc*`/`cudaGraphics*` (≈ 110 `cuda*` identifier occurrences across viewer + vis_interface, measured) and ICMAKER CMake.

## Consequences for a Vulkan port
- CUDA IPC handles cannot be reused. Options: (a) copy cube lists into shared memory (what the earlier attempt did: host-side copies, slower but simple); (b) Vulkan external memory (`VK_KHR_external_memory_win32` / `_fd`) shared with the viewer; (c) in-process viewer (no IPC; earlier plan `IN_PROCESS_VISUALIZATION.md`).
- The viewer needs a rendering API decision: keep OpenGL (GLUT is long dormant) or move to Vulkan graphics (swapchain, pipelines). That is a *separate* project from the compute port — see [design_options.md](../vulkan_migration/design_options.md).
- Windows pitfalls observed by the earlier attempt (claimed, unverified): viewer hung at `SharedMemoryManagerVoxelMaps.cpp:37`; hard-coded 512 MB segment size in two places (`SharedMemoryManager.h`, `VisProvider.cpp`); orphaned named mutexes after crashes. The provider side keeps its own usage counter (`shm_states` in `VisProvider.cpp`) to remove a segment when the last user leaves.
- Reading back map contents to host for display requires explicit device→host readback in Vulkan (no pointer sharing); design the "extract occupied voxels" compute pass + compact output buffer first.
