**GPU-Voxels** is a library which allows high resolution volumetric collision detection
between animated 3D models and live pointclouds from 3D sensors of all kinds.
It is mainly developed for robotics planning and monitoring tasks.
See our videos on http://www.gpu-voxels.org/demos/ to get an impression.

It was written for CUDA. This fork keeps the CUDA backend and is adding a **Vulkan backend** (compute shaders instead of CUDA kernels) with a small in-process viewer. Both backends are selected with CMake options (see below). The Vulkan backend is
work in progress: what it covers, what was compared against CUDA and what is missing is written down in
[Agent/knowledge_base/architecture/summary.md](Agent/knowledge_base/architecture/summary.md) and
[Agent/knowledge_base/known_issues.md](Agent/knowledge_base/known_issues.md).

## Support:
Find more information on our website http://www.gpu-voxels.org
Doxygen is available at http://www.gpu-voxels.org/doxygen/html/ (this is the documentation of the CUDA library as published by the original authors)

If you have questions or experience any problems with the software, please post to our GitHub Bugtracker:
at https://github.com/Olli1080/gpu-voxels/issues

## Dependencies:
The dependencies are declared in `vcpkg.json` and installed by vcpkg (the repository contains vcpkg as the git submodule `vcpkg/`; it has to be
bootstrapped once). What a build needs depends on the options:

**CUDA backend** (`GVL_BACKEND_CUDA`, default ON):

- CUDA toolkit with `nvcc` (installed separately, not through vcpkg; developed with CUDA 12.8)
- Eigen 3 (vcpkg's `eigen3`, found with `find_package(Eigen3)`), glm, stb, zlib, TinyXML 2
- PCL, optional (see `GVL_WITH_PCL`)
- the vendored `packages/icl_core` (logging, config), built with this backend only

**Vulkan backend** (`GVL_BACKEND_VULKAN`, default OFF):

- Vulkan SDK (loader, `glslangValidator`; the validation layer is needed by the tests unless `GVL_ALLOW_NO_VALIDATION=1` is set); the environment
  variable `VULKAN_SDK` should be set (CMake uses it to find the SDK)
- a Vulkan 1.3 device with `synchronization2` and `shaderInt64`
- Vulkan Memory Allocator (vcpkg), TinyXML 2 (style file of the viewer)
- GLFW (vcpkg) for the window of the viewer, only with `GVL_VULKAN_VIEWER`
- PCL, optional (see `GVL_WITH_PCL`)

**Both:** PCL is used to read `.pcd` files (`GVL_WITH_PCL`, default ON for both backends). Without it `.pcd` files are refused, `.xyz` and
`.binvox` files are read by the library itself. **Tests** (`BUILD_TESTS`, default ON): GoogleTest; with the CUDA backend also Boost.Test and CppUnit
for the tests of `icl_core`.

**Not part of the build at the moment** (the sources are still in the repository): the old OpenGL visualizer (`packages/gpu_visualization`,
`vis_interface`; GLEW, GLM, GLUT), the example programs (`packages/examples`), `gvl_ompl_planning`, `example_how_to_link`, and the URDF robot
(needs urdfdom and orocos-kdl, switched off in `gpu_voxels/robot/CMakeLists.txt`). Kinect support was removed.

## Compiling:
The options are in `cmake/Options.cmake`: `GVL_BACKEND_CUDA`, `GVL_BACKEND_VULKAN`, `GVL_CUDA_ARCHITECTURES` (default `all`, for example `89` builds only
for an RTX 4090), `GVL_VULKAN_VIEW` and `GVL_VULKAN_VIEWER` (default ON with Vulkan), `GVL_WITH_PCL`, `BUILD_TESTS`. The first configure builds the
vcpkg packages, which takes a while.

Vulkan backend only (Windows, Visual Studio generator):

        cmake -S . -B build/vulkan -G "Visual Studio 18 2026" -A x64 -DCMAKE_TOOLCHAIN_FILE=%CD%/vcpkg/scripts/buildsystems/vcpkg.cmake -DGVL_BACKEND_CUDA=OFF -DGVL_BACKEND_VULKAN=ON
        cmake --build build/vulkan --config Debug --parallel
        ctest --test-dir build/vulkan -C Debug

CUDA backend (Windows, Ninja; CUDA 12.8 does not accept the newest MSVC toolset, so the build was done in a shell prepared with
`vcvarsall.bat x64 -vcvars_ver=14.44`):

        cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=<repository>/vcpkg/scripts/buildsystems/vcpkg.cmake -DGVL_BACKEND_CUDA=ON -DGVL_CUDA_ARCHITECTURES=89
        cmake --build build/cuda
        ctest --test-dir build/cuda

`CMakePresets.json` defines presets for both backends; they have not been used for the builds described here.

No example programs and no visualizer are built, with either backend (except the Vulkan viewer below).

### Viewer (Vulkan backend)
`gvl_viewer` (target `gvl_viewer`, with `GVL_VULKAN_VIEWER`) opens a window with the voxel maps, octrees, primitives and a robot of a small
demo scene:

        gvl_viewer --facade

`--png out.png --size 1280x720` renders the same picture into a file without a window. Other options and the keys are listed in
[Agent/knowledge_base/architecture/vulkan_viewer.md](Agent/knowledge_base/architecture/vulkan_viewer.md).

## Known issues
- Developed and tested on Windows 11 with an NVIDIA RTX 4090 (CUDA 12.8, Vulkan 1.4 driver, MSVC). Linux and other GPUs and drivers were not tried for
  the current state of the code.
- The Vulkan backend was compared against a CUDA run only for a few scenarios; everything else is tested against independent CPU references, which is
  not the same as parity with CUDA. The list is in [Agent/knowledge_base/known_issues.md](Agent/knowledge_base/known_issues.md).
- Morton-code voxel lists are not built in the Vulkan backend (and not implemented in CUDA either).

## Maintainers:
The library is developed and maintained as a fork by:
- Oliver Zahn

Former Contributors are (in temporal order):

- Christian Jülg
- Andreas Hermann
- Sebastian Klemm
- Florian Drews
- Matthias Wagner
- Felix Mauch
- Herbert Pietrzyk
- Gabriele Bolano
- Florian Fervers
- Erik Albert

## License:
The software in this repository underlies different licenses.
Please consider the LICENSE.txt in the different package directories.
Feel free to contact us for any licensing questions.

To clarify this:

- The Build System (icmaker and according scripts) is distributed under BSD license
- GPU-Voxels is distributed under the CDDL license
- icl_core helber library is distributed under CDDL license

Please let us know if you are using the **GPU-Voxels** library,
as we are curious to find out how it enables other people's work or research.

We would be grateful if scientific publications resulting from projects
that make use of **GPU-Voxels** would cite our overview paper:
*A. Hermann, F. Drews, J. Bauer, S. Klemm, A. Roennau and R. Dillmann:
**“Unified GPU Voxel Collision Detection for Mobile Manipulation Planning”**
in Intelligent Robots and Systems (IROS 2014, Chicago, September 2014)*

Copyright (c) 2014, *FZI Forschungszentrum Informatik*, Karlsruhe, Germany
All rights reserved.

## Thanks:
The **GPU-Voxels** library was developed at the *FZI Forschungszentrum Informatik*, Karlsruhe, Germany.

The research leading to the results has received funding from... 
- the **German Federal Ministry of Education and Research (BMBF)** under grant agreement no. 01IM12006C (ISABEL).
- the **European Union** program **Horizon 2020** under grant agreement no. 680734 (HORSE).
- the **Baden-Württemberg Stiftung** through the KolRob reasearch project.

