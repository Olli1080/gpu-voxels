# Cleanup Candidates (first round done 2026-11-05)

Noticed on 2026-10-22 in `C:\Code\2026-10-22_gpu-voxels`. Approve via [open_questions.md](open_questions.md) Q12 before acting; deletions go through git so they are recoverable.

| Path | Observation | Suggested action |
|---|---|---|
| ~~`packages/examples/RobotVsEnvironment - Kopie.cpp`~~ | **Done 2026-11-05:** not a duplicate (full diff read): an older variant with the Hollie arm from `.xyz` point clouds, three translation joints and the old `DHParameters` API; renamed to `RobotVsEnvironmentHollie.cpp`, still not built | kept |
| ~~`packages/gpu_voxels/CMakeLists_old.txt`~~ | **Done 2026-11-05:** deleted (no reference in any CMake or source) | deleted |
| ~~`Dependencies.conf` (x2), `packages/icl_core/package.xml`, `pom.xml`, `icl_core_config.pch`, `icl_core_logging.pch`~~ | **Done 2026-11-05:** deleted (remnants of ICMAKER / ROS / Maven; the `.pch` files were tracked although `.gitignore` ignores `*.pch`; nothing referenced any of them) | deleted |
| ICMAKER relics in module `CMakeLists.txt` (commented blocks, `if(OFF)` sections, `#ICMAKER_*`) | Dead code | Remove during the CMake port of examples/viewer/tests, not before |
| `packages/gpu_voxels/gpu_voxels/helpers/include/gpu_voxels/helpers/tfHelper.h`, `Kinect.*` | ROS-/OpenNI-dependent helpers, conditionally built | Decide with scope question Q6 |
| Mojibake in `README.md` | Encoding damage | Fix with the README rewrite (the README still describes CUDA only) |
| `.gitignore` (≈ 7 KB Visual Studio template) | Mostly irrelevant entries; one entry `packages/gpu_voxels/models/franka model/**` | Leave; low value |

Not in this repo but related: the leftovers in the archives (`test_octree_build_out_v*.txt`, `fix_leaks.py`, `.old` headers) — see [mistakes_to_avoid.md](../knowledge_base/vulkan_migration/mistakes_to_avoid.md) D2/D3. Do not copy them over.
