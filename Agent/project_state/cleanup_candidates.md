# Cleanup Candidates (proposals only — nothing deleted)

Noticed on 2026-10-22 in `C:\Code\2026-10-22_gpu-voxels`. Approve via [open_questions.md](open_questions.md) Q12 before acting; deletions go through git so they are recoverable.

| Path | Observation | Suggested action |
|---|---|---|
| `packages/examples/RobotVsEnvironment - Kopie.cpp` | German "copy" of `RobotVsEnvironment.cpp` (space in name) | Diff against the original, then delete |
| `packages/gpu_voxels/CMakeLists_old.txt` | Legacy build file, not referenced by CMake | Delete or move to docs |
| `packages/gpu_voxels/Dependencies.conf`, `packages/icl_core/Dependencies.conf`, `package.xml`, `pom.xml` | Remnants of earlier build/packaging systems (ICMAKER / ROS / Maven) | Keep until ICMAKER removal is decided; then drop |
| ICMAKER relics in module `CMakeLists.txt` (commented blocks, `if(OFF)` sections, `#ICMAKER_*`) | Dead code | Remove during the CMake port of examples/viewer/tests, not before |
| `packages/gpu_voxels/gpu_voxels/helpers/include/gpu_voxels/helpers/tfHelper.h`, `Kinect.*` | ROS-/OpenNI-dependent helpers, conditionally built | Decide with scope question Q6 |
| Mojibake in `README.md` | Encoding damage | Fix with the README rewrite (the README still describes CUDA only) |
| `.gitignore` (≈ 7 KB Visual Studio template) | Mostly irrelevant entries; one entry `packages/gpu_voxels/models/franka model/**` | Leave; low value |

Not in this repo but related: the leftovers in the archives (`test_octree_build_out_v*.txt`, `fix_leaks.py`, `.old` headers) — see [mistakes_to_avoid.md](../knowledge_base/vulkan_migration/mistakes_to_avoid.md) D2/D3. Do not copy them over.
