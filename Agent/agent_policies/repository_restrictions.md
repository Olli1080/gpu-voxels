# Repository Restrictions

## Do not modify
| Path | Why |
|---|---|
| `vcpkg/` (git submodule, pinned `d4c0a46cb8` = commit of 2026-10-28 since the user-approved bump of 2026-10-30; the old upstream pin was `984f9232b2` of 2025-06-06) | Third-party; changing the pin again is a user decision (see [build_system.md](../knowledge_base/architecture/build_system.md)) |
| `LICENSE*.txt` (CDDL, BSD, PBA, stb) | Licensing; per-file license headers must be preserved when moving/porting code |
| `packages/icl_core/**` | Vendored FZI logging/config library; change only for a concrete build/port reason, minimal diff |
| `packages/gpu_voxels/models/**` | Binary assets (135 `.binvox`, `.stl`, `.urdf`, `.pcd`); `franka model/**` is git-ignored on purpose |

## Preserve
- **Upstream attribution:** original authors are listed in `README.md`; keep `\author` headers in ported files. The PBA code has its own license (`LICENSE_PBA.txt`).
- **Folder convention:** `include/gpu_voxels/<module>/…` public headers, `src/…` sources, per-module `CMakeLists.txt` creating target `gpu_voxels_<module>` with `FILE_SET HEADERS`. New modules follow it.
- **File-suffix convention:** `.h/.cpp` plain C++, `.hpp` templated headers, `.cu/.cuh` CUDA sources, `.cuhpp` templated CUDA code. A Vulkan backend introduces its own suffixes/dirs (e.g. `.comp` / `.slang`); never rename `.cu` to `.cpp` in place (guardrail 1).
- **C++ standard:** C++20 (`cxx_std_20`); CMake ≥ 3.23.
- **Logging:** CUDA modules use the `LOGGING_*` macros from `icl_core_logging`; the Vulkan backend uses `GVL_LOG_ERROR/WARNING/INFO` from `gpu_voxels/vulkan/Log.h` (no icl_core there). No `std::cout`/`printf` in library code.
- **Error handling:** CUDA code uses `HANDLE_CUDA_ERROR` / `CHECK_CUDA_ERROR` (`cuda_handling.h`); the Vulkan equivalent must be introduced as a documented abstraction, not by redefining the CUDA macros to no-ops.

## Line endings / encoding
- Text files in the working tree are **CRLF** (measured with Python byte counts: 575 CRLF vs 152 LF files, the LF ones mostly binary; `core.autocrlf=true` in the system gitconfig, no root `.gitattributes`). Note: MSYS `grep` strips CR, so `grep -c $''` wrongly reports 0 — count bytes in Python. When editing an existing file preserve its line endings (open with `newline=''`); write new source/CMake files with CRLF as well, otherwise git warns "LF will be replaced by CRLF". Don't mass-normalise (unreadable diffs) and don't change `core.autocrlf`.
- `README.md` contains mojibake (`JÃ¼lg`, `â€œ…â€`) — pre-existing UTF-8/CP1252 damage; fix only on request.
- New files: UTF-8 without BOM.

## Build outputs
- `.gitignore` ignores `build/**`, `x64/`, `vcpkg_installed/`, `*.log`, `*.tmp`, `.vscode`, `Intermediate/`.
- **Caution:** it also contains `**/[Pp]ackages/*` followed by `!packages/**` — the re-include is what keeps `packages/` tracked. Don't remove that pair.
- Never write test dumps/logs/generated files to the repo root or into tracked directories.

## Dependencies
- Dependencies are declared in `vcpkg.json` (manifest mode) with overlay ports in `overlay-ports/` (`eigen3` 3.4.0 with nvcc patches, `cudnn`). Adding a dependency = edit `vcpkg.json`, state why in the plan, and record the version in [build_system.md](../knowledge_base/architecture/build_system.md).
- Do not add ROS/catkin dependencies to the core build.
