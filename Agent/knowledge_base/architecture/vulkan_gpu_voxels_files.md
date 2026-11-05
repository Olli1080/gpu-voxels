# Point cloud files and boxes in the façade (slice S7-2, 2026-11-02)

Status: **Tested, not Parity** (the CUDA behaviour is a reading of `PointcloudFileHandler.cpp`, `XyzFileReader.cpp`, `PcdFileReader.cpp`, `BinvoxFileReader.cpp`, `GeometryGeneration.cu` and `GpuVoxels.cu`; nothing ran under CUDA). Builds on [vulkan_gpu_voxels_facade.md](vulkan_gpu_voxels_facade.md). Header `vulkan/include/gpu_voxels/vulkan/PointCloudFile.h`, code `vulkan/src/PointCloudFile.cpp`; the two façade calls are in `GpuVoxels.h` / `GpuVoxels.cpp`. Plan: [S7 checklist](../../Analysis/2026-11-02_s7_facade_checklist.md), rows 15 and 20.

The `.xyz` and `.binvox` readers are plain host code (no CUDA, thrust, Eigen or PCL). The `.pcd` reader is **PCL's** (`pcl::io::loadPCDFile<pcl::PointXYZ>`, as in CUDA) and exists only with the option `GVL_WITH_PCL` (macro `_BUILD_GVL_WITH_PCL_SUPPORT_`); without it `.pcd` files are refused with a message that names `GVL_WITH_PCL`. Until 2026-11-05 the Vulkan side had a parser of its own (ascii and binary); it was deleted in favour of PCL's loader.

## Readers
| Call | Contract |
|---|---|
| `read_xyz_file(file)` | the first three numbers of every line are a point. Empty lines, `#` lines and lines that do not start with three numbers (a header `x y z`) are skipped (a warning counts them), more columns are ignored. `+1`, `1e-1`, tabs and CRLF are fine |
| `read_pcd_file(file)` | PCL's loader: every format PCL reads (also `binary_compressed`), all points including NaN ones. A missing file is a `PointCloudFileError` ("could not open"), so is a failure of PCL (e.g. an empty file); a text without a PCD header gives an empty cloud (PCL's behaviour, the CUDA reader returns it as well). Always an error in a build without `GVL_WITH_PCL` |
| `read_binvox_file(file)` | header with `dim` (depth height width), `translate`, `scale`, `data`, in any order (CUDA needs `dim` before `scale`); unknown keywords skip their line; run length coded `(value, count)` pairs; a voxel with value 1 is a point `scale / width * index + translate`, x outermost, then z, then y. Decoded run by run without a voxel array |
| `read_point_cloud_file(file)` | by extension, `.xyz` / `.pcd` / `.binvox`, **case insensitive** (CUDA: lower case only) |
| `shift_point_cloud_to_zero(points)` | minimum of every axis to 0; points that are not finite do not count and stay as they are; an empty cloud is fine (CUDA: undefined) |
| `load_point_cloud(file, model_path, options)` | `read(model_path / file)` (an absolute `file` replaces `model_path`), then shift, then `p = scaling * p + offset` |

Every problem with a file (cannot open, unknown extension, malformed) throws `PointCloudFileError` (a `VulkanError`) with the file name and the reason.

## Box
`box_voxel_coordinates(min, max, spacing, side, limit)` = CUDA `createBoxOfPoints(min, max, delta, side)`: the voxels from `floor(max(min, 0) / side)` to `ceil(max / side)` **inclusive** on every axis (the voxel at the upper corner is in even when `max` is a multiple of the side), step `round(spacing / side)` voxels, x outermost and z innermost, in float arithmetic like CUDA. Voxels at or beyond `limit` are not generated and only counted in `total`; more than `kMaxBoxVoxels` (2^26) inside the limit throw before anything is generated.

## Façade calls
| Call | Contract |
|---|---|
| `insert_point_cloud_from_file(map, file, model_path, meaning, shift_to_zero, offset, scaling, outside*)` | `load_point_cloud`, then the map's `insert_point_cloud`. Returns `false` and logs for an unknown map **or any file problem** (as CUDA); the map is not touched then. The meaning rules of the map still throw. `outside` = points not inserted |
| `insert_box_into_map(min, max, map, meaning, points_per_voxel = 1, outside*)` | the box with `spacing = side / points_per_voxel` and the map's dimensions as limit (octree: 2^14), inserted as a coordinate list. `outside` = voxels beyond the map (counted, not generated) plus what the map drops. `false` for an unknown map |

## Differences from CUDA
| CUDA | Here |
|---|---|
| `.xyz`: every triple of a line is a point (six numbers give two points); other lines silently | first three numbers only; lines without three numbers are skipped with a warning |
| `.binvox`: truncated data silently accepted; `scale` / `width` order dependent (static variables); `value == 1` is a point | truncated data and runs beyond the volume are errors; order independent |
| `shiftPointCloudToZero` of an empty cloud reads `points[0]` | empty cloud: nothing; non-finite points ignored |
| extension compare is case sensitive | case insensitive |
| `createBoxOfPoints`: `round(delta / side) == 0` (`points_per_voxel` of 3 or more) never ends; `min > max` throws `std::exception`; a negative maximum casts a negative float to `uint32_t` | step at least 1 (so `points_per_voxel` makes no difference above 2, as in CUDA below that); `VulkanError` for an empty box (voxel minimum above maximum, also for a negative maximum), `points_per_voxel` 0, corners that are not finite or beyond +-2^31 |
| `insertBoxIntoMap` builds the whole box before the map drops what is outside | the box is clipped to the map first (the count is the same) |
| a file that cannot be read: log + `false` | same, only for file errors; the Vulkan side has no other reason to return `false` here |

## Verification
- `tests/vulkan/test_point_cloud_file.cpp`, 11 cases (8 `PointCloudFiles`, 3 `FileFacadeTest`): **the readers against the repository's model files** with count, sums, extremes, first and last point computed by an independent Python decoder (`hollie/arm_0_link.xyz`, `hollie_from_pointcloud2.pcd` binary with rgba, `robot4cmRes.pcd` ascii, `pointcloud_0002.pcd` with intensity, `coordinate_system_100.binvox`); files written by the tests (xyz syntax variants, a pcd file read by PCL, a missing and an empty pcd file, a hand-made binvox volume with known runs); extension, shift and the load order; the box against an independent loop (inclusive corners, cut at 0, rounding, steps, limits, errors by message); the façade calls against the classes used directly (bit map, octree, bit list, three option sets, failures leave the map untouched, outside counts of a box beyond the map).
- Mutation runner `tools/mutation/mutate_s7b_files.py` (75 mutations): **71 killed, 4 equivalent** (X1 / P16: comments that are not skipped as comments are skipped as lines without numbers, or become an unused header key; G12 / G13: the class drops and counts what the box would put outside, so clipping to a bigger limit gives the same result). Gaps closed while writing it: a missing `FIELDS` line was caught by the length check with a message that also contains the word (the message is checked now), an unknown binvox keyword line hid nothing because the real `dim` line came after it (it comes before now), the start at a limit with a step above 1, a limit that the step hits exactly, a box with a voxel minimum one above the maximum, and the messages for every refused box. Log `tools/mutation/logs/s7b_files.log`.
- **Not verified:** CUDA parity; the `.pcd` reader is PCL's and is only tested on the three model files and one small ascii file (no `binary_compressed` file is in the repository); in a build without `GVL_WITH_PCL` the model tests are skipped and the refusal is tested; a box with exactly `kMaxBoxVoxels` voxels (the limit test uses one more, the exact boundary would need 800 MB).
