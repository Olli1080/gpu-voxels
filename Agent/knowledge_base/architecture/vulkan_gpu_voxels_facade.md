# `GpuVoxels` façade, maps behind one type (slice S7-1, 2026-11-02)

Status: **Tested, not Parity** (the CUDA behaviour is a reading of `GpuVoxels.cu`, `GpuVoxelsMap.h` and the map classes; nothing ran under CUDA). Plan and checklist: [S7 checklist](../../Analysis/2026-11-02_s7_facade_checklist.md). Header `vulkan/include/gpu_voxels/vulkan/GpuVoxels.h`, code `vulkan/src/GpuVoxels.cpp`, namespace `gpu_voxels::vulkan`. The legacy header is untouched.

Not part of this slice: URDF, viewer (later). Primitive arrays, list collisions with maps and the example smoke tests (S7-5): [vulkan_gpu_voxels_examples.md](vulkan_gpu_voxels_examples.md). Merge and disk I/O (S7-4): [vulkan_gpu_voxels_merge_io.md](vulkan_gpu_voxels_merge_io.md). Point cloud files and boxes (S7-2): [vulkan_gpu_voxels_files.md](vulkan_gpu_voxels_files.md); robots (S7-3): [vulkan_gpu_voxels_robots.md](vulkan_gpu_voxels_robots.md).

## Types
| Type | Contract |
|---|---|
| `MapType` (in `vulkan_types.h`) | the CUDA `MT_*` values in the same order. Built: bit map, prob. map, distance map, bit list, counting list, deterministic and probabilistic octree. `MT_PROBAB_VOXELLIST` and the two Morton list types are **not built** (as in CUDA, where `addMap` throws "not implemented"); here `VulkanError` with the type name and "not implemented" |
| `map_type_name(type)` | `"MT_BITVECTOR_VOXELMAP"`, ..., `"unknown map type"` for a value outside the enum |
| `GpuVoxelsMap(context, type, dimensions, side)` | owns **one** of the concrete classes (`std::variant` of `unique_ptr`s); a failing constructor of the class (e.g. a distance map above 1023 voxels per axis) propagates |
| `GpuVoxelsMapSharedPtr` | what `add_map` / `get_map` hand out. A handle must not outlive the `GpuVoxels` (the `Context`) that made it |
| `GpuVoxels(Context&)` | uses an existing context (not owned); `get_instance()` is the singleton (a `weak_ptr` like CUDA: the same object while someone holds it) and **owns** its context, created in `initialize()` with the default `ContextCreateInfo` |

## `GpuVoxelsMap` members
| Member | Behaviour |
|---|---|
| `type()` | the tag given at construction |
| `as<T>()`, `is<T>()` | `T*` / `bool`, **checked**: `nullptr` / `false` when the map is another class (CUDA: an unchecked `static_cast`). Both octree kinds are `Octree` (ask `type()` or `Octree::kind()`). Collisions, queries and downloads are reached through `as<T>()` as in CUDA |
| `insert_point_cloud`, `insert_coordinate_list`, `insert_meta_point_cloud` (one meaning) | forwarded to the class; return the number of points that were not inserted. Meaning rules stay those of the class (counting list: ignored, prob. map: update, distance map: occupied only, octree: occupied or free; others throw) |
| `insert_meta_point_cloud(cloud, meanings)` | one meaning per sub cloud, forwarded. The distance map has no such overload: a wrong number of meanings throws, any meaning but occupied throws **before** anything is inserted, otherwise it is the one-meaning insert (the only meaning it knows) |
| `clear_map()` | `clear_map()` of maps, `clear()` of lists and octrees |
| `clear_bit_voxel_meaning(m)` | bit map: the class; prob. map and octrees: occupied clears the map, every other meaning throws (CUDA: logs an error); distance map and lists: **throw** "not supported" (CUDA: no-op / logs) |
| `needs_rebuild()`, `rebuild()`, `rebuild_if_needed()` | octrees: the class (`rebuild()` returns true); every other type is never fragmented: `false` (CUDA: logs "not supported" and returns false). `rebuild_if_needed` as in CUDA (true when it rebuilt) |
| `memory_usage_bytes()` | `memory_usage_bytes()` of the class, octrees: `memory_usage().bytes` |
| `dimensions()`, `metric_dimensions()`, `voxel_side_length()` | maps: their size; **lists: the reference map** (CUDA returns `(entries, 1, 1)`, use `as<BitVoxelList>()->size()`); octrees: 2^14 per axis (the CUDA metric variant has a typo, `dim.z()` for y). Metric = voxels * side |

## `GpuVoxels` members
| Member | Behaviour |
|---|---|
| `initialize(dimensions, side)` | empty dimension, a side length that is not positive and finite, or existing maps throw (the size of the maps would no longer match); nothing changes then. An owning instance creates its context here |
| `add_map(type, name)` | before `initialize`: throws. Name used: error log + empty pointer (the first map stays). Types not built: throws, nothing is registered |
| `del_map`, `get_map`, `clear_map(name)`, `clear_map(name, meaning)` | unknown name: error log + `false` / empty. A handle from before `del_map` stays valid |
| `insert_point_cloud_into_map`, `insert_meta_point_cloud_into_map` (2 overloads) | `false` for an unknown map, else `true`; an optional `uint32_t* outside` receives the number of points not inserted (CUDA loses it) |
| `visualize_map` | `false` without a viewer (CUDA without `GPUVOXEL_VISUALIZE` the same); with a `ViewerScene` (2026-11-04, [vulkan_viewer.md](vulkan_viewer.md)) it calls the registered hook (`set_visualization_hooks`): the map becomes a layer, `true` for a displayable map; `visualize_primitives_array` likewise |
| `dimensions()`, `voxel_side_length()` | what `initialize` stored (CUDA: `getDimensions`, `getVoxelSideLength`) |

Decisions D1 .. D6 of the checklist apply as written: thin handle (D1), unsupported operations throw while unknown names keep CUDA's log + `false` (D2, **not confirmed by the user**), no locking (D3), Vulkan value types in the signatures (D4), code next to the classes (D5), headless smoke tests instead of the viewer examples (D6, S7-5). Logging uses `GVL_LOG_ERROR` from `vulkan/Log.h` (until 2026-11-05 the icl_core stream `VulkanLog` with `LOGGING_ERROR_C`; same texts).

## Differences from CUDA (read from `GpuVoxels.cu`, `GpuVoxelsMap.h`, the map classes)
| CUDA | Here |
|---|---|
| `as<T>()` is an unchecked `static_cast` | checked |
| unsupported operations log and return a value (`true` for self-collision insertion, `false` for `merge`) | throw `VulkanError` with the operation and the type |
| `getDimensions` of a list returns the number of entries | the reference map's dimensions |
| `clearBitVoxelMeaning` of prob. map / octree: other meanings log an error; distance map / lists: no-op or log | throws / throws |
| `rebuild`, `needsRebuild` of maps and lists log an error | `false`, silently |
| `initialize` can be repeated (not checked how it treats existing maps) | refused while maps exist |
| `addMap` of a missing type: `std::exception` | `VulkanError` |

## Verification
- `tests/vulkan/test_gpu_voxels.cpp`, 14 cases (13 `FacadeTest`, 1 `FacadeNames`): every built type is exactly its class (and the const `as<T>()`); types not built and a failing constructor register nothing; names; **the inserts of all seven types through the façade equal the class used directly** (twin built in the test: points, coordinates, meta cloud with one meaning and with one per sub cloud, outside counts, whole content, with bad points) plus a host-computed expectation of the whole bit map; meanings per sub cloud (wrong numbers, distance map); `clear_map` and `clear_bit_voxel_meaning` for every type; rebuild (non-octrees `false`, octrees with a policy that asks for it and with dead data that only a rebuild frees); sizes and dimensions; the named inserts with the counter; `initialize` rules; the singleton with its own context; the names of the types as literals.
- Mutation runner `tools/mutation/mutate_s7a_facade.py` (59 mutations of the dispatch, names, initialisation and singleton): **all 59 killed, none equivalent** (log `tools/mutation/logs/s7a_facade.log`). Two test gaps were closed while writing it: a rebuild that does nothing (the extraction does not change, so the test now needs dead data and compares the memory), and `add_map` before `initialize` (a map without size throws too, so the message is checked).
- **Not verified:** CUDA parity; the owning instance was only run with the default `ContextCreateInfo`; a handle that outlives its `GpuVoxels` is documented as forbidden, not detected.
