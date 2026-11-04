# Analysis — Index

Home for **investigation tasks**: one file per investigation, named `<YYYY-MM-DD>_<topic>.md`, created from [_template.md](_template.md). An analysis answers a question and ends with findings + evidence + a recommendation; lasting facts are then promoted into `knowledge_base/` and linked from here.

| File | Content |
|---|---|
| [_template.md](_template.md) | Template for a new analysis (question, method, evidence, findings, confidence, follow-ups) |
| [2026-11-04_k41_kernel_stub_placeholder.md](2026-11-04_k41_kernel_stub_placeholder.md) | **K41 root cause (CUDA kernels that could not be launched):** declaration and definition of a `__global__` template differ in the top-level `const` of a by-value parameter, nvcc then calls a placeholder `cudaLaunchKernel(NULL)`; evidence from the object code and the generated host code, the 17 affected kernel host functions, the fix and the check tool |
| [2026-11-03_cuda_parity.md](2026-11-03_cuda_parity.md) | **CUDA parity (scenarios 1 - 5: list x map collisions, list merges, map x map collisions, prob. map inserts, sensor data; K44: CUDA takes negative coordinates as voxel 0):** how CUDA is built and run on this machine (Ninja + MSVC 14.44, one architecture), the method (shared inputs, CUDA dump, golden file, Vulkan test against four lines), result: all 441 lines match the models, Vulkan equals CUDA wherever the offset rules agree, CUDA's linear offset confirmed; K39 (CUDA's types form cannot launch on the RTX 4090: 80 registers x 1024 threads) |
| [2026-11-02_s7_facade_checklist.md](2026-11-02_s7_facade_checklist.md) | **S7 checklist (`GpuVoxels` façade):** every public member and every `GpuVoxelsMap` virtual by map type against the Vulkan state (R / W / N / O), the `MapType` mapping, six design questions with defaults, what the examples call, slice plan S7-1 .. S7-5 |
| [2026-11-01_octree_dynamic_timings.md](2026-11-01_octree_dynamic_timings.md) | **Timings of the dynamic octree (RTX 4090, Release), before and after the fast insert (S6d-5):** insert of 1 voxel 2.2 -> 0.2 ms, 1 M voxels 63 -> 9 ms, 200 small inserts 4.6 -> 0.6 ms median; rebuild 3 - 22 ms for 14 - 344 MiB, chunks do not slow queries, garbage count 30 -> 0.25 ms per call after S6d-6 (GPU), propagate of 1 M pending ids 13 -> 5 ms after S6d-7 (exactly sized lists) |
| [2026-10-31_s6d_design.md](2026-10-31_s6d_design.md) | **S6d design (dynamic octree updates):** what CUDA `insertVoxel` / `propagate` / `rebuild` do (read), the chunk-list prerequisite, sub-slice plan S6d-0 .. S6d-4 |
| [2026-11-02_s6f_sensor_free_space_design.md](2026-11-02_s6f_sensor_free_space_design.md) | **S6f design (point cloud insert with free space):** what CUDA does (two voxelizations, ray cast on the free-space grid, packing into cubes, soft insert, object insert), CUDA quirks not copied, dense count map design, slices S6f-1 .. S6f-4 |
| [2026-10-31_machine_baseline.md](2026-10-31_machine_baseline.md) | **Baseline on the new machine (i9-14900K, RTX 4090):** build and test results, device facts, timings next to the Intel UHD numbers, K35 (load balancer livelock on a wide GPU) |
| [2026-10-30_octree_checkpoint/](2026-10-30_octree_checkpoint/summary.md) | **Octree checkpoint (after S6c):** 64-bit sort, bytes per voxel (faithful vs index pool vs sorted levels), frontier sizes, balancer vs wavefront, pointer-fault experience, revisit triggers; recommendation for S6d (decision Q24) |
| [2026-10-29_octree_design_review/](2026-10-29_octree_design_review/summary.md) | **Octree design review (S6):** CUDA design facts, porting hazards, three candidate designs, recommendation (index pool + wavefront), slice plan and the decisions Q19-Q22 |
| [2026-10-22_workspace_survey.md](2026-10-22_workspace_survey.md) | Initial survey of the three workspaces and the failed migration attempts: method, what was and was not verified, pointers to resulting knowledge |

## Suggested future analyses (not started)
- Layout equivalence: host structs (`BitVoxel<256>`, `DistanceVoxel`, `ProbabilisticVoxel`, `MetaPointCloudStruct`) vs. GLSL — incl. the 9/10-bit distance-voxel packing question (K14).
- Raw Vulkan runtime spike vs. Kompute on one kernel (`kernelInsertGlobalPointCloud` + collide): code size, dispatch overhead, validation cleanliness.
- Device capability survey of target GPUs (limits, subgroup sizes, atomics, 64-bit) once Q4 is answered.
- Public-API surface inventory: every header symbol exposing `thrust::`/`cuda` types (for the API-change decision, Q7).
