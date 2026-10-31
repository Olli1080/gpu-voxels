# Analysis — Index

Home for **investigation tasks**: one file per investigation, named `<YYYY-MM-DD>_<topic>.md`, created from [_template.md](_template.md). An analysis answers a question and ends with findings + evidence + a recommendation; lasting facts are then promoted into `knowledge_base/` and linked from here.

| File | Content |
|---|---|
| [_template.md](_template.md) | Template for a new analysis (question, method, evidence, findings, confidence, follow-ups) |
| [2026-10-31_machine_baseline.md](2026-10-31_machine_baseline.md) | **Baseline on the new machine (i9-14900K, RTX 4090):** build and test results, device facts, timings next to the Intel UHD numbers, K35 (load balancer livelock on a wide GPU) |
| [2026-10-30_octree_checkpoint/](2026-10-30_octree_checkpoint/summary.md) | **Octree checkpoint (after S6c):** 64-bit sort, bytes per voxel (faithful vs index pool vs sorted levels), frontier sizes, balancer vs wavefront, pointer-fault experience, revisit triggers; recommendation for S6d (decision Q24) |
| [2026-10-29_octree_design_review/](2026-10-29_octree_design_review/summary.md) | **Octree design review (S6):** CUDA design facts, porting hazards, three candidate designs, recommendation (index pool + wavefront), slice plan and the decisions Q19-Q22 |
| [2026-10-22_workspace_survey.md](2026-10-22_workspace_survey.md) | Initial survey of the three workspaces and the failed migration attempts: method, what was and was not verified, pointers to resulting knowledge |

## Suggested future analyses (not started)
- Layout equivalence: host structs (`BitVoxel<256>`, `DistanceVoxel`, `ProbabilisticVoxel`, `MetaPointCloudStruct`) vs. GLSL — incl. the 9/10-bit distance-voxel packing question (K14).
- Raw Vulkan runtime spike vs. Kompute on one kernel (`kernelInsertGlobalPointCloud` + collide): code size, dispatch overhead, validation cleanliness.
- Device capability survey of target GPUs (limits, subgroup sizes, atomics, 64-bit) once Q4 is answered.
- Public-API surface inventory: every header symbol exposing `thrust::`/`cuda` types (for the API-change decision, Q7).
