# Mutation testing — method, runners, results

Every slice of the Vulkan backend is "Tested" in the sense of [definition_of_done.md](../agent_policies/definition_of_done.md): independent CPU reference **and** mutation testing. No CUDA is available here, so the tests are the only evidence; mutation testing shows that they can fail.

## What a mutation is
A *mutant* is the source with exactly one small, plausible bug (a `>` that becomes `>=`, a swapped index, a dropped term, a constant moved by one, a missing barrier-relevant write). The test suite is run against it:
- **killed** = at least one test fails (also: the mutant does not compile, or the test run hangs and is stopped by the timeout: a hanging kernel is a failure the tests notice);
- **survived** = all tests pass, i.e. the tests cannot tell the buggy code from the correct code.

A survivor is never ignored. It is one of: a **test gap** (add a test that kills it, then rerun the mutant), **equivalent** (the mutant changes nothing observable: prove it and document it), or **redundant code** (the original construct is dead or duplicated by another check: remove it, which also removes the mutant). Rule of this project: every mutant is killed, proven equivalent, or its code is removed.

## How the runners work (`tools/mutation/`)
- A runner holds a table `(name, kind, file, original text, mutated text)`; `kind` selects the build target and the tests that must fail (`gpu` = `test_vulkan` with a filter, `cpu` = `test_octree_reference`, `fmt` = reference tests first, `sort` = the sort tests).
- For each mutation: replace the text (it must occur **exactly once**), build the target, run the filtered tests, restore the original file. A first baseline run checks that the unmutated code builds and passes. At the end the sources are restored and rebuilt; the last line lists the survivors.
- `ONLY=<name prefix>` runs a subset (to rerun survivors after adding tests); `DONE_LOG=<log>` skips the ones a log shows as killed; `verify10.py <runner>` checks that every original text still occurs exactly once (run it after an interrupted run and after refactoring: a pattern count of 0 means the runner no longer matches the code, e.g. the mutated construct was removed).
- Usage and paths: [../../tools/README.md](../../tools/README.md).

## Rules and pitfalls learned
1. **One runner at a time.** The runners share one build tree and rewrite the sources; two runners corrupt each other. Do not edit a file a running runner mutates; tests and docs may be edited, atomically (write a temporary file, then `os.replace`), because the runner rebuilds from the working tree.
2. **Line endings:** the repository files use CRLF, runner patterns are written with `\n` and converted (`old.replace('\n', '\r\n')`); new files are written as CRLF. A pattern with a different indentation count is a common reason for "pattern count 0".
3. **Hanging mutants:** a wrong loop bound in a kernel can hang the GPU; the runners use a timeout per test run and count it as killed. The Intel watchdog turns an endless kernel into `ErrorDeviceLost` after about 5 s (K25).
4. **Equivalent boundary mutants** typically move a threshold that only decides *when* a loop stops (`stack_threshold + 1`, `idle_threshold + 1`): the result is the same. Document them with the reason; they are the only survivors the project accepts.
5. **A downstream check hides a boundary mutant:** `if (n > limit + 1)` survives when the next function throws for `n = limit + 1` anyway. Compare the **message** of the exception (see `tests/vulkan/test_size_limits.cpp`) or remove the redundant check.
6. **Interrupted runs:** run `verify10.py <runner>` (0 bad patterns = nothing left mutated); repair by hand from `git diff` if not. Never kill a runner with `taskkill` while it is mutating without checking afterwards.
7. Time: one mutation costs one incremental build plus one filtered test run: about 20 s in the S3 harness (5 s build, 6-14 s tests, [vulkan_prob_voxel_map.md](architecture/vulkan_prob_voxel_map.md)); rounds with slow tests take much longer (the balancer tests need about 3 minutes when they pass) and a hanging mutant costs the full timeout. Do not poll: one look at the log answers a progress question.
8. Large runner files are written with the Write tool or a script file, not with shell here-documents (quotes inside the mutation texts break them).

## Runners and results
The logs in `tools/mutation/logs/` are the first runs; the survivors they list were closed afterwards (new tests, removed code) and rerun, the final numbers are on the slice pages.

| Runner | Slice (page with the details) | Mutations | Result |
|---|---|---|---|
| `mutate.py`, `mutate2.py`, `mutate_m3.py` | S1 BitVoxelMap ([vulkan_bit_voxel_map.md](architecture/vulkan_bit_voxel_map.md)) | 8 (first round; `mutate2.py`, `mutate_m3.py` are later variants of the same shader harness) | all killed |
| `mutate3.py`, `mutate4.py` | S2 MetaPointCloud, kinematics ([vulkan_meta_cloud_and_robot.md](architecture/vulkan_meta_cloud_and_robot.md)) | 16 | all killed (one survivor closed by a FREE-only test) |
| `mutate5.py` | S4 voxel lists ([vulkan_voxel_lists.md](architecture/vulkan_voxel_lists.md)) | 48 | all killed (one survivor closed) |
| `mutate6.py` | P2 scan / sort / iota ([vulkan_primitives.md](architecture/vulkan_primitives.md)) | 13 | all killed (one equivalent replaced, `iota` test added) |
| `mutate7.py` | S3 ProbVoxelMap, shaders and C++ ([vulkan_prob_voxel_map.md](architecture/vulkan_prob_voxel_map.md)) | 63 | 61 killed, 2 equivalent |
| `mutate8.py` | S5 DistanceVoxelMap ([vulkan_distance_map.md](architecture/vulkan_distance_map.md)) | 71 | 69 killed, 2 result-equivalent, 3 redundant code paths removed |
| `mutate9.py` | S6a runtime, Morton, reference ([vulkan_octree_foundation.md](architecture/vulkan_octree_foundation.md)) | 83 | all killed |
| `mutate10.py` | S6b build / find / extract, 64-bit sort ([vulkan_octree_build.md](architecture/vulkan_octree_build.md)) | 138 | 137 killed, 1 provably equivalent |
| `mutate11.py` .. `mutate13.py` | S6c collisions: maps / lists / ids (63), tree vs tree (54), hardening round (15) ([vulkan_octree_collision.md](architecture/vulkan_octree_collision.md)) | 132 | all killed or removed as redundant, gaps closed by new tests |
| `mutate14.py` | S6c load balancer ([vulkan_octree_collision.md](architecture/vulkan_octree_collision.md#load-balancer-second-implementation)) | 66 | 60 killed, 4 equivalent boundary mutants, 2 redundant constructs removed |
| `mutate15.py` | block-pair counter of the reference (`tests/reference/octree_reference.h`, checkpoint) | 6 | all killed |
| `mutate_s6d1_reference.py` | S6d-1 / S6d-3 dynamic reference (`tests/reference/octree_dynamic_reference.h`) | 39 | 38 killed, 1 equivalent for consistent trees |
| `mutate_s6d2_insert.py` | S6d-2 / S6d-3 insert and propagate, shaders and host ([vulkan_octree_dynamic.md](architecture/vulkan_octree_dynamic.md)) | 77 | 71 killed, 6 explained survivors |
| `mutate_s6d4_rebuild.py` | S6d-4 rebuild compaction: scan / copy shaders, host code, `needs_rebuild` policy ([vulkan_octree_dynamic.md](architecture/vulkan_octree_dynamic.md)) | 30 | 26 killed, 4 explained survivors (CP9 flags: live nodes carry none; H6 / N4: only speed; H7: the address range check cannot fail on this device) |
| `mutate_s6d5_fast_insert.py` | S6d-5 fast insert: count kernel, batches, GPU sort path, bounded propagate and its kernels ([vulkan_octree_dynamic.md](architecture/vulkan_octree_dynamic.md)); the S6d-2 runner got 10 repaired patterns (all killed) and now runs under GPU assisted validation too | 34 | 27 killed, 7 equivalent (K4, K5, K6, S5, P4, P6, Q6) |
| `mutate16.py` | K34 size limits ([known_issues.md](known_issues.md)) | 12 | see the K34 row |

The older runners (`mutate.py` .. `mutate8.py`) are kept as they were written; they have their own small harness (shaders only) and may need adjusted test filters if the tests were renamed since. `mutate9.py` and later share the table format described above.

## Mutants that follow wrong device addresses
A mutant of pointer code makes later kernels read or write through wrong addresses; on the GPU that is a page fault (Windows event `nvlddmkm` 153, no TDR) and stresses the driver. `mutate_s6d4_rebuild.py` therefore runs the tests with **GPU assisted validation in safe mode** (`VK_KHRONOS_VALIDATION_GPUAV_ENABLE=true`, `..._GPUAV_SAFE_MODE=true`, sync and core validation off): the instrumented shader skips the invalid access and the layer reports it, which the fixture counts as a failure. Measured 2026-10-31: no new event 153 over 25 mutants, tests about ten times slower. Earlier runners do not do this; copy the `test_env` lines when a runner mutates pointers. A mutant killed this way is killed by the validation, not by a value comparison (S2).

## Open ideas (not done)
- Run several runners in parallel from separate checkouts / build trees on a machine with many cores (not tried; the current runners must not share a tree).
- Mutate the host-side C++ of S1-S5 more thoroughly (the first runners concentrate on the shaders).
