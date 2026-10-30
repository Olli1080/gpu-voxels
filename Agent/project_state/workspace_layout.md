# Workspace Layout (three checkouts of the same project)

All under `C:\Code\`. Only the first is a working area.

| Folder | Branch / HEAD | State (measured 2026-10-22) | Role |
|---|---|---|---|
| `2026-10-22_gpu-voxels` | `master` @ `dd47249` (2025-06-23), tracks `origin/master` | Clean; now also contains `Agent/` + `AGENTS.md` | **Working area for the new Vulkan attempt** |
| `gpu-voxels` | `kompute` @ `43a68a5` (2026-05-02); other local branches `kompute_backend`, `kompute_migration` (= master), `sycl_migration`, `master` | **303 uncommitted paths** (203 modified, 1 deleted, 99 untracked). Contains the only copy of 14 shaders and the bridge headers' later edits | **Read-only archive** of attempts 1–3 ([attempt_history.md](../knowledge_base/vulkan_migration/attempt_history.md)). Has its own `agent/`, `plans/`, `MAIN_PLAN.md`, `AGENTS.md` (old conventions: `agent/policies`, `plans_wip`, … — superseded by this `Agent/` folder) |
| `gpu-voxels__` | `kompute_backend` unborn (no commits), `master` exists; `vcpkg` submodule metadata broken (`.git/modules/vcpkg` missing) | Working tree ≈ attempt 2 layout (macro bridge `GVL_*`, `parallel::`), 159 paths differ from master, `core/src` and `gpu_voxels/src` missing; `AGENTS.md` links to a non-existent `agent/summary.md` | **Dead copy**; ignore except for curiosity |

## Safe ways to look at the archives
```bash
git -C C:/Code/gpu-voxels log --oneline --all --graph | head -40
git -C C:/Code/gpu-voxels show <branch>:<path>
git -C C:/Code/gpu-voxels diff --stat master..kompute_backend
```
Never in those folders: `checkout`, `reset`, `clean`, `stash`, builds that write into the tree, or editing. To compare file contents without git, read them directly.

## Relationship diagram (text)
```
upstream FZI master ──► origin master (dd47249, 2025-06-23) ◄── 2026-10-22_gpu-voxels  (work here)
                              │
                              ├── sycl_migration (5 commits, Mar 26–Apr 1)
                              │       └── kompute_backend (+11, Apr 2–8)      ≈ gpu-voxels__ working tree
                              └── kompute (14 commits, Apr 26–May 2) + 303 dirty files  = C:\Code\gpu-voxels
                                      (kompute_migration == master: empty)
```

## Sync hygiene
This workspace's `.git` and the archive's `.git` are separate clones of the same GitHub repo (`origin`). Pushed: `origin/sycl_migration` at `df54593` (4 of the 5 SYCL commits). **Local only in the archive:** `173f12c` (last SYCL commit), branches `kompute_backend`, `kompute` and everything uncommitted. If that disk is lost, attempts 2–3 are lost. User decision (Q8): leave it as is — see [open_questions.md](open_questions.md).
