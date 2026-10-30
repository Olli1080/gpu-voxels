# Working Rules

## Scope and workspace
- Work **only** in `C:\Code\2026-10-22_gpu-voxels` (branch `master`, remote `origin` = Olli1080/gpu-voxels, `upstream` = FZI).
- `C:\Code\gpu-voxels` and `C:\Code\gpu-voxels__` are **read-only reference material** (failed attempts). Do not edit, clean, reset or build there. `C:\Code\gpu-voxels` holds ~300 uncommitted changes and 49 uncommitted `.comp` shaders that exist nowhere else — see [workspace_layout.md](../project_state/workspace_layout.md).
- Use absolute paths in commands. The shell's working directory drifts between tool calls.

## Git
- Local commits on the work branch `vulkan-migration` are allowed (user decision Q11, 2026-10-29). Do not push, open PRs, delete branches or touch the `vcpkg` submodule pointer unless the user asks.
- Never `git reset --hard`, `git clean`, force-push or rewrite history without explicit instruction.
- Work for a migration belongs on a dedicated branch created at the user's request, not on `master`.
- Commit messages and PR text describe what is *verified*, not what is planned.

## Destructive or irreversible actions
- Look at a target before deleting/overwriting it. Prefer moving to a clearly named folder over deleting.
- Removing a public API, a feature, a test, or a code path in order to make something compile is **forbidden** (see [migration_guardrails.md](migration_guardrails.md) rule 3). If it truly must go, ask first.

## Asking vs. deciding
- Ask the user when the choice is architectural (graphics API binding, shader language, whether CUDA stays, supported hardware, scope of visualizer/ROS/examples). Collect pending questions in [open_questions.md](../project_state/open_questions.md).
- Decide yourself on naming, file layout inside `Agent/`, and other cheap, reversible things; mention the decision.

## Reporting
- State outcomes faithfully: failing tests, skipped steps, unverified assumptions are part of the report.
- Distinguish **measured** (you ran it) from **inferred** (you read it) from **claimed** (a document says so).
- Plan checkboxes, commit messages and earlier agent summaries are *claims*. The previous attempts marked phases complete while the code contained stubs (see [mistakes_to_avoid.md](../knowledge_base/vulkan_migration/mistakes_to_avoid.md)).

## Temporary files
- Scratch output (logs, test dumps, experiments) goes to the session scratchpad or an ignored `build/` dir, never the repo root. The failed attempt left 15 `test_octree_build_out_v*.txt` files and `test_minimal_out.txt` (UTF-16 PowerShell dumps) in its root.
- Do not redirect native-program output with PowerShell `>` into files that will be read later; it writes UTF-16 and wraps stderr lines in `NativeCommandError` records. Use `cmd /c "... > file 2>&1"` or Bash.

## Tooling notes for this machine
- Windows 11, PowerShell + Git Bash. CMake and Ninja present; Visual Studio 2022 and 2026 (folder "18") installed.
- No `nvcc`, no `cl` on PATH. Vulkan SDK 1.4.363.0 is installed (`VULKAN_SDK=C:\VulkanSDK\1.4.363.0`; tools incl. glslang/glslc/dxc/slangc and the validation layer in `%VULKAN_SDK%\Bin`, not on PATH). Details: [current_state.md](../project_state/current_state.md).
- Downloads/installs need the user's go-ahead (state source and purpose first). Reference project for CMake/package conventions: `C:\Code\REACH` ([reach_reference.md](../knowledge_base/reach_reference.md)).
