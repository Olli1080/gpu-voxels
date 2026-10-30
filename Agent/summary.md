# Agent Folder — Index

Central place for everything an agent needs to work in this repository (GPU-Voxels, CUDA → Vulkan migration).
Entry point from the repo root: [../AGENTS.md](../AGENTS.md).

| Folder | Purpose | Index |
|---|---|---|
| `agent_policies/` | What you **must** respect: chores, restrictions, definition of done, migration guardrails | [summary](agent_policies/summary.md) |
| `Analysis/` | Results of investigation tasks (one file per investigation) + template | [summary](Analysis/summary.md) |
| `knowledge_base/` | Emerging knowledge: architecture of the CUDA code, lessons of failed migrations, known issues/workarounds | [summary](knowledge_base/summary.md) |
| `plans_wip/` | Plans currently being worked on or proposed | [summary](plans_wip/summary.md) |
| `plans_done/` | Completed plans (kept for history) | [summary](plans_done/summary.md) |
| `project_state/` | Where the project stands right now, workspace layout, open questions | [summary](project_state/summary.md) |

## Conventions
- Every folder (and subfolder) contains a `summary.md` that links to its children; in leaf folders it links to every file with a one-line description.
- Files stay ≤ 400 lines. Above that, split (new file for a new aspect, new subfolder when a topic gets ≥ 4 files) and update the indexes. Details: [documentation_chores.md](agent_policies/documentation_chores.md).
- One aspect per file. Facts that were *measured* are marked as such; things that are *inferred* are marked "inferred".
- Dates are absolute (YYYY-MM-DD).

## Fastest path for a new agent
1. [project_state/current_state.md](project_state/current_state.md) — what exists, what builds.
2. [agent_policies/migration_guardrails.md](agent_policies/migration_guardrails.md) and [knowledge_base/vulkan_migration/mistakes_to_avoid.md](knowledge_base/vulkan_migration/mistakes_to_avoid.md).
3. [knowledge_base/architecture/summary.md](knowledge_base/architecture/summary.md) — how the CUDA library is built.
4. [plans_wip/summary.md](plans_wip/summary.md) — what is planned next.
5. [../tools/README.md](../tools/README.md) — verification scripts (mutation runners, `check_docs.py`); [project_state/machine_change_14900k.md](project_state/machine_change_14900k.md) — the planned machine change.
