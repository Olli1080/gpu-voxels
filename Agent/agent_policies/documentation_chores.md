# Documentation Chores (maintaining `Agent/`)

## Index rule
- Every folder under `Agent/` has `summary.md`. In a folder with subfolders it links to each subfolder's `summary.md`; in a leaf folder it links to each file with a one-line description.
- Adding, renaming, splitting or deleting a file ⇒ update the folder's `summary.md` **in the same change**. Also update the parent index if a folder is added/removed.
- Links are relative. After edits, verify that no link is dead (a link-check one-liner is in [plans_done/2026-10-22_agent_folder_bootstrap.md](../plans_done/2026-10-22_agent_folder_bootstrap.md)).

## Size rule
- Target ≤ 400 lines per file; treat anything larger as a split candidate (warn at 300).
- Splitting: new **file** when a new aspect appears; new **subfolder** (with its own `summary.md`) when a topic reaches ≥ 4 files. Never mix orthogonal aspects in one file (e.g. don't put a build recipe into an architecture description).
- When splitting, leave no redirect stubs; fix all inbound links instead (`grep -r "oldname" Agent AGENTS.md`).

## Where does it go?
| You have… | Put it in |
|---|---|
| A rule the next agent must obey | `agent_policies/` |
| Result of an investigation ("why is X slow", "what uses Y") | `Analysis/<YYYY-MM-DD>_<topic>.md` (use `_template.md`) |
| A lasting fact about the code, tools, drivers; a known bug + workaround | `knowledge_base/` |
| A proposal or in-progress work breakdown | `plans_wip/` |
| A finished plan | move to `plans_done/`, add outcome + evidence section, fix indexes |
| Current status, open questions, workspace facts | `project_state/` |

## Plan lifecycle
1. Draft in `plans_wip/<topic>.md` with: goal, non-goals, steps with *verification level* (see [definition_of_done.md](definition_of_done.md)), risks, **Exceptions table** (stub register).
2. Update the plan as work proceeds: tick a step only with the evidence link/command next to it.
3. On completion move the file to `plans_done/`, append *Outcome* (what was verified, what was not).
4. A plan that is abandoned moves to `plans_done/` with status "abandoned" and the reason — failures are knowledge.

## Writing style
- Facts first, short sentences, tables for comparisons. Mark **measured / inferred / claimed**.
- Give paths and commands exactly (copy-pasteable). Give commit hashes/branch names for anything in the reference repos.
- Don't paste long logs; summarise and name the log's location.

## End-of-session chore
Before finishing a session: update [current_state.md](../project_state/current_state.md) (what changed, what is verified), move newly learned pitfalls into `knowledge_base/`, and list unanswered questions in [open_questions.md](../project_state/open_questions.md).
