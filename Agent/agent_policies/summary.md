# agent_policies — Index

Binding rules for every agent working in this repository. Read all of them once; re-read the one that applies before acting.

| File | Covers |
|---|---|
| [working_rules.md](working_rules.md) | General behaviour: scope, git, destructive actions, asking questions, reporting honestly |
| [definition_of_done.md](definition_of_done.md) | What "done / verified / working" means; evidence required; banned shortcuts |
| [migration_guardrails.md](migration_guardrails.md) | Hard rules for the CUDA → Vulkan migration, each derived from a concrete past failure |
| [repository_restrictions.md](repository_restrictions.md) | Files/dirs that must not be touched, encoding/line-ending/license constraints, build-output hygiene |
| [documentation_chores.md](documentation_chores.md) | How to maintain `Agent/` (indexes, ≤400-line rule, plan lifecycle, what to log where) |

Precedence: explicit user instruction > these policies > individual plan documents.
If a policy blocks something the user asked for, say so and ask — do not silently deviate.
