# knowledge_base — Index

Emerging knowledge about this project: how the CUDA library is built, what earlier migration attempts taught us, known issues and workarounds.
Facts are tagged **measured** (command run in this workspace), **inferred** (read from code), or **claimed** (stated by a document, not re-verified).

| Folder / file | Content |
|---|---|
| [architecture/](architecture/summary.md) | Structure of the current CUDA library: modules, data model, CUDA/Thrust coupling inventory, build system, visualization/IPC |
| [vulkan_migration/](vulkan_migration/summary.md) | Everything about moving to Vulkan: history of failed attempts, mistakes to avoid, Kompute/GLSL gotchas, reusable assets, design options |
| [reach_reference.md](reach_reference.md) | Reference project `C:\Code\REACH`: package/CMake conventions, test setup, Vulkan practice and lessons to reuse |
| [mutation_testing.md](mutation_testing.md) | How the Vulkan code is mutation-tested: method, the runners in `tools/mutation/`, pitfalls, results per slice |
| [known_issues.md](known_issues.md) | Known bugs, environment pitfalls and workarounds (build, tooling, repo hygiene) |

Add new topics as new files in the matching folder; add a new folder only when a topic reaches ≥ 4 files.
