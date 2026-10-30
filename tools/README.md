# tools/ — scripts used to verify the Vulkan migration

Plain Python 3 scripts without third-party dependencies (written and run with Python 3.13 on Windows). They find the repository
themselves (two folders up from `tools/mutation/`, one up from `tools/`); `GVL_ROOT` overrides that, `VULKAN_SDK` overrides the SDK path
(default `C:\VulkanSDK\1.4.363.0`). Why mutation testing and how the runners work: [../Agent/knowledge_base/mutation_testing.md](../Agent/knowledge_base/mutation_testing.md).

| Path | Content |
|---|---|
| `check_docs.py` | Checks `Agent/`: relative links resolve, every `.md` is listed in the `summary.md` of its folder, every folder has a `summary.md`, no file over 400 lines. Run after every documentation change: `python tools/check_docs.py` (exit code 1 on problems). |
| `mutation/mutate*.py` | One mutation runner per slice or review round (list and meaning: the mutation testing page). Each runner holds a table of mutations (name, kind, file, original text, mutated text), builds and runs the tests for every mutation and prints `KILLED` / `SURVIVED`. They **modify the sources while they run** (restored afterwards): run one at a time, never edit the mutated files meanwhile. |
| `mutation/verify9.py`, `verify10.py <runner>` | Check that every original pattern of a runner still occurs exactly once in its source file, i.e. that no file was left mutated (after an interrupted run) and that the runner still matches the code. |
| `mutation/logs/` | The logs of the runs as they happened (first runs, with the survivors that were closed afterwards by new tests or by removing redundant code; the final state of each slice is in its page under `Agent/knowledge_base/architecture/`). `mutate6.log` is the log of a run that aborted with a syntax error and was repeated. |
| `vk_layer_settings_sync_validation.txt` | `vk_layer_settings.txt` used for the synchronization validation experiment of K20 (copy next to the executable or point `VK_LAYER_SETTINGS_PATH` at it). |

## Running a runner (example)
```
python tools/mutation/verify10.py mutate14.py          # every pattern still present exactly once?
python tools/mutation/mutate14.py > mutate14.log       # all mutations (about one build + test run per mutation)
set ONLY=T10 & python tools/mutation/mutate14.py       # only the mutations whose name starts with T10 (mutate5 .. 14, 16)
```
The runners expect the build directory `build/verify-vk` (Vulkan on, Debug; it also contains `test_octree_reference`); only `mutate15.py`
uses `build/verify-cpu` ([build_system.md](../Agent/knowledge_base/architecture/build_system.md)). `DONE_LOG=<log>` (mutate7 .. 14) skips the mutations that a log shows as killed.
