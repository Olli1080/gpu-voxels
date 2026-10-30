# Definition of Done

"Done", "migrated", "working", "verified" and "parity" are reserved words. Use them only when the evidence below exists.

## Levels (use these exact words in plans and reports)
| Level | Meaning | Evidence required |
|---|---|---|
| **Designed** | Approach written down | Plan file in `plans_wip/` |
| **Compiles** | Target builds with warnings reviewed | Build log excerpt (command + result) |
| **Runs** | Executes on a real Vulkan device without validation-layer errors | Command, device name, validation-layer result |
| **Tested** | Automated test exercises the behaviour and *can fail* | Test name + output; test compares against an **independent** reference |
| **Parity** | Output equals the original (CUDA) behaviour | Same inputs run through CUDA original, results compared; or golden data produced by the CUDA build and checked in |
| **Complete** | Parity for the whole public surface of the component, no stub, no `TODO: port`, no disabled code path | Checklist per method of the component's public header |

## What does NOT count as evidence
- A checked box, a commit message, or an agent's summary.
- A test whose "gold standard" is code written by the same agent in the same session (previous attempt: `TestGoldStandard.cpp` re-implemented `BitVector` with mocked CUDA keywords and compared against itself).
- A function that returns `0` / `false` / does nothing so that a caller links (previous attempt: `ProbVoxelMap::collideWith` returned 0 while the phase was marked done).
- A run on a CPU fallback presented as a GPU result.
- Output that was not actually read (e.g. test log dumped in UTF-16 and never inspected).

## Banned shortcuts
- Empty/`return false`/`return 0` bodies, `// Stub`, `// Placeholder`, `#if 0` around real logic, `if(OFF)` around CMake targets — unless the item is listed in an *Exceptions* table of the active plan with the user's consent.
- Mock headers that fake CUDA (`cuda_runtime_api.h` with `#define __global__`) in the **test** tree.
- Hard-coded absolute paths to a user/temp directory (previous attempt included a file from `C:/Users/olive/.gemini/tmp/...`).
- Duplicating a source as both `.cu` and `.cpp` in the tree.

## Stub register
If something must be temporarily unimplemented, add one line to the active plan's *Exceptions* table:
`symbol | file | reason | exit criterion | owner/date`. A phase cannot be closed while its table is non-empty.

## Hardware caveat
No NVIDIA GPU/CUDA on this machine. "Parity with CUDA" therefore needs either (a) golden data generated elsewhere and committed with provenance, or (b) a user-run comparison. Until then the strongest honest claim is **Tested** against an independent CPU reference.
