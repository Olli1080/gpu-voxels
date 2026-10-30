# Plan: Agent Folder Bootstrap (2026-10-22) — DONE

## Goal
Investigate the workspace (this repo + the two failed-migration folders) and record findings in a structured `Agent/` folder with index files, plus a short root `AGENTS.md`.

## Steps (all done)
1. Survey structure, CMake, dependencies, git state of the three workspaces.
2. Post-mortem of earlier SYCL/Kompute attempts.
3. Write `agent_policies/`, `knowledge_base/`, `project_state/`, `Analysis/`, `plans_wip/`, `plans_done/`, each with `summary.md`.
4. Write root `AGENTS.md`.
5. Verify links and file sizes.

## Outcome
- Nothing under `packages/`, build files or the sibling folders was modified. No commit was made.
- Verified by script: every relative markdown link resolves; all files ≤ 400 lines (see snippet).
- Not verified: anything requiring CUDA or a Vulkan build (see [current_state.md](../project_state/current_state.md)).
- Follow-up: user answers to [open_questions.md](../project_state/open_questions.md).

## Link/size check (run from repo root)
```bash
python - <<'EOF'
import re, pathlib
root = pathlib.Path('.')
files = list(pathlib.Path('Agent').rglob('*.md')) + [pathlib.Path('AGENTS.md')]
bad = 0
for f in files:
    text = f.read_text(encoding='utf-8')
    if text.count('\n') + 1 > 400: print('TOO LONG', f); bad += 1
    for m in re.finditer(r'\]\(([^)#]+)(#[^)]*)?\)', text):
        t = m.group(1)
        if t.startswith(('http', 'mailto')): continue
        if not (f.parent / t).resolve().exists(): print('DEAD', f, t); bad += 1
print('files', len(files), 'problems', bad)
EOF
```
Also check that every folder has `summary.md` and that every `.md` in a folder is linked from its `summary.md`.
