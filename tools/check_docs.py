"""Doc checks for Agent/: relative links resolve, every .md is linked from its folder's summary.md, files <= 400 lines."""
import os
import re
import sys

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
AGENT = os.path.join(ROOT, 'Agent')
problems = []
files = []
for d, _, names in os.walk(AGENT):
    for n in names:
        if n.endswith('.md'):
            files.append(os.path.join(d, n))

link_re = re.compile(r'\[[^\]]*\]\(([^)\s]+)\)')
for f in files:
    text = open(f, encoding='utf-8').read()
    if text.count('\n') + 1 > 400:
        problems.append(f'{f}: more than 400 lines')
    for target in link_re.findall(text):
        if target.startswith(('http://', 'https://', 'mailto:', '#')):
            continue
        path = target.split('#')[0]
        if not path:
            continue
        resolved = os.path.normpath(os.path.join(os.path.dirname(f), path))
        if not os.path.exists(resolved):
            problems.append(f'{os.path.relpath(f, ROOT)}: broken link {target}')

for d, subdirs, names in os.walk(AGENT):
    summary = os.path.join(d, 'summary.md')
    if not os.path.exists(summary):
        problems.append(f'{os.path.relpath(d, ROOT)}: no summary.md')
        continue
    text = open(summary, encoding='utf-8').read()
    for n in names:
        if n.endswith('.md') and n != 'summary.md' and n not in text:
            problems.append(f'{os.path.relpath(summary, ROOT)}: does not list {n}')
    for s in subdirs:
        if s not in text:
            problems.append(f'{os.path.relpath(summary, ROOT)}: does not list folder {s}')

print(f'{len(files)} files, {len(problems)} problems')
for p in problems:
    print(' ', p)
sys.exit(1 if problems else 0)
