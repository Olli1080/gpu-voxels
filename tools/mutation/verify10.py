import os, sys
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, sys.argv[1] if len(sys.argv) > 1 else 'mutate10.py'), encoding='utf-8').read()
start = src.index('ROOT =')
end = src.index('\nonly = os.environ')
ns = {'os': os, '__file__': os.path.abspath(__file__)}
exec(src[start:end], ns)
bad = 0
for name, kind, rel, old, new in ns['mutations']:
    t = open(os.path.join(ns['ROOT'], rel), 'rb').read().decode()
    c = t.count(old.replace('\n', '\r\n')) if '\r\n' in t else t.count(old)
    if c != 1:
        print('pattern count', c, ':', name)
        bad += 1
print('checked', len(ns['mutations']), 'bad', bad)
