"""Checks that every mutation's ORIGINAL pattern is present in its source file (i.e. no file is left mutated)."""
import os

src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mutate9.py'), encoding='utf-8').read()
start = src.index('ROOT =')
end = src.index('\nonly = os.environ')
ns = {'os': os, '__file__': os.path.abspath(__file__)}
exec(src[start:end].replace("GPU_FILTER = ", "GPU_FILTER = ").replace("env = dict(os.environ, VULKAN_SDK=r'C:\\VulkanSDK\\1.4.363.0')", ""), ns)
bad = 0
for name, kind, rel, old, new in ns['mutations']:
    t = open(os.path.join(ns['ROOT'], rel), 'rb').read().decode()
    if old.replace('\n', '\r\n') not in t:
        print('original pattern missing:', name)
        bad += 1
print('checked', len(ns['mutations']), 'missing', bad)
