"""Mutation check for slice S2 (meta cloud transform / insertion / self collision / DH chain)."""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
VK = os.path.join(ROOT, r'packages\gpu_voxels\gpu_voxels\vulkan')
EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

mutations = [
    ('Q10 self collision: FREE bit counts as collision partner', r'shaders\bit_voxel_insert_meta_selfcoll.comp', '(masked[w] & 0xFFFFFFFEu)', '(masked[w] & 0xFFFFFFFFu)'),
]

def read(path):
    return open(path, 'rb').read()

def build():
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', 'test_vulkan'],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0

def run_tests():
    r = subprocess.run([EXE], capture_output=True, text=True, env=env)
    return re.findall(r'^\[  FAILED  \] VulkanTest\.(\w+) \(', r.stdout, re.M)

touched = {}
try:
    for name, rel, old, new in mutations:
        path = os.path.join(VK, rel)
        original = touched.setdefault(path, read(path))
        text = original.decode('utf-8')
        crlf = '\r\n' in text
        o = old.replace('\n', '\r\n') if crlf else old
        n = new.replace('\n', '\r\n') if crlf else new
        if o not in text:
            print(f'{name}: PATTERN NOT FOUND in {rel}', flush=True)
            continue
        open(path, 'wb').write(text.replace(o, n, 1).encode('utf-8'))
        ok = build()
        if not ok:
            print(f'{name}: BUILD FAILED', flush=True)
        else:
            failed = run_tests()
            print(f'{name}: {"KILLED" if failed else "SURVIVED  <-- weak tests"}; failing: {", ".join(failed) if failed else "-"}', flush=True)
        open(path, 'wb').write(original)
finally:
    for path, data in touched.items():
        open(path, 'wb').write(data)
    ok = build()
    failed = run_tests()
    print('restored; build ok =', ok, '; failing tests at baseline:', failed, flush=True)
