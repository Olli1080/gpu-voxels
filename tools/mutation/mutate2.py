"""Mutation check of the BitVoxelMap tests: each shader mutation must make at least one test fail."""
import os, re, shutil, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SH = os.path.join(ROOT, r'packages\gpu_voxels\gpu_voxels\vulkan\shaders')
EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

originals = {f: open(os.path.join(SH, f), 'rb').read() for f in os.listdir(SH)}

def restore():
    for f, data in originals.items():
        open(os.path.join(SH, f), 'wb').write(data)

def build():
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', 'test_vulkan'],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0

def run_tests():
    r = subprocess.run([EXE, '--gtest_filter=*BitMap*'], capture_output=True, text=True, env=env)
    failed = re.findall(r'^\[  FAILED  \] VulkanTest\.(\w+) \(', r.stdout, re.M)
    return failed, r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ''

mutations = [
    ('T1 low bits not masked (0xE instead of 0xF0)', 'bit_voxel_collide_types.comp', 'c &= 0xFFFFFFF0u;', 'c &= 0xFFFFFFFEu;'),
    ('T2 colliding types not aggregated', 'bit_voxel_collide_types.comp', 'atomicOr(group_types[w], collided[w]);', ';'),
    ('T3 types written to wrong result word', 'bit_voxel_collide_types.comp', 'atomicOr(result[1u + gl_LocalInvocationIndex]', 'atomicOr(result[2u + gl_LocalInvocationIndex % 7u]'),
    ('T4 collision bit not written', 'bit_voxel_collide_types.comp', 'mine[base + meaning_word(BVM_COLLISION)] |= meaning_mask(BVM_COLLISION);', ''),
    ('T5 count not accumulated', 'bit_voxel_collide_types.comp', 'atomicAdd(result[0], group_count);', ';'),
]

try:
    for name, fname, old, new in mutations:
        restore()
        path = os.path.join(SH, fname)
        text = open(path, 'rb').read().decode('utf-8')
        # files may have CRLF line endings; patterns are single-line
        if old not in text:
            print(f'{name}: PATTERN NOT FOUND in {fname}')
            continue
        open(path, 'wb').write(text.replace(old, new, 1).encode('utf-8'))
        if not build():
            print(f'{name}: BUILD FAILED')
            continue
        failed, last = run_tests()
        verdict = 'KILLED' if failed else 'SURVIVED  <-- weak tests'
        print(f'{name}: {verdict}; failing: {", ".join(failed) if failed else "-"}')
finally:
    restore()
    ok = build()
    failed, last = run_tests()
    print('restored, build ok =', ok, '| baseline:', last)
