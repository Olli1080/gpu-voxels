"""Mutation check for slice P2 (scan / radix sort / iota). Every mutation must be killed by the list tests."""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
VK = os.path.join(ROOT, r'packages\gpu_voxels\gpu_voxels\vulkan')
EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))
FILTER = os.environ.get('MUT_FILTER', '--gtest_filter=*Scan*:*Sort*:*Primitives*:*List*:-*MillionPoints:*MoreThanOneUploadChunk')

S = lambda n: 'shaders\\' + n
mutations = [
    ('P1 scan: wrong block total', S('scan_blocks.comp'), 'sums[gl_WorkGroupID.x] = temp[255];', 'sums[gl_WorkGroupID.x] = temp[254];'),
    ('P2 scan: second element of a pair', S('scan_blocks.comp'), 'out_data[i1] = exclusive + a;', 'out_data[i1] = exclusive;'),
    ('P3 scan: too few scan rounds', S('scan_blocks.comp'), 'offset < 256u', 'offset < 128u'),
    ('P4 scan_add: wrong block of an element', S('scan_add.comp'), 'offsets[i >> 9u]', 'offsets[i >> 8u]'),
    ('P5 histogram: key mask ignored', S('radix_histogram.comp'), '& pc.mask', '& 255u'),
    ('P6 scatter: rank counts equal digits <=', S('radix_scatter.comp'), 'for (uint j = 0u; j < t; ++j)', 'for (uint j = 0u; j <= t; ++j)'),
    ('P7 scatter: rank counts later elements (unstable)', S('radix_scatter.comp'), 'for (uint j = 0u; j < t; ++j)', 'for (uint j = 1u; j < t; ++j)'),
    ('P8 histogram: layout not digit-major', S('radix_histogram.comp'), 'hist[t * pc.groups + gl_WorkGroupID.x]', 'hist[t + gl_WorkGroupID.x * 256u]'),
    ('P9 iota: off by one', S('fill_iota.comp'), 'out_data[i] = i;', 'out_data[i] = i + 1u;'),
    ('P10 cpp: second scan level skipped for 2 blocks', r'src\Primitives.cpp', 'if (blocks > 1)', 'if (blocks > 2)'),
    ('P11 cpp: odd pass count result not copied back', r'src\Primitives.cpp', 'batch.copy(*src_keys, keys, bytes);\n\t\t\tbatch.copy(*src_values, values, bytes);', ''),
    ('P12 cpp: mask of the last pass', r'src\Primitives.cpp', 'bits_left >= 8 ? 255u : ((1u << bits_left) - 1u)', '255u'),
    ('P13 cpp: pass count rounds down', r'src\Primitives.cpp', '(key_bits + 7) / 8', 'key_bits / 8 + (key_bits == 32 ? 0 : 0)'),
]

only = os.environ.get('ONLY')


def read(path):
    return open(path, 'rb').read()


def build():
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', 'test_vulkan'],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0


def run_tests():
    r = subprocess.run([EXE, FILTER], capture_output=True, text=True, env=env)
    return re.findall(r'^\[  FAILED  \] VulkanTest\.(\w+)', r.stdout, re.M), r.returncode


survivors = []
touched = {}
try:
    for name, rel, old, new in mutations:
        if only and not name.startswith(only):
            continue
        path = os.path.join(VK, rel)
        original = touched.setdefault(path, read(path))
        text = original.decode('utf-8')
        crlf = '\r\n' in text
        o = old.replace('\n', '\r\n') if crlf else old
        n = new.replace('\n', '\r\n') if crlf else new
        if o not in text:
            print(f'{name}: PATTERN NOT FOUND in {rel}', flush=True)
            survivors.append(name + ' (pattern)')
            continue
        open(path, 'wb').write(text.replace(o, n, 1).encode('utf-8'))
        if not build():
            print(f'{name}: BUILD FAILED', flush=True)
            survivors.append(name + ' (build)')
        else:
            failed, code = run_tests()
            if failed or code != 0:
                print(f'{name}: KILLED by {len(failed)} test(s): {", ".join(sorted(set(failed))[:3])}', flush=True)
            else:
                print(f'{name}: SURVIVED  <-- weak tests', flush=True)
                survivors.append(name)
        open(path, 'wb').write(original)
finally:
    for path, data in touched.items():
        open(path, 'wb').write(data)
    ok = build()
    failed, code = run_tests()
    print('restored; build ok =', ok, '; failing tests at baseline:', failed, 'exit', code, flush=True)
print('SURVIVORS:', survivors, flush=True)
