# Hand-written mutation runner (CUDA parity, scenario 1: list x map collisions): one mutation at a time, build, run only the parity test, restore.
# The parity test must notice every change of the collision semantics, because the golden file comes from a CUDA run.
# Usage: python mutate_cuda_parity.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk. Rebuild before using the executable afterwards.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
SH = root + 'packages/gpu_voxels/gpu_voxels/vulkan/shaders/list_collide.comp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

MUT = [
    ('P1 default: the list voxel is not inspected (S4 behaviour)', SH, "if (list_bits == 0u)\n                continue;", ""),
    ('P2 raw rule: uses the converted level', SH, "map_occupied = float(int(map_vox[index])) >= pc.threshold;", "map_occupied = int(map_vox[index]) >= pc.level;"),
    ('P3 mask: no shared bit needed', SH, "if (shared_bits == 0u)\n                continue;", ""),
    ('P4 types: the bits are not collected', SH, "atomicOr(result[1u + w], list_vox[lb + w]);", "atomicOr(result[1u + w], 0u);"),
    ('P5 offset: 3D bounds check replaced by the array end', SH, "if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= int(pc.dim_x) || c.y >= int(pc.dim_y) || c.z >= int(pc.dim_z))",
     "if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= int(pc.dim_x) + 1 || c.y >= int(pc.dim_y) || c.z >= int(pc.dim_z))"),
    ('P6 bit map: FREE counts as occupied', SH, "uint map_bits = map_vox[mb] & 0xFFFFFFFEu;", "uint map_bits = map_vox[mb];"),
]

TESTS = 'VulkanTest.CudaParity*'
only = tuple(sys.argv[1].split(',')) if len(sys.argv) > 1 and sys.argv[1] else ()
originals = {}


def read(path):
    if path not in originals:
        originals[path] = open(path, encoding='utf-8', newline='').read()
    return originals[path]


try:
    for name, path, old, new in MUT:
        if only and not name.startswith(only):
            continue
        orig = read(path)
        nl = '\r\n' if '\r\n' in orig else '\n'
        text = orig.replace('\r\n', '\n')
        if text.count(old) != 1:
            print(name, ': PATTERN COUNT', text.count(old), flush=True)
            continue
        open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new).replace('\n', nl))
        b = subprocess.run(['cmake', '--build', root + 'build/verify-vk', '--config', 'Debug', '--target', 'test_vulkan', '--parallel'],
                           env=env, capture_output=True, text=True)
        if b.returncode != 0:
            print(name, ': BUILD FAILED', flush=True)
            open(path, 'w', encoding='utf-8', newline='').write(orig)
            continue
        try:
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_filter=' + TESTS],
                               env=env, capture_output=True, text=True, timeout=600)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
