# Hand-written mutation runner (slice S7-5, second part: ProbVoxels, ProbMap2DistMap and RaycastingSensorExample as smoke tests): one mutation at a
# time, build, run the three example tests, restore. These places are covered by class-level tests already; the point is that the example call
# sequences notice them too.
# Usage: python mutate_s75_examples2.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk. Rebuild before using the executable afterwards.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
PC = V + 'src/ProbVoxelMap.cpp'
DC = V + 'src/DistanceVoxelMap.cpp'
RAY = V + 'shaders/prob_ray.glsl'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

MUT = [
    ('X1 prob: swept volume meaning minus 128', PC, "static_cast<int32_t>(meaning) - eBVM_UNCERTAIN_OCC_PROB", "static_cast<int32_t>(meaning) - 128"),
    ('X2 ray: robot cut disabled', RAY, "if (pc.cut_robot != 0u)", "if (false)"),
    ('X3 ray: robot bit 0 instead of the given bit', RAY, "(pc.robot_bit & 31u)) & 1u", "0u) & 1u"),
    ('X4 prob: free update of rays ignored', PC, "params.free_inc = options.model.free_update;", "params.free_inc = 0;"),
    ('X5 prob: hit update ignored', PC, "params.hit_inc = options.model.occupied_update;", "params.hit_inc = 0;"),
    ('X6 dist: merge level always minimal', DC, "ProbVoxelMap::probability_from_float(occupancy_threshold) };", "ProbVoxelMap::probability_from_float(0.0f) };"),
    ('X7 dist: merge level always maximal', DC, "ProbVoxelMap::probability_from_float(occupancy_threshold) };", "ProbVoxelMap::probability_from_float(1.0f) };"),
]

TESTS = 'ExampleTest.ProbVoxels:ExampleTest.ProbMap2DistMap:ExampleTest.RaycastingSensorExample'
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
            failed = [l for l in r.stdout.splitlines() if l.startswith('[  FAILED  ] ExampleTest.')]
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED by ' + ', '.join(sorted(set(f.split()[3].split(',')[0] for f in failed)))
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
