# Hand-written mutation runner (slice S6d-7, exact propagate path with exactly sized work lists): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6d7_exact_lists.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
DOWN = V + 'shaders/octree_propagate_down.glsl'
CPP = V + 'src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')

MUT = [
    ('E1 count pass counts nothing', DOWN, "        if (enters_next_list(child, subtree))\n            atomicAdd(counter[1], 1u);", "        if (false)\n            atomicAdd(counter[1], 1u);"),
    ('E2 count pass counts every child (equivalent: the list is only larger)', DOWN, "        if (enters_next_list(child, subtree))\n            atomicAdd(counter[1], 1u);", "        if (true)\n            atomicAdd(counter[1], 1u);"),
    ('E3 count pass does not return and changes the tree', DOWN, "            atomicAdd(counter[1], 1u);\n        return;\n", "            atomicAdd(counter[1], 1u);\n"),
    ('E4 count pass counts into the word of the real pass', DOWN, "atomicAdd(counter[1], 1u);", "atomicAdd(counter[0], 1u);"),
    ('E5 count pass skips the condition on PART', DOWN, "return (child.status & OCT_PART) != 0u && ((child.flags", "return true && ((child.flags"),
    ('H1 path choice inverted (equivalent: both paths give the same tree)', CPP, "if (worst * 16 > exact_list_threshold)", "if (worst * 16 <= exact_list_threshold)"),
    ('H2 count not used for the size (equivalent: lists of 8 entries per node)', CPP, "capacity = read_word(counter, 1);", "capacity = worst;"),
    ('H3 count read from the wrong word', CPP, "capacity = read_word(counter, 1);", "capacity = read_word(counter, 0);"),
    ('H4 overflow of the exact list not detected (equivalent while the count pass is right)', CPP, "				if (count[level - 1] > capacity)\n					throw", "				if (false)\n					throw"),
    ('H5 write pass without the capacity guard (equivalent while the count pass is right)', CPP, "PropagatePush{ count[level], uint32_t(level), 0, uint32_t(capacity), UINT32_MAX });", "PropagatePush{ count[level], uint32_t(level), 0, UINT32_MAX, UINT32_MAX });"),
    ('H6 count pass over too few nodes', CPP, "dispatch(batch, kernels.prop_down, { &*list[level], &no_list, &counter }, worst, PropagatePush{ count[level], uint32_t(level), 0, 0, UINT32_MAX, 1 });", "dispatch(batch, kernels.prop_down, { &*list[level], &no_list, &counter }, count[level], PropagatePush{ count[level], uint32_t(level), 0, 0, UINT32_MAX, 1 });"),
    ('H7 count pass is a real pass', CPP, "PropagatePush{ count[level], uint32_t(level), 0, 0, UINT32_MAX, 1 });", "PropagatePush{ count[level], uint32_t(level), 0, 0, UINT32_MAX, 0 });"),
    ('H8 list sized for the count minus one', CPP, "DeviceBuffer next(context, std::max<uint64_t>(16, capacity * 16));", "DeviceBuffer next(context, std::max<uint64_t>(16, capacity > 1 ? (capacity - 1) * 16 : 16));"),
    ('H9 counter of the real pass not cleared', CPP, "				DeviceBuffer next(context, std::max<uint64_t>(16, capacity * 16));\n				DeviceBuffer counter = zero_counter();", "				DeviceBuffer next(context, std::max<uint64_t>(16, capacity * 16));\n				DeviceBuffer counter(context, 16);"),
]

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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast',
                                '--gtest_filter=VulkanTest.OctreeExact*:VulkanTest.OctreePropagate*:VulkanTest.OctreeInsert*:VulkanTest.OctreeAfterInserts*:VulkanTest.OctreeSoft*:VulkanTest.OctreeSubtree*'],
                               env=test_env, capture_output=True, text=True, timeout=1500)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
