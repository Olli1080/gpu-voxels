# Hand-written mutation runner (slice S6d-6, garbage counted on the GPU): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6d6_garbage.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
# Not run on purpose: removing the clamp `min(counter[level], item_capacity)` of the kernel (in a corrupt tree the kernel would read behind the list
# and dereference address 0, which can fault the GPU; GPU assisted validation does not check buffer references).
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
SCAN = V + 'shaders/octree_garbage_scan.glsl'
CPP = V + 'src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')

MUT = [
    ('GS1 only nodes with PART count their child block', SCAN, "if (node.child != 0ul)", "if (node.child != 0ul && (node.status & OCT_PART) != 0u)"),
    ('GS2 blocks are counted on their own level', SCAN, "atomicAdd(counter[pc.level - 1u], 1u)", "atomicAdd(counter[pc.level], 1u)"),
    ('GS3 next list written past its capacity', SCAN, "if (index < pc.next_capacity)", "if (true)"),
    ('GS4 child address of the high word only', SCAN, "next[2u * index] = uint(node.child & 0xFFFFFFFFul);", "next[2u * index] = 0u;"),
    ('GS5 one slot per block only', SCAN, "const uint s = g & 7u;", "const uint s = 0u;"),
    ('GH1 capacity not bounded by the allocated blocks', CPP, "capacity[level - 1] = std::min<uint64_t>(capacity[level] * 8, levels[level - 1].blocks());", "capacity[level - 1] = capacity[level] * 8;"),
    ('GH2 more blocks than allocated not detected', CPP, "				if (found[level] > capacity[level])\n					return false;\n", ""),
    ('GH3 root block not counted', CPP, "			start[kRootLevel] = 1;\n", ""),
    ('GH4 no barrier between the levels', CPP, "				if (step > 0)\n					batch.barrier();\n", ""),
    ('GH5 lists of the levels read from the wrong buffer', CPP, "step == 0 ? &root_list : (step % 2 == 1 ? &list_a : &list_b)", "step == 0 ? &root_list : (step % 2 == 1 ? &list_b : &list_a)"),
    ('GH6 garbage = reachable blocks', CPP, "garbage[level] = levels[level].blocks() - found[level];", "garbage[level] = found[level];"),
    ('GH7 no fallback for a corrupt tree', CPP, "if (impl_->host_garbage_count || !impl_->count_garbage_on_gpu(unreachable))", "if (impl_->host_garbage_count)"),
    ('GH8 level 1 writes a list of the leaf blocks (equivalent: only counted)', CPP, "level > 1 ? uint32_t(capacity[level - 1]) : 0u", "uint32_t(capacity[level - 1])"),
    ('GH9 counter buffer not cleared (equivalent if a fresh buffer is zero: it is uploaded)', CPP, "			counter.upload<uint32_t>(start);\n", ""),
    ('GH10 only the first 13 levels dispatched', CPP, "for (uint32_t step = 0; step < kRootLevel; ++step)", "for (uint32_t step = 0; step + 1 < kRootLevel; ++step)"),
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
                                '--gtest_filter=VulkanTest.OctreeGarbage*:VulkanTest.OctreeNeedsRebuild*:VulkanTest.OctreeInsert*:VulkanTest.OctreeAfterInserts*:VulkanTest.OctreeRebuild*'],
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
