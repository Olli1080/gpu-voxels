# Hand-written mutation runner (slice S6d-5, fast insert): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6d5_fast_insert.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
S = root + 'packages/gpu_voxels/gpu_voxels/vulkan/shaders/'
CPP = root + 'packages/gpu_voxels/gpu_voxels/vulkan/src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')
COUNT = S + 'octree_insert_count.comp'
DOWN = S + 'octree_propagate_down.glsl'
LEAVES = S + 'octree_propagate_leaves.glsl'
UP = S + 'octree_propagate_up.glsl'

MUT = [
    ('K1 count starts below the target level', COUNT, "for (uint level = pc.target; level < f.x && level < 14u; ++level)", "for (uint level = 0u; level < f.x && level < 14u; ++level)"),
    ('K2 count one level too many', COUNT, "level < f.x && level < 14u", "level <= f.x && level < 14u"),
    ('K3 wrong shift of the head test', COUNT, "const uint shift = 3u * (level + 1u);", "const uint shift = 3u * level;"),
    ('K4 kind 2 voxels are counted (equivalent: their loop is empty)', COUNT, "        if (f.w == 1u)\n        {\n            const uint64_t id = id_at(i);", "        if (f.w != 0u)\n        {\n            const uint64_t id = id_at(i);"),
    ('K4b kind 0 voxels (the leaf exists) are counted', COUNT, "        if (f.w == 1u)\n        {\n            const uint64_t id = id_at(i);", "        if (f.w <= 1u)\n        {\n            const uint64_t id = id_at(i);"),
    ('K5 previous voxel creating not tested (equivalent: implied by the equal parent id)', COUNT, "i == 0u || !creates(i - 1u, level) || ", "i == 0u || "),
    ('K6 group sum added only when not zero (equivalent)', COUNT, " && group_count[gl_LocalInvocationIndex] != 0u", ""),
    ('K7 group sum added to the next level', COUNT, "atomicAdd(stats[gl_LocalInvocationIndex], group_count[gl_LocalInvocationIndex]);", "atomicAdd(stats[gl_LocalInvocationIndex + 1u], group_count[gl_LocalInvocationIndex]);"),
    ('K8 garbage counter shares the word of level 13', S + 'octree_insert_walk.glsl', "atomicAdd(stats[14], 1u);", "atomicAdd(stats[13], 1u);"),
    ('H1 statistics not cleared (the fresh buffer happens to be zero)', CPP, "				batch.fill(stats, 0);\n", ""),
    ('H2 no barrier between walk and count', CPP, "				batch.barrier();\n				dispatch(batch, kernels.insert_count,", "				dispatch(batch, kernels.insert_count,"),
    ('H3 no barrier after the link kernel', CPP, "				batch.barrier();\n				std::swap(heads_cur, heads_up);", "				std::swap(heads_cur, heads_up);"),
    ('H4 levels without new blocks are not skipped (equivalent: no head voxel)', CPP, "				if (totals[level] == 0)\n					continue;\n				dispatch(batch, kernels.insert_heads", "				dispatch(batch, kernels.insert_heads"),
    ('S1 sort key of 20 bits', CPP, "primitives.record_sort_pairs64(batch, sorted, values, n, 42, keep_alive);", "primitives.record_sort_pairs64(batch, sorted, values, n, 20, keep_alive);"),
    ('S2 no sort at all', CPP, "				if (n > 1)\n				{\n					primitives.record_sort_pairs64(batch, sorted, values, n, 42, keep_alive);", "				if (false)\n				{\n					primitives.record_sort_pairs64(batch, sorted, values, n, 42, keep_alive);"),
    ('S3 distinct ids by node of level 1 not by id', CPP, "dispatch(batch, kernels.heads, { &sorted, &heads }, n, HeadsPush{ n, 0 });", "dispatch(batch, kernels.heads, { &sorted, &heads }, n, HeadsPush{ n, 3 });"),
    ('S4 distinct count without the last head', CPP, "count = read_word(offsets, n - 1) + read_word(heads, n - 1);\n			return unique_ids;", "count = read_word(offsets, n - 1);\n			return unique_ids;"),
    ('S5 paths swapped (equivalent: both give the same tree)', CPP, "if (valid.size() <= gpu_sort_limit)", "if (valid.size() > gpu_sort_limit)"),
    ('P1 level 1 list bounded by the inserts not by 8 times', CPP, "capacity[1] = std::min<uint64_t>(8 * pending_bound,", "capacity[1] = std::min<uint64_t>(pending_bound,"),
    ('P2 soft inserts do not make the bound unknown', CPP, "pending_bounded = pending_bounded && bounded;", "pending_bounded = true;"),
    ('P3 bound counts batches not ids', CPP, "			pending_bound += ids;\n", "			pending_bound += 1;\n"),
    ('P4 bound not forgotten after propagate (equivalent: still an upper bound)', CPP, "				propagate_exact();\n			forget_pending();\n		}", "				propagate_exact();\n		}"),
    ('P5 too few threads per level', CPP, "capacity[level] * 8, PropagatePush{ 0,", "capacity[level], PropagatePush{ 0,"),
    ('P6 leaves also without a top-down chain to level 2 (equivalent: the count is 0)', CPP, "if (lowest == 2 && capacity[1] > 0)", "if (capacity[1] > 0)"),
    ('P7 bound violation not detected', CPP, "				if (reached[level] > capacity[level])\n", "				if (false)\n"),
    ('P8 bound halved', CPP, "capacity[level] = std::min<uint64_t>(pending_bound, 8 * levels[level].blocks());", "capacity[level] = std::min<uint64_t>(pending_bound / 2, 8 * levels[level].blocks());"),
    ('Q1 down kernel does not clamp the item count', DOWN, "min(counter[pc.level], pc.item_capacity)", "counter[pc.level]"),
    ('Q2 down kernel writes past the next list', DOWN, "        if (index < pc.next_capacity)\n", "        if (true)\n"),
    ('Q3 down kernel counts into its own level', DOWN, "counter[pc.from_counts != 0u ? pc.level - 1u : 0u]", "counter[pc.from_counts != 0u ? pc.level : 0u]"),
    ('Q4 down kernel takes the count from the wrong source', DOWN, "pc.from_counts != 0u ? min(", "pc.from_counts == 0u ? min("),
    ('Q5 leaves kernel reads the count of level 2', LEAVES, "counts[1], pc.item_capacity", "counts[2], pc.item_capacity"),
    ('Q6 leaves kernel does not clamp the item count (equivalent: the clamped top-down kernel of level 2 writes at most 8 times its capacity)', LEAVES, "min(counts[1], pc.item_capacity)", "counts[1]"),
    ('Q7 up kernel reads the count of the level below', UP, "counts[pc.level], pc.item_capacity", "counts[pc.level - 1u], pc.item_capacity"),
    ('Q8 up kernel does not clamp the item count', UP, "min(counts[pc.level], pc.item_capacity)", "counts[pc.level]"),
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
                                '--gtest_filter=VulkanTest.OctreeInsert*:VulkanTest.OctreeAfterInserts*:VulkanTest.Probabilistic*:VulkanTest.OctreeSoft*:VulkanTest.OctreeSubtree*:VulkanTest.CheckTree*:VulkanTest.OctreePropagate*:VulkanTest.OctreeRebuild*'],
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
