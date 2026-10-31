# Hand-written mutation runner (slice S6d-4): one mutation at a time, build, run the tests, restore. Usage: python mutate_s6d4_rebuild.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
S = root + 'packages/gpu_voxels/gpu_voxels/vulkan/shaders/'
CPP = root + 'packages/gpu_voxels/gpu_voxels/vulkan/src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# The tests run with GPU assisted validation in safe mode: a mutant that follows a wrong device address is stopped by the instrumented shader
# and reported as a validation error, it never reaches the GPU as a page fault (nvlddmkm event 153, which stresses the driver).
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')
SCAN = S + 'octree_compact_scan.glsl'
COPY = S + 'octree_compact_copy.glsl'

MUT = [
    ('S1 scan follows stale child pointers of nodes without PART', SCAN, "if ((node.status & OCT_PART) != 0u)\n    {\n        index = atomicAdd", "if (node.child != 0ul)\n    {\n        index = atomicAdd"),
    ('S2 scan writes past the capacity (not observable: the guard only protects memory)', SCAN, "        if (index < pc.capacity)\n", "        if (true)\n"),
    ('S3 link always 0', SCAN, "    link[g] = index;\n", "    link[g] = 0u;\n"),
    ('S4 high word of the block address lost', SCAN, "next[2u * index + 1u] = uint(node.child >> 32);", "next[2u * index + 1u] = 0u;"),
    ('S5 root address high word lost', CPP, "uint32_t(root), uint32_t(root >> 32) }", "uint32_t(root), 0u }"),
    ('CP1 stale child pointer kept on nodes without PART', COPY, "    else\n        node.child = 0ul;\n", ""),
    ('CP2 wrong child block size', COPY, "uint64_t(link[g]) * uint64_t(pc.child_block_bytes)", "uint64_t(link[g]) * uint64_t(OCT_INNER_BLOCK_BYTES)"),
    ('CP3 leaf block copies one word only', COPY, "w < OCT_LEAF_BLOCK_BYTES / 4u", "w < 1u"),
    ('CP4 every node lands in slot 0', COPY, "oct_store_inner(to, s, node);", "oct_store_inner(to, 0u, node);"),
    ('CP5 collision bit dropped', COPY, "    OctNode node = oct_load_inner(from, s);\n", "    OctNode node = oct_load_inner(from, s);\n    node.status &= ~OCT_COLLISION;\n"),
    ('CP6 occupancy dropped', COPY, "    OctNode node = oct_load_inner(from, s);\n", "    OctNode node = oct_load_inner(from, s);\n    node.occupancy = 0;\n"),
    ('CP7 child index off by one', COPY, "uint64_t(link[g]) * uint64_t(pc.child_block_bytes)", "uint64_t(link[g] + 1u) * uint64_t(pc.child_block_bytes)"),
    ('CP8 all leaf blocks land in block 0', COPY, "pc.base + uint64_t(g) * uint64_t(OCT_LEAF_BLOCK_BYTES)", "pc.base"),
    ('CP9 flags dropped (equivalent after propagate: live nodes carry none)', COPY, "    OctNode node = oct_load_inner(from, s);\n", "    OctNode node = oct_load_inner(from, s);\n    node.flags = 0u;\n"),
    ('H1 capacity not bounded by the allocated blocks', CPP, "std::min<uint64_t>(count[level] * 8, levels[level - 1].blocks())", "count[level] * 8"),
    ('H2 pending updates not propagated', CPP, "		void rebuild()\n		{\n			propagate();\n", "		void rebuild()\n		{\n"),
    ('H3 old chunks of the levels kept', CPP, "				levels[level].chunks.clear();\n				if (fresh[level])", "				if (fresh[level])"),
    ('H4 root level not rewritten', CPP, "for (uint32_t level = 0; level <= kRootLevel; ++level)\n				if (count[level] > 0)\n				{\n					const uint32_t child_bytes", "for (uint32_t level = 0; level < kRootLevel; ++level)\n				if (count[level] > 0)\n				{\n					const uint32_t child_bytes"),
    ('H5 child base of the wrong level', CPP, "level > 0 ? base[level - 1] : 0, child_bytes", "level > 0 ? base[level] : 0, child_bytes"),
    ('H6 garbage flag kept (equivalent: the walk then counts 0 garbage)', CPP, "			}\n			may_have_garbage = false;\n		}\n\n		// ------------------------------------------------------------------------------------------------ inspection", "			}\n		}\n\n		// ------------------------------------------------------------------------------------------------ inspection"),
    ('H7 address range of the new chunks not checked (the device never hands out such an address)', CPP, "					check_chunk(*fresh[level]);\n", ""),
    ('H8 duplicate blocks not detected', CPP, "				if (found > capacity)\n", "				if (false)\n"),
    ('N1 limit reached only above the memory in use', CPP, "bytes >= policy.max_bytes", "bytes > policy.max_bytes"),
    ('N2 limit 0 means always', CPP, "policy.max_bytes != 0 && bytes >= policy.max_bytes", "bytes >= policy.max_bytes"),
    ('N3 garbage bytes with one block size', CPP, "stats.garbage_blocks[level] * octree_format::block_bytes(impl_->kind, level);", "stats.garbage_blocks[level] * 64;"),
    ('N4 garbage fraction evaluated without garbage (equivalent: the walk counts 0)', CPP, "policy.max_garbage_fraction > 0.0 && impl_->may_have_garbage", "policy.max_garbage_fraction > 0.0"),
    ('N5 fraction compared with the garbage count not bytes', CPP, "return stats.bytes > 0 && double(garbage) >= policy.max_garbage_fraction * double(stats.bytes);", "return stats.bytes > 0 && double(garbage) >= policy.max_garbage_fraction;"),
    ('P1 fraction above 1 accepted', CPP, "policy.max_garbage_fraction <= 1.0", "policy.max_garbage_fraction <= 2.0"),
    ('P2 NaN accepted', CPP, "if (!(policy.max_garbage_fraction >= 0.0 && policy.max_garbage_fraction <= 1.0))", "if (policy.max_garbage_fraction < 0.0 || policy.max_garbage_fraction > 1.0)"),
    ('P3 policy not stored', CPP, "		impl_->policy = policy;\n", ""),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe',
                                '--gtest_filter=VulkanTest.OctreeRebuild*:VulkanTest.OctreeNeedsRebuild*:VulkanTest.OctreeInsertsAfterARebuild*'],
                               env=test_env, capture_output=True, text=True, timeout=900)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
