# Hand-written mutation runner (slice S6d): one mutation at a time, build, run the tests, restore. Usage: python mutate_s6d2_insert.py [name prefix]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk or build/verify-cpu.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
S = root + 'packages/gpu_voxels/gpu_voxels/vulkan/shaders/'
CPP = root + 'packages/gpu_voxels/gpu_voxels/vulkan/src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

MUT = [
    ('W1 walk does not flag', S + 'octree_insert_walk.comp', "        atomicOr(b.w[2u * slot], OCT_NEEDS_UPDATE << 8u);\n", ""),
    ('W2 leaf exists without PART test', S + 'octree_insert_walk.comp', "level == 1u && (node.status & OCT_PART) != 0u && node.child != 0ul;", "level == 1u && node.child != 0ul;"),
    ('W3 walk descends into level 1', S + 'octree_insert_walk.comp', "if (level > 1u && (node.status", "if (level > 2u && (node.status"),
    ('W4 replaced not counted', S + 'octree_insert_walk.comp', "    if (!leaf_exists && node.child != 0ul)\n        atomicAdd(replaced[0], 1u);\n", ""),
    ('H1 creates one level too many', S + 'octree_insert_heads.comp', "f.w == 1u && f.x > pc.level;", "f.w == 1u && f.x >= pc.level;"),
    ('H2 head test ignores the previous voxel creating', S + 'octree_insert_heads.comp', "(i == 0u || !creates(i - 1u) || ", "(i == 0u || "),
    ('H3 wrong shift', S + 'octree_insert_heads.comp', "const uint shift = 3u * (pc.level + 1u);", "const uint shift = 3u * pc.level;"),
    ('L1 no LAST_LEVEL on level 1', S + 'octree_insert_link.comp', "OCT_DYNAMIC | (level == 1u ? OCT_LAST_LEVEL : 0u);", "OCT_DYNAMIC;"),
    ('L2 new nodes not flagged', S + 'octree_insert_link.comp', "words.w[2u * s] = status | (OCT_NEEDS_UPDATE << 8u);", "words.w[2u * s] = status;"),
    ('L3 invalid leaves without the invalid bits', S + 'octree_insert_link.comp', "words.w[0] = 0x87878787u;", "words.w[0] = 0x80808080u;"),
    ('L4 wrong parent selection', S + 'octree_insert_link.comp', "if (f.x == level + 1u)", "if (f.x == level)"),
    ('L5 parent without PART', S + 'octree_insert_link.comp', "p.w0 = (w0 & 0xFFFFu) | OCT_PART | (uint(block", "p.w0 = (w0 & 0xFFFFu) | (uint(block"),
    ('L6 child pointer shift', S + 'octree_insert_link.comp', "p.w1 = uint(block >> 16);", "p.w1 = uint(block >> 17);"),
    ('L7 parent index off by one', S + 'octree_insert_link.comp', "excl_up[i] + heads_up[i] - 1u;", "excl_up[i] + heads_up[i];"),
    ('L8 parent slot', S + 'octree_insert_link.comp', "(uint(id >> (3u * (level + 1u))) & 7u) * 8u", "(uint(id >> (3u * level)) & 7u) * 8u"),
    ('F1 new leaf without DYNAMIC', S + 'octree_insert_leaves.comp', "(OCT_DYNAMIC | pc.datum) << shift", "pc.datum << shift"),
    ('F2 existing leaf mask without PART (equivalent: leaves have no PART)', S + 'octree_insert_leaves.comp', "~((OCT_OCCUPANCY_MASK | OCT_PART) << shift)", "~(OCT_OCCUPANCY_MASK << shift)"),
    ('F3 new leaf block index', S + 'octree_insert_leaves.comp', "excl0[i] + heads0[i] - 1u", "excl0[i]"),
    ('F4 existing leaf keeps the old occupancy', S + 'octree_insert_leaves.comp', "atomicAnd(words.w[slot >> 2u], ~((OCT_OCCUPANCY_MASK | OCT_PART) << shift));", ""),
    ('D1 expansion condition inverted', S + 'octree_propagate_down.comp', "if ((status & OCT_OCCUPANCY_MASK) == 0u)", "if ((status & OCT_OCCUPANCY_MASK) != 0u)"),
    ('D2 expansion copies PART', S + 'octree_propagate_down.comp', "(parent_status & ~(OCT_PART | OCT_LAST_LEVEL) & 255u)", "(parent_status & ~OCT_LAST_LEVEL & 255u)"),
    ('D3 flag of non-PART child kept', S + 'octree_propagate_down.comp', "        flags &= ~OCT_NEEDS_UPDATE;\n", "        ;\n"),
    ('D4 level 2 children only when flagged', S + 'octree_propagate_down.comp', "|| pc.level == 2u))", "|| false))"),
    ('D5 children pushed although not flagged', S + 'octree_propagate_down.comp', "((flags & OCT_NEEDS_UPDATE) != 0u || pc.level == 2u)", "true"),
    ('P1 invalid leaf test', S + 'octree_propagate_leaves.comp', "== OCT_OCCUPANCY_MASK)", "== 0u)"),
    ('P2 level 1 node without LAST_LEVEL', S + 'octree_propagate_leaves.comp', "oct_parent(kids, 1u, 0ul)", "oct_parent(kids, 99u, 0ul)"),
    ('P3 flag of the level 1 node kept', S + 'octree_propagate_leaves.comp', "& ~OCT_NEEDS_UPDATE;\n    node.w0", "& 255u;\n    node.w0"),
    ('U1 LAST_LEVEL above level 1', S + 'octree_propagate_up.comp', "oct_parent(kids, 99u, 0ul)", "oct_parent(kids, 1u, 0ul)"),
    ('U2 flag kept after the bottom-up step', S + 'octree_propagate_up.comp', "& ~OCT_NEEDS_UPDATE;\n    node.w0", "& 255u;\n    node.w0"),
    ('C1 total of a level without the last head', CPP, "totals[level] = uint64_t(read_word(excl_a, n - 1)) + read_word(heads_a, n - 1);", "totals[level] = uint64_t(read_word(excl_a, n - 1));"),
    ('C2 top-down stops one level early', CPP, "for (int level = int(kRootLevel); level >= 2 && count[level] > 0; --level)", "for (int level = int(kRootLevel); level >= 3 && count[level] > 0; --level)"),
    ('C3 dropped not counted', CPP, "			dropped += ids.size() - valid.size();\n", ""),
    ('C4 duplicates kept', CPP, "			valid.erase(std::unique(valid.begin(), valid.end()), valid.end());\n			if (valid.size() > kMaxVoxels)\n				throw VulkanError(\"Octree::insert: too many", "			if (valid.size() > kMaxVoxels)\n				throw VulkanError(\"Octree::insert: too many"),
    ('C5 probabilistic insert allowed', CPP, "			if (kind != OctreeKind::Deterministic)\n				throw VulkanError(\"Octree::insert: only", "			if (false)\n				throw VulkanError(\"Octree::insert: only"),
    ('C6 garbage flag never set', CPP, "			if (read_word(replaced, 0) > 0)\n				may_have_garbage = true;\n", ""),
    ('C7 unreachable blocks ignored', CPP, "				if (unreachable != 0 && !may_have_garbage)", "				if (false)"),
    ('C8 stale child status rule applied to every node', CPP, "if ((node.status & kPart) && (expected.status", "if ((expected.status"),
]

only = sys.argv[1] if len(sys.argv) > 1 else ''
done = set(sys.argv[2].split(',')) if len(sys.argv) > 2 else set()
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
                                '--gtest_filter=VulkanTest.OctreeInsert*:VulkanTest.OctreeAfterInserts*:VulkanTest.CheckTree*'],
                               env=env, capture_output=True, text=True, timeout=150)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
