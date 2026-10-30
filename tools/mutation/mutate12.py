"""Mutation check for slice S6c part 1 (collisions with maps / lists / ids; derived from the S6b runner). Original docstring: S6b (64-bit sort, octree node formats, build/find/extract kernels, Octree host code).
kind 'gpu'  : test_vulkan with the octree filter (fail fast)
kind 'sort' : test_vulkan with the 64-bit sort filter
kind 'fmt'  : test_octree_reference (OctreeFormat tests) first, test_vulkan octree filter when that survives
kind 'cpu'  : test_octree_reference only"""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
V = r'packages\gpu_voxels\gpu_voxels\vulkan'
SH = V + r'\shaders'
NODES = SH + r'\octree_nodes.glsl'
BUILD = SH + r'\octree_build.glsl'
FIND = SH + r'\octree_find.glsl'
EXTRACT = SH + r'\octree_extract.glsl'
HEADS = SH + r'\octree_heads.comp'
COMPACT = SH + r'\octree_compact_ids.comp'
MORTON_IDS = SH + r'\octree_morton_ids.comp'
GID = SH + r'\octree_gid.glsl'
COLLIDE = SH + r'\octree_collide.glsl'
CLEAR = SH + r'\octree_clear_collision.comp'
TREES = SH + r'\octree_collide_trees.glsl'
HIST64 = SH + r'\radix_histogram64.comp'
SCATTER64 = SH + r'\radix_scatter64.comp'
CPP = V + r'\src\Octree.cpp'
PRIM = V + r'\src\Primitives.cpp'
FORMAT = V + r'\include\gpu_voxels\vulkan\OctreeFormat.h'
REF = r'tests\reference\octree_reference.h'
CPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\Debug\test_octree_reference.exe')
GPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
OCTREE_FILTER = '--gtest_filter=*OctreeTreeCollision*:-*LargeTrees*'
SORT_FILTER = '--gtest_filter=*SortPairs64*'
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

mutations = [
    # ---- tree vs tree kernel
    ('GT1 item index', 'gpu', TREES, 'const uint item = t >> 3u;', 'const uint item = t >> 2u;'),
    ('GT2 b address words', 'gpu', TREES, 'packUint2x32(uvec2(items[6u * item + 2u], items[6u * item + 3u]))', 'packUint2x32(uvec2(items[6u * item + 3u], items[6u * item + 2u]))'),
    ('GT3 flags word', 'gpu', TREES, 'const uint flags = items[6u * item + 4u];', 'const uint flags = items[6u * item + 5u];'),
    ('GT4 a active bit', 'gpu', TREES, 'const bool a_active = (flags & 1u) != 0u;', 'const bool a_active = (flags & 2u) != 0u;'),
    ('GT5 lane address', 'gpu', TREES, 'a_base + uint64_t(lane * OCT_NODE_BYTES)', 'a_base + uint64_t((lane + 1u) * OCT_NODE_BYTES)'),
    ('GT6 collide needs both', 'gpu', TREES, 'const bool collide = node_occupied(a) && node_occupied(b);', 'const bool collide = node_occupied(a) || node_occupied(b);'),
    ('GT7 end here', 'gpu', TREES, 'const bool end_here = pc.min_level >= pc.level;', 'const bool end_here = pc.min_level > pc.level;'),
    ('GT8 last level at the end', 'gpu', TREES, '!= 0u) && !end_here;', '!= 0u);'),
    ('GT9 count condition', 'gpu', TREES, 'if ((!a_part && !b_part) || end_here)', 'if ((!a_part && !b_part) && end_here)'),
    ('GT10 counter index', 'gpu', TREES, 'atomicAdd(counters[1u + pc.level], 1u);', 'atomicAdd(counters[pc.level], 1u);'),
    ('GT11 marked node', 'gpu', TREES, 'OctInnerBlock(a_active ? a_address : b_address).w[0]', 'OctInnerBlock(a_active ? b_address : a_address).w[0]'),
    ('GT12 mark option', 'gpu', TREES, 'if (pc.mark != 0u)\n            atomicOr(OctInnerBlock', 'if (pc.mark == 0u)\n            atomicOr(OctInnerBlock'),
    ('GT13 refine condition', 'gpu', TREES, 'if (!is_last_level && (a_part || b_part) && !end_here)', 'if (!is_last_level && (a_part && b_part) && !end_here)'),
    ('GT14 next active', 'gpu', TREES, 'const bool a_next_active = a_active && a_part;', 'const bool a_next_active = a_part;'),
    ('GT15 next address', 'gpu', TREES, 'const uint64_t a_next = a_next_active ? a.child : a_address;', 'const uint64_t a_next = a.child;'),
    ('GT16 next flags', 'gpu', TREES, '(a_next_active ? 1u : 0u) | (b_next_active ? 2u : 0u)', '(a_next_active ? 2u : 0u) | (b_next_active ? 1u : 0u)'),
    ('GT17 leaf stage level', 'gpu', TREES, 'if (pc.level == 1u && pc.min_level == 0u && (a_part || b_part))', 'if (pc.level == 2u && pc.min_level == 0u && (a_part || b_part))'),
    ('GT18 leaf stage min level', 'gpu', TREES, 'if (pc.level == 1u && pc.min_level == 0u && (a_part || b_part))', 'if (pc.level == 1u && pc.min_level <= 1u && (a_part || b_part))'),
    ('GT19 leaf stage needs parts', 'gpu', TREES, 'pc.min_level == 0u && (a_part || b_part))', 'pc.min_level == 0u && (a_part && b_part))'),
    ('GT20 both leaves', 'gpu', TREES, 'node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)) && node_occupied(oct_load_leaf(OctLeafBlock(b.child), j))', 'node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)) || node_occupied(oct_load_leaf(OctLeafBlock(b.child), j))'),
    ('GT21 a leaves against uniform b', 'gpu', TREES, 'if (node_occupied(b) && node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)))', 'if (node_occupied(a) && node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)))'),
    ('GT22 b leaves against uniform a', 'gpu', TREES, 'else if (node_occupied(a) && node_occupied(oct_load_leaf(OctLeafBlock(b.child), j)))', 'else if (node_occupied(b) && node_occupied(oct_load_leaf(OctLeafBlock(b.child), j)))'),
    ('GT23 mark b leaf', 'gpu', TREES, 'mark_leaf(b.child, j);', 'mark_leaf(a.child, j);'),
    ('GT24 det leaf mark', 'gpu', TREES, 'w[slot >> 2u], OCT_COLLISION << ((slot & 3u) * 8u)', 'w[slot >> 2u], OCT_COLLISION << ((slot & 3u) * 4u)'),
    ('GT25 prob leaf mark', 'gpu', TREES, 'w[slot >> 1u], OCT_COLLISION << ((slot & 1u) * 16u)', 'w[slot >> 1u], OCT_COLLISION << ((slot & 1u) * 8u)'),
    ('GT26 prob unknown test', 'gpu', TREES, 'n.occupancy != OCT_PROB_UNKNOWN && n.occupancy >= pc.tree_threshold', 'n.occupancy >= pc.tree_threshold'),
    ('GT27 prob threshold', 'gpu', TREES, 'n.occupancy >= pc.tree_threshold', 'n.occupancy > pc.tree_threshold'),
    ('GT28 prob node bytes', 'gpu', TREES, 'const uint OCT_NODE_BYTES = 12u;', 'const uint OCT_NODE_BYTES = 8u;'),
    ('GT29 det node bytes', 'gpu', TREES, 'const uint OCT_NODE_BYTES = 8u;', 'const uint OCT_NODE_BYTES = 12u;'),
    ('GT30 det occupied bit', 'gpu', TREES, 'return (n.status & OCT_OCCUPIED) != 0u;', 'return (n.status & OCT_UNKNOWN) != 0u;'),
    # ---- host
    ('CT1 kind check', 'gpu', CPP, 'if (other.kind != kind)', 'if (false)'),
    ('CT2 self check', 'gpu', CPP, 'if (&other == this)', 'if (false)'),
    ('CT3 uniform roots', 'gpu', CPP, 'if (!a_part && !b_part)\n			{\n				if (!(occupied(root_a)', 'if (!a_part || !b_part)\n			{\n				if (!(occupied(root_a)'),
    ('CT4 root box', 'gpu', CPP, 'for (uint32_t i = 0; i < kRootLevel; ++i)\n				box *= 8;', 'for (uint32_t i = 0; i <= kRootLevel; ++i)\n				box *= 8;'),
    ('CT5 start address of a', 'gpu', CPP, 'const uint64_t a_start = a_part ? root_a.child : root_address();', 'const uint64_t a_start = a_part ? root_a.child : other.root_address();'),
    ('CT6 start address of b', 'gpu', CPP, 'const uint64_t b_start = b_part ? root_b.child : other.root_address();', 'const uint64_t b_start = b_part ? root_b.child : root_address();'),
    ('CT7 first flags', 'gpu', CPP, '(a_part ? 1u : 0u) | (b_part ? 2u : 0u), 0u };', '(a_part ? 2u : 0u) | (b_part ? 1u : 0u), 0u };'),
    ('CT8 first level', 'gpu', CPP, 'for (int level = int(kRootLevel) - 1; level >= 1 && item_count > 0; --level)\n			{\n				const uint32_t zero = 0;\n				counters.upload(&zero, 4, 0);\n				const uint64_t capacity', 'for (int level = int(kRootLevel) - 2; level >= 1 && item_count > 0; --level)\n			{\n				const uint32_t zero = 0;\n				counters.upload(&zero, 4, 0);\n				const uint64_t capacity'),
    ('CT9 last level', 'gpu', CPP, 'level >= 1 && item_count > 0; --level)\n			{\n				const uint32_t zero = 0;\n				counters.upload(&zero, 4, 0);\n				const uint64_t capacity', 'level >= 2 && item_count > 0; --level)\n			{\n				const uint32_t zero = 0;\n				counters.upload(&zero, 4, 0);\n				const uint64_t capacity'),
    ('CT10 level weight', 'gpu', CPP, 'level < kLevelCount; ++level, weight *= 8)', 'level < kLevelCount; ++level, weight *= 4)'),
    ('CT11 level counters', 'gpu', CPP, 'volume += uint64_t(counts[1 + level]) * weight;', 'volume += uint64_t(counts[level]) * weight;'),
    ('CT12 min level', 'gpu', CPP, 'TreePush{ item_count, uint32_t(level), options.min_level, options.mark_collisions ? 1u : 0u,', 'TreePush{ item_count, uint32_t(level), 0u, options.mark_collisions ? 1u : 0u,'),
    ('CT13 mark option', 'gpu', CPP, 'options.min_level, options.mark_collisions ? 1u : 0u,', 'options.min_level, 1u,'),
    ('CT14 threshold', 'gpu', CPP, 'options.mark_collisions ? 1u : 0u, options.occupancy_threshold, uint32_t(capacity) });', 'options.mark_collisions ? 1u : 0u, 10, uint32_t(capacity) });'),
    ('CT15 det occupied', 'gpu', CPP, ': (n.status & kOccupied) != 0;\n			};\n			const HostNode root_a', ': (n.status & kUnknown) != 0;\n			};\n			const HostNode root_a'),
    ('CT16 root mark', 'gpu', CPP, 'marked.status |= kCollision;', 'marked.status |= kStatic;'),
    ('CT17 prob occupied', 'gpu', CPP, '(n.occupancy != kProbUnknown && n.occupancy >= options.occupancy_threshold)', '(n.occupancy >= options.occupancy_threshold)'),
    ('CT18 root with child', 'gpu', CPP, 'const bool a_part = (root_a.status & kPart) && root_a.child != 0, b_part', 'const bool a_part = (root_a.status & kPart), b_part'),
    # ---- reference
    ('RT1 count condition', 'gpu', REF, 'if ((!a_part && !b_part) || end_here)\n					{\n						uint64_t box = 1;', 'if ((!a_part && !b_part) && end_here)\n					{\n						uint64_t box = 1;'),
    ('RT2 mark side', 'gpu', REF, '(a_active ? r.flagged_this : r.flagged_other).insert(a_active ?', '(a_active ? r.flagged_other : r.flagged_this).insert(a_active ?'),
    ('RT3 refine', 'gpu', REF, 'if (!is_last && (a_part || b_part) && !end_here)', 'if (!is_last && (a_part && b_part) && !end_here)'),
    ('RT4 stand in', 'gpu', REF, 'const bool a_next = a_active && a_part, b_next = b_active && b_part;', 'const bool a_next = a_part, b_next = b_part;'),
    ('RT5 leaves', 'gpu', REF, 'if (level == 1 && min_level == 0 && (a_part || b_part))', 'if (level == 1 && min_level == 0 && (a_part && b_part))'),
    ('RT6 root special case', 'gpu', REF, 'if (!a_part && !b_part)\n			{\n				if (rec.occupied(*this, ra)', 'if (!a_part || !b_part)\n			{\n				if (rec.occupied(*this, ra)'),
]

only = os.environ.get('ONLY')
done = set()
if os.environ.get('DONE_LOG'):
    for line in open(os.environ['DONE_LOG'], encoding='utf-8'):
        if ': KILLED' in line:
            done.add(line.split(': KILLED')[0])


def read(path):
    return open(path, 'rb').read()


def build(target):
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', target],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    failed = re.findall(r'^\[  FAILED  \] (\S+)', r.stdout, re.M)
    return failed, r.returncode


def evaluate(kind):
    """returns (status, detail): status in KILLED / SURVIVED / BUILD"""
    if kind in ('fmt', 'cpu'):
        if not build('test_octree_reference'):
            return 'BUILD', ''
        failed, code = run([CPU_EXE, '--gtest_fail_fast'])
        if failed or code != 0:
            return 'KILLED', 'cpu: ' + ', '.join(sorted(set(failed))[:2])
        if kind == 'cpu':
            return 'SURVIVED', ''
    if not build('test_vulkan'):
        return 'BUILD', ''
    flt = SORT_FILTER if kind == 'sort' else OCTREE_FILTER
    failed, code = run([GPU_EXE, flt, '--gtest_fail_fast'])
    if failed or code != 0:
        return 'KILLED', ', '.join(sorted(set(failed))[:2]) + (f' (exit {code})' if not failed else '')
    return 'SURVIVED', ''


survivors = []
touched = {}
try:
    for name, kind, rel, old, new in mutations:
        if only and not name.startswith(only):
            continue
        if name in done:
            continue
        path = os.path.join(ROOT, rel)
        original = touched.setdefault(path, read(path))
        text = original.decode('utf-8')
        crlf = '\r\n' in text
        o = old.replace('\n', '\r\n') if crlf else old
        n = new.replace('\n', '\r\n') if crlf else new
        if text.count(o) != 1:
            print(f'{name}: PATTERN {"NOT FOUND" if text.count(o) == 0 else "AMBIGUOUS"} in {rel}', flush=True)
            survivors.append(name + ' (pattern)')
            continue
        open(path, 'wb').write(text.replace(o, n, 1).encode('utf-8'))
        status, detail = evaluate(kind)
        if status == 'KILLED':
            print(f'{name}: KILLED by {detail}', flush=True)
        elif status == 'BUILD':
            print(f'{name}: BUILD FAILED', flush=True)
            survivors.append(name + ' (build)')
        else:
            print(f'{name}: SURVIVED  <-- weak tests', flush=True)
            survivors.append(name)
        open(path, 'wb').write(original)
finally:
    for path, data in touched.items():
        open(path, 'wb').write(data)
    ok1 = build('test_octree_reference')
    ok2 = build('test_vulkan')
    f1, c1 = run([CPU_EXE])
    f2, c2 = run([GPU_EXE, OCTREE_FILTER])
    f3, c3 = run([GPU_EXE, SORT_FILTER])
    print('restored; build ok =', ok1, ok2, '; failing at baseline:', f1, f2, f3, 'exit', c1, c2, c3, flush=True)
print('SURVIVORS:', survivors, flush=True)
