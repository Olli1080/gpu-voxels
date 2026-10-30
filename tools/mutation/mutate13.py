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
    ('H1 refine condition', 'gpu', TREES, 'if (!is_last_level && (a_part || b_part) && !end_here)', 'if (!is_last_level && (a_part && b_part) && !end_here)'),
    ('H2 next address of a', 'gpu', TREES, 'const uint64_t a_next = a_part ? a.child : a_address;', 'const uint64_t a_next = a.child;'),
    ('H3 next address of b', 'gpu', TREES, 'const uint64_t b_next = b_part ? b.child : b_address;', 'const uint64_t b_next = b.child;'),
    ('H4 next flags', 'gpu', TREES, '(a_part ? 1u : 0u) | (b_part ? 2u : 0u)', '(a_part ? 2u : 0u) | (b_part ? 1u : 0u)'),
    ('H5 child check of a', 'gpu', TREES, 'const bool a_part = (a.status & OCT_PART) != 0u && a.child != 0ul;', 'const bool a_part = (a.status & OCT_PART) != 0u;'),
    ('H6 child check of b', 'gpu', TREES, 'const bool b_part = (b.status & OCT_PART) != 0u && b.child != 0ul;', 'const bool b_part = (b.status & OCT_PART) != 0u;'),
    ('H7 last level of b', 'gpu', TREES, '((a.status | b.status) & OCT_LAST_LEVEL) != 0u;', '(a.status & OCT_LAST_LEVEL) != 0u;'),
    ('H8 last level of a', 'gpu', TREES, '((a.status | b.status) & OCT_LAST_LEVEL) != 0u;', '(b.status & OCT_LAST_LEVEL) != 0u;'),
    ('H9 leaves of a against uniform b', 'gpu', TREES, 'if (node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)))', 'if (true)'),
    ('H10 leaves of b against uniform a', 'gpu', TREES, 'else if (node_occupied(oct_load_leaf(OctLeafBlock(b.child), j)))', 'else if (true)'),
    ('H11 both parts', 'gpu', TREES, 'if (a_part && b_part)\n            {\n                if (node_occupied', 'if (a_part || b_part)\n            {\n                if (node_occupied'),
    ('CT14 threshold', 'gpu', CPP, 'options.mark_collisions ? 1u : 0u, options.occupancy_threshold, uint32_t(capacity) });', 'options.mark_collisions ? 1u : 0u, 10, uint32_t(capacity) });'),
    ('CT18 root with child', 'gpu', CPP, 'const bool a_part = (root_a.status & kPart) && root_a.child != 0, b_part', 'const bool a_part = (root_a.status & kPart), b_part'),
    ('CT19 root with child of b', 'gpu', CPP, 'b_part = (root_b.status & kPart) && root_b.child != 0;', 'b_part = (root_b.status & kPart);'),
    ('RT7 stand in of the reference', 'gpu', REF, 'visit(a_part ? child(a_tree, a, c) : a, b_part ? child(b_tree, b, c) : b, level - 1, a_part, b_part);', 'visit(a_part ? child(a_tree, a, c) : a, b_part ? child(b_tree, b, c) : b, level - 1, a_part || b_part, b_part);'),
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
