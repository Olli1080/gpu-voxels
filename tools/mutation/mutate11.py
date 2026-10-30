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
HIST64 = SH + r'\radix_histogram64.comp'
SCATTER64 = SH + r'\radix_scatter64.comp'
CPP = V + r'\src\Octree.cpp'
PRIM = V + r'\src\Primitives.cpp'
FORMAT = V + r'\include\gpu_voxels\vulkan\OctreeFormat.h'
REF = r'tests\reference\octree_reference.h'
CPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\Debug\test_octree_reference.exe')
GPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
OCTREE_FILTER = '--gtest_filter=*OctreeCollid*:*OctreeCollision*:-*LargeSets*'
SORT_FILTER = '--gtest_filter=*SortPairs64*'
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

mutations = [
    # ---- collide kernel
    ('GC1 bit map occupancy', 'gpu', COLLIDE, 'case 0u: has_flags = true; occupied = any_flag(8u * i); break;', 'case 0u: has_flags = true; occupied = any_flag(8u * i + 1u); break;'),
    ('GC2 prob threshold', 'gpu', COLLIDE, 'occupied = int(payload[i]) >= pc.map_threshold; break;', 'occupied = int(payload[i]) > pc.map_threshold; break;'),
    ('GC3 list occupancy', 'gpu', COLLIDE, 'case 2u: has_flags = true; occupied = any_flag(8u * i); break;', 'case 2u: has_flags = true; occupied = any_flag(i); break;'),
    ('GC4 ids without flags', 'gpu', COLLIDE, 'occupied = !has_flags || any_flag(8u * i);', 'occupied = !has_flags && any_flag(8u * i);'),
    ('GC5 ids flag option', 'gpu', COLLIDE, 'has_flags = (pc.options & 8u) != 0u;', 'has_flags = (pc.options & 4u) != 0u;'),
    ('GC6 id words', 'gpu', COLLIDE, 'packUint2x32(uvec2(positions[2u * i], positions[2u * i + 1u]))', 'packUint2x32(uvec2(positions[2u * i + 1u], positions[2u * i]))'),
    ('GC7 id bound', 'gpu', COLLIDE, 'if ((id >> 42) != 0ul)\n            return;\n    }', 'if ((id >> 43) != 0ul)\n            return;\n    }'),
    ('GC8 list y', 'gpu', COLLIDE, 'int(positions[3u * i + 1u]), int(positions[3u * i + 2u])', 'int(positions[3u * i + 2u]), int(positions[3u * i + 2u])'),
    ('GC9 map y', 'gpu', COLLIDE, 'ivec3(int(i % pc.dim_x), int(t % pc.dim_y), int(t / pc.dim_y))', 'ivec3(int(i % pc.dim_x), int(t % pc.dim_x), int(t / pc.dim_y))'),
    ('GC10 map row', 'gpu', COLLIDE, 'const uint t = i / pc.dim_x;', 'const uint t = i / pc.dim_y;'),
    ('GC11 offset sign', 'gpu', COLLIDE, 'c += ivec3(pc.offset_x, pc.offset_y, pc.offset_z);', 'c -= ivec3(pc.offset_x, pc.offset_y, pc.offset_z);'),
    ('GC12 negative coordinates', 'gpu', COLLIDE, 'if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(16384))))', 'if (any(greaterThanEqual(c, ivec3(16384))))'),
    ('GC13 upper bound', 'gpu', COLLIDE, 'greaterThanEqual(c, ivec3(16384))', 'greaterThanEqual(c, ivec3(16385))'),
    ('GC14 min level', 'gpu', COLLIDE, 'while (level > pc.min_level &&', 'while (level >= pc.min_level &&'),
    ('GC15 slot', 'gpu', COLLIDE, 'slot = uint(id >> (3u * level)) & 7u;', 'slot = uint(id >> (3u * (level + 1u))) & 7u;'),
    ('GC16 unknown test of prob nodes', 'gpu', COLLIDE, 'node.occupancy != OCT_PROB_UNKNOWN && node.occupancy >= pc.tree_threshold', 'node.occupancy >= pc.tree_threshold'),
    ('GC17 det collision bit', 'gpu', COLLIDE, '(node.status & OCT_OCCUPIED) != 0u;\n    const bool unknown', '(node.status & OCT_FREE) != 0u;\n    const bool unknown'),
    ('GC18 det unknown bit', 'gpu', COLLIDE, 'const bool unknown = (node.status & OCT_UNKNOWN) != 0u;', 'const bool unknown = (node.status & OCT_FREE) != 0u;'),
    ('GC19 unknown option', 'gpu', COLLIDE, '(pc.options & 4u) != 0u && unknown', '(pc.options & 2u) != 0u && unknown'),
    ('GC20 collision counter', 'gpu', COLLIDE, 'atomicAdd(results[0], 1u);', 'atomicAdd(results[1], 1u);'),
    ('GC21 flag option', 'gpu', COLLIDE, 'if ((pc.options & 1u) != 0u)\n    {\n        if (level == 0u)', 'if ((pc.options & 2u) != 0u)\n    {\n        if (level == 0u)'),
    ('GC22 det leaf bit', 'gpu', COLLIDE, 'OCT_COLLISION << ((slot & 3u) * 8u)', 'OCT_COLLISION << ((slot & 3u) * 4u)'),
    ('GC23 prob leaf bit', 'gpu', COLLIDE, 'OCT_COLLISION << ((slot & 1u) * 16u)', 'OCT_COLLISION << ((slot & 1u) * 8u)'),
    ('GC24 inner node word', 'gpu', COLLIDE, 'atomicOr(OctInnerBlock(block).w[OCT_INNER_WORDS * slot], OCT_COLLISION);', 'atomicOr(OctInnerBlock(block).w[slot], OCT_COLLISION);'),
    ('GC25 flags option', 'gpu', COLLIDE, '(pc.options & 2u) != 0u && has_flags', '(pc.options & 4u) != 0u && has_flags'),
    ('GC26 flag words', 'gpu', COLLIDE, 'atomicOr(results[2u + w], payload[8u * i + w]);', 'atomicOr(results[2u + w], payload[8u * i]);'),
    ('GC27 root slot', 'gpu', COLLIDE, 'OctNode node = oct_load_inner(OctInnerBlock(block), 0u);', 'OctNode node = oct_load_inner(OctInnerBlock(block), 1u);'),
    ('GC28 prob node words', 'gpu', COLLIDE, 'const uint OCT_INNER_WORDS = 3u;', 'const uint OCT_INNER_WORDS = 2u;'),
    ('GC29 det node words', 'gpu', COLLIDE, 'const uint OCT_INNER_WORDS = 2u;', 'const uint OCT_INNER_WORDS = 3u;'),
    ('GC30 leaf level', 'gpu', COLLIDE, 'if (level == 0u)\n            node = oct_load_leaf(OctLeafBlock(block), slot);', 'if (level == 1u)\n            node = oct_load_leaf(OctLeafBlock(block), slot);'),
    ('GC31 unoccupied probes', 'gpu', COLLIDE, 'if (!occupied)\n        return;', ''),
    ('GC32 no collision', 'gpu', COLLIDE, 'if (!collision)\n        return;', ''),
    ('GC33 descent needs child', 'gpu', COLLIDE, '(node.status & OCT_PART) != 0u && node.child != 0ul)', '(node.status & OCT_PART) != 0u)'),
    # ---- clear kernel
    ('CL1 clear operation', 'gpu', CLEAR, 'words[i * pc.stride] &= ~pc.mask;', 'words[i * pc.stride] |= pc.mask;'),
    ('CL2 stride', 'gpu', CLEAR, 'words[i * pc.stride] &= ~pc.mask;', 'words[i] &= ~pc.mask;'),
    # ---- host code
    ('CP1 bit map mode', 'gpu', CPP, 'return impl_->collide(0, &map.voxel_buffer()', 'return impl_->collide(1, &map.voxel_buffer()'),
    ('CP2 prob map mode', 'gpu', CPP, 'return impl_->collide(1, &map.voxel_buffer()', 'return impl_->collide(0, &map.voxel_buffer()'),
    ('CP3 list buffers', 'gpu', CPP, 'impl_->collide(2, &list.voxel_buffer(), &list.coordinates_buffer()', 'impl_->collide(2, &list.coordinates_buffer(), &list.voxel_buffer()'),
    ('CP4 ids with flags', 'gpu', CPP, 'Dim3ui{ 1, 1, 1 }, true, options);', 'Dim3ui{ 1, 1, 1 }, false, options);'),
    ('CP5 ids without flags', 'gpu', CPP, 'nullptr, &ids, morton_ids.size(), Dim3ui{ 1, 1, 1 }, false, options);', 'nullptr, &ids, morton_ids.size(), Dim3ui{ 1, 1, 1 }, true, options);'),
    ('CP6 set flag option', 'gpu', CPP, '(options.set_collision_flags ? 1u : 0u)', '(options.set_collision_flags ? 0u : 1u)'),
    ('CP7 compute flag option', 'gpu', CPP, '(options.compute_voxel_flags ? 2u : 0u)', '(options.compute_voxel_flags ? 0u : 2u)'),
    ('CP8 unknown option', 'gpu', CPP, '(options.count_unknown ? 4u : 0u)', '(options.count_unknown ? 0u : 4u)'),
    ('CP9 thresholds', 'gpu', CPP, 'options.map_occupancy_threshold, options.tree_occupancy_threshold, 0 });', 'options.tree_occupancy_threshold, options.map_occupancy_threshold, 0 });'),
    ('CP10 prob map flags check', 'gpu', CPP, 'if (options.compute_voxel_flags && mode == 1)', 'if (options.compute_voxel_flags && mode == 0)'),
    ('CP11 unknown result', 'gpu', CPP, 'result.collisions_with_unknown = raw[1];', 'result.collisions_with_unknown = raw[0];'),
    ('CP12 flag words', 'gpu', CPP, 'result.voxel_flags[w] = raw[2 + w];', 'result.voxel_flags[w] = raw[2];'),
    ('CP13 inner node words', 'gpu', CPP, 'uint32_t(inner_block_bytes(kind) / 32)', 'uint32_t(inner_block_bytes(kind) / 16)'),
    ('CP14 clear mask', 'gpu', CPP, 'uint32_t(kCollision) };', 'uint32_t(kStatic) };'),
    ('CP15 prob leaf mask', 'gpu', CPP, '0x00200020u : 0x20202020u', '0x00400040u : 0x20202020u'),
    ('CP16 det leaf mask', 'gpu', CPP, '0x00200020u : 0x20202020u', '0x00200020u : 0x40404040u'),
    ('CP17 inner node count', 'gpu', CPP, 'static_cast<uint32_t>(blocks * 8), uint32_t(inner_block_bytes(kind) / 32)', 'static_cast<uint32_t>(blocks * 4), uint32_t(inner_block_bytes(kind) / 32)'),
    ('CP18 leaf word count', 'gpu', CPP, 'static_cast<uint32_t>(blocks * leaf_block_bytes(kind) / 4)', 'static_cast<uint32_t>(blocks * leaf_block_bytes(kind) / 8)'),
    ('CP19 check_tree ignores the collision bit', 'gpu', CPP, 'expected.status != (node.status & ~kCollision)', 'expected.status != node.status'),
    ('CP20 flag count check', 'gpu', CPP, 'flags.size() != morton_ids.size()', 'flags.size() > morton_ids.size()'),
    ('CP21 clear skips levels', 'gpu', CPP, 'if (level > 0)\n				{\n					push = {', 'if (level > 1)\n				{\n					push = {'),
    # ---- reference
    ('RC1 stop level', 'gpu', REF, 'node->level > min_level)\n				node = &nodes_.at({ node->level - 1, id >> (3 * (node->level - 1)) });\n			return *node;\n		}\n\n		/// Collision', 'node->level >= min_level)\n				node = &nodes_.at({ node->level - 1, id >> (3 * (node->level - 1)) });\n			return *node;\n		}\n\n		/// Collision'),
    ('RC2 det occupied', 'gpu', REF, '(n.status & status::OCCUPIED) != 0 : (n.occupancy != kUnknownOccupancy', '(n.status & status::UNKNOWN) != 0 : (n.occupancy != kUnknownOccupancy'),
    ('RC3 unknown counted only for non-colliding', 'gpu', REF, 'if (unknown)\n					++result.with_unknown;\n				if (!occupied)\n					continue;', 'if (unknown && !occupied)\n					++result.with_unknown;\n				if (!occupied)\n					continue;'),
    ('RC4 flags or', 'gpu', REF, 'result.flags[w] |= (*flags)[i][w];', 'result.flags[w] = (*flags)[i][w];'),
    ('RC5 flagged node', 'gpu', REF, 'result.flagged.insert({ n.level, n.id });', 'result.flagged.insert({ n.level, n.id >> 3 });'),
    ('RC6 outside ids', 'gpu', REF, 'if (ids[i] >= (uint64_t(1) << 42))\n					continue;', 'if (ids[i] > (uint64_t(1) << 42))\n					continue;'),
    ('RC7 dump collision bit', 'gpu', REF, 'n.status |= status::COLLISION;\n			return result;', 'n.status |= status::STATIC;\n			return result;'),
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
