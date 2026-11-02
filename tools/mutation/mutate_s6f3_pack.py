# Hand-written mutation runner (slice S6f-3, Octree::pack_free_space: dense level-0 map from the ray counts, packing of 2x2x2 groups into cubes level by level, sorted lists): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6f3_pack.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
VALUES = V + 'shaders/octree_free_values.comp'
PACK = V + 'shaders/octree_free_pack.comp'
CPP = V + 'src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')

CELL_OLD = "const uvec3 cell = uvec3(pc.lo_x, pc.lo_y, pc.lo_z) + 2u * g + uvec3(k & 1u, (k >> 1u) & 1u, k >> 2u);"
CELL_NEW = "const uvec3 cell = uvec3(pc.lo_x, pc.lo_y, pc.lo_z) + 2u * g + uvec3((k >> 1u) & 1u, k & 1u, k >> 2u);"

MUT = [
    ('F1 count 1 is not free', VALUES, "if (count != 0u)", "if (count > 1u)"),
    ('F2 count limit 12', VALUES, "min(count, 255u)", "min(count, 12u)"),
    ('F3 min instead of max', VALUES, "max(pc.free_update * int(min(count, 255u)), pc.min_probability)", "min(pc.free_update * int(min(count, 255u)), pc.min_probability)"),
    ('F4 box edge inclusive', VALUES, "rel.x < int(pc.box_size_x)", "rel.x <= int(pc.box_size_x)"),
    ('F5 y and z swapped in the count index', VALUES, "pc.box_size_x * (uint(rel.y) + pc.box_size_y * uint(rel.z))", "pc.box_size_x * (uint(rel.z) + pc.box_size_z * uint(rel.y))"),
    ('F6 x of the cell from y', VALUES, "pc.lo_x + x, pc.lo_y + y, pc.lo_z + z", "pc.lo_x + y, pc.lo_y + y, pc.lo_z + z"),
    ('F7 deterministic value taken for probabilistic', VALUES, "pc.prob == 0u ? 1 : max", "pc.prob != 0u ? 1 : max"),
    ('G1 pack without the equal values', PACK, "free_mask == 255u && equal", "free_mask == 255u"),
    ('G2 pack equal values without all being free', PACK, "free_mask == 255u && equal", "equal"),
    ('G3 equality ignored', PACK, "equal = equal && v[k] == v[0];", "equal = equal;"),
    ('G4 empty group not skipped (equivalent: nothing is emitted for it)', PACK, "if (free_mask == 0u)\n        return;\n", ""),
    ('G5 parent not halved', PACK, "uvec3(pc.lo_x, pc.lo_y, pc.lo_z) / 2u + g -", "uvec3(pc.lo_x, pc.lo_y, pc.lo_z) + g -"),
    ('G6 parents not counted', PACK, "atomicAdd(counter[1], 1u);", "atomicAdd(counter[1], 0u);"),
    ('G7 parent value of the last child', PACK, "next_cells[parent.x + pc.next_size_x * (parent.y + pc.next_size_y * parent.z)] = v[0];", "next_cells[parent.x + pc.next_size_x * (parent.y + pc.next_size_y * parent.z)] = v[7];"),
    ('G8 emitted cells not counted', PACK, "const uint slot = atomicAdd(counter[0], 1u);", "const uint slot = atomicAdd(counter[0], 0u);"),
    ('G9 capacity guard off by one (equivalent while the counts agree)', PACK, "slot < pc.capacity)", "slot <= pc.capacity)"),
    ('G10 writes in the counting pass too (equivalent: the capacity is 0 there)', PACK, "if (pc.count_only == 0u && slot < pc.capacity)", "if (slot < pc.capacity)"),
    ('G11 cell id with x and y swapped', PACK, CELL_OLD, CELL_NEW),
    ('G12 value of the first cell for every cell', PACK, "out_values[slot] = v[k];", "out_values[slot] = v[0];"),
    ('G13 group row from the wrong size', PACK, "(idx / hx) % hy", "(idx / hx) % hx"),
    ('G14 only positive values are free', PACK, "if (v[k] != 0)", "if (v[k] > 0)"),
    ('G15 one group too many (equivalent in safe mode: it reads a zero group)', PACK, "if (idx >= pc.groups)", "if (idx > pc.groups)"),
    ('G16 next map not written', PACK, "if (pc.count_only == 0u)\n            next_cells", "if (false)\n            next_cells"),
    ('P1 box not aligned down', CPP, "lo[a] = (box_min[a] >> l) & ~1u;", "lo[a] = box_min[a] >> l;"),
    ('P2 box not aligned up', CPP, ">> l) | 1u) - lo[a] + 1;", ">> l)) - lo[a] + 1;"),
    ('P3 limit exclusive', CPP, "if (cells > ray_map_limit || cells > UINT32_MAX)", "if (cells >= ray_map_limit || cells > UINT32_MAX)"),
    ('P4 packing above the root not detected (equivalent: cannot happen)', CPP, "if (parents > 0 && l + 1 == kLevelCount)", "if (false)"),
    ('P5 empty level does not stop (equivalent: one more empty pass)', CPP, "if (emitted == 0 && parents == 0)\n\t\t\t\t\tbreak;", "if (emitted == 0 && parents == 0)\n\t\t\t\t\t;"),
    ('P6 second pass not checked (equivalent: defensive)', CPP, "if (read_word(counter, 0) != emitted || read_word(counter, 1) != parents)", "if (false)"),
    ('P7 sort over 3 bits', CPP, "emitted, 42, keep_alive);", "emitted, 3, keep_alive);"),
    ('P8 no sort', CPP, "if (emitted > 1)\n\t\t\t\t\t\tprimitives", "if (false)\n\t\t\t\t\t\tprimitives"),
    ('P9 deterministic values for a probabilistic tree', CPP, "kind == OctreeKind::Probabilistic ? 1u : 0u, kFreeUpdateProbability", "0u, kFreeUpdateProbability"),
    ('P10 size of the counts not checked', CPP, "if (cells != map.counts.size())", "if (false)"),
    ('P11 end of the box not checked', CPP, "|| box_size[a] > Octree::kCoordinateLimit - box_min[a])", "|| false)"),
    ('P12 start of the box not checked', CPP, "|| box_min[a] >= Octree::kCoordinateLimit ||", "|| false ||"),
    ('P13 occupancy of a deterministic tree reported', CPP, "static_cast<int8_t>(kind == OctreeKind::Probabilistic ? values[i] : 0)", "static_cast<int8_t>(values[i])"),
    ('P14 update and minimum swapped in the push', CPP, "kFreeUpdateProbability, kMinProbability });", "kMinProbability, kFreeUpdateProbability });"),
    ('P15 box size and minimum swapped in the push', CPP, "{ box_min[0], box_min[1], box_min[2] },\n\t\t\t\t\t\t{ box_size[0], box_size[1], box_size[2] }, kind", "{ box_size[0], box_size[1], box_size[2] },\n\t\t\t\t\t\t{ box_min[0], box_min[1], box_min[2] }, kind"),
    ('P16 next map not cleared', CPP, "batch.fill(next, 0);", "batch.fill(next, 1);"),
    ('P17 counter of the second pass not cleared', CPP, "counter.upload(zero, sizeof(zero));\n\t\t\t\tpush.count_only = 0;", "push.count_only = 0;"),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=VulkanTest.OctreePack*:VulkanTest.OctreeRaysAndPacking'],
                               env=test_env, capture_output=True, text=True, timeout=1200)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
