# Hand-written mutation runner (slice S6f-4, Octree::insert_point_cloud_with_free_space: the whole sensor pipeline, the value binding of the two insert kernels, the id shift): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6f4_sensor.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
LEAVES = V + 'shaders/octree_insert_leaves.glsl'
NODES = V + 'shaders/octree_insert_nodes.glsl'
SHIFT = V + 'shaders/octree_shift_ids.comp'
CPP = V + 'src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')

LEAF_LINE = "const int delta = pc.use_values != 0u ? values[i] : (pc.datum == OCT_OCCUPIED ? 127 : -127);"
NODE_LINE = "node.occupancy = pc.use_values != 0u ? values[i] : (pc.datum == OCT_OCCUPIED ? 127 : -127);"

MUT = [
    ('L1 leaves take the first value for every voxel', LEAVES, LEAF_LINE, LEAF_LINE.replace("values[i]", "values[0]")),
    ('L2 leaves ignore the values', LEAVES, LEAF_LINE, "const int delta = (pc.datum == OCT_OCCUPIED ? 127 : -127);"),
    ('N1 nodes take the first value for every node', NODES, NODE_LINE, NODE_LINE.replace("values[i]", "values[0]")),
    ('N2 nodes ignore the values', NODES, NODE_LINE, "node.occupancy = (pc.datum == OCT_OCCUPIED ? 127 : -127);"),
    ('S1 shift by half', SHIFT, "<< pc.shift;", "<< (pc.shift / 2u);"),
    ('S2 high word of the shifted id', SHIFT, "ids[2u * i + 1u] = uint(id >> 32);", "ids[2u * i + 1u] = uint(id >> 31);"),
    ('I1 leaves told there are no values', CPP, "InsertLeafPush{ n, datum_bits, values ? 1u : 0u, 0, base[0] }", "InsertLeafPush{ n, datum_bits, 0u, 0, base[0] }"),
    ('I2 nodes told there are no values', CPP, "target, values ? 1u : 0u, 0, base[target] }", "target, 0u, 0, base[target] }"),
    ('I3 values buffer not bound to the leaves', CPP, "values ? values : &no_values }, n, InsertLeafPush", "&no_values }, n, InsertLeafPush"),
    ('I4 values buffer not bound to the nodes', CPP, "values ? values : &no_values }, n,\n\t\t\t\t\tInsertNodePush", "&no_values }, n,\n\t\t\t\t\tInsertNodePush"),
    ('F1 free space inserted hard', CPP, "OctreeDatum::Free, OctreeInsertMode::Soft, prob ? &cubes[l].values : nullptr);", "OctreeDatum::Free, OctreeInsertMode::Hard, prob ? &cubes[l].values : nullptr);"),
    ('F2 objects inserted soft', CPP, "OctreeDatum::Occupied, OctreeInsertMode::Hard, prob ? &objects.occupancy : nullptr);", "OctreeDatum::Occupied, OctreeInsertMode::Soft, prob ? &objects.occupancy : nullptr);"),
    ('F3 free cubes without their values', CPP, "OctreeInsertMode::Soft, prob ? &cubes[l].values : nullptr);", "OctreeInsertMode::Soft, nullptr);"),
    ('F4 objects without their occupancy', CPP, "OctreeInsertMode::Hard, prob ? &objects.occupancy : nullptr);", "OctreeInsertMode::Hard, nullptr);"),
    ('F5 cubes inserted as occupied', CPP, "target, OctreeDatum::Free, OctreeInsertMode::Soft", "target, OctreeDatum::Occupied, OctreeInsertMode::Soft"),
    ('F6 objects inserted as free', CPP, "object_shift, OctreeDatum::Occupied, OctreeInsertMode::Hard", "object_shift, OctreeDatum::Free, OctreeInsertMode::Hard"),
    ('F7 cube level without the scale', CPP, "const uint32_t target = l + free_shift;", "const uint32_t target = l;"),
    ('F8 cube ids shifted by the level of the free grid', CPP, "shift_ids(cubes[l].ids, cubes[l].count, 3 * target);", "shift_ids(cubes[l].ids, cubes[l].count, 3 * l);"),
    ('F9 objects inserted on level 0', CPP, "insert_sorted(objects.ids, objects.count, object_shift,", "insert_sorted(objects.ids, objects.count, 0,"),
    ('F10 object ids shifted by the free scale', CPP, "shift_ids(objects.ids, objects.count, 3 * object_shift);", "shift_ids(objects.ids, objects.count, 3 * free_shift);"),
    ('F11 origin without the free-space scale', CPP, "double(pose(a, 3)) / (double(voxel_side_length) * options.free_space_scale)", "double(pose(a, 3)) / double(voxel_side_length)"),
    ('F12 negative origin accepted', CPP, "if (!(v >= 0.0 && v < double(Octree::kCoordinateLimit >> free_shift)))", "if (!(v < double(Octree::kCoordinateLimit >> free_shift)))"),
    ('F13 origin limit of the tree grid', CPP, "if (!(v >= 0.0 && v < double(Octree::kCoordinateLimit >> free_shift)))", "if (!(v >= 0.0 && v < double(Octree::kCoordinateLimit)))"),
    ('F14 object grid limit of the tree grid', CPP, "options.object_model, Octree::kCoordinateLimit >> object_shift);", "options.object_model, Octree::kCoordinateLimit);"),
    ('F15 free grid limit of the tree grid', CPP, "{}, Octree::kCoordinateLimit >> free_shift);", "{}, Octree::kCoordinateLimit);"),
    ('F16 free space always bounded for propagate', CPP, "note_pending(cubes[l].count, target == 0);", "note_pending(cubes[l].count, true);"),
    ('F17 free grid taken from the objects', CPP, "if (options.free_space_scale != options.object_scale)", "if (false)"),
    ('F18 cubes not counted', CPP, "result.free_cubes += cubes[l].count;", ";"),
    ('F19 no propagate', CPP, "note_pending(objects.count, true);\n\t\t\t\t}\n\t\t\t}\n\t\t\tif (propagate_now)\n\t\t\t\tpropagate();", "note_pending(objects.count, true);\n\t\t\t\t}\n\t\t\t}"),
    ('F20 empty level of cubes not skipped', CPP, "if (cubes[l].count == 0)", "if (false)"),
    ('F21 outside count of the objects from the free grid (equivalent: all grids cover the same extent)', CPP, "result.outside_object = objects.outside;", "result.outside_object = free_runs.outside;"),
    ('F22 free voxels counted from the objects', CPP, "result.free_voxels = free_runs.count;", "result.free_voxels = objects.count;"),
    ('F23 scale limit 2^14', CPP, "s <= (1u << 13)", "s <= (1u << 14)"),
    ('F24 any scale accepted', CPP, "(s & (s - 1)) == 0 && ", ""),
    ('F25 scale 0 accepted', CPP, "s >= 1 && ", ""),
    ('F26 object level from the free scale', CPP, "std::countr_zero(options.object_scale)", "std::countr_zero(options.free_space_scale)"),
    ('F27 objects voxelized on the tree grid', CPP, "voxel_side_length * float(options.object_scale), options.object_model", "voxel_side_length, options.object_model"),
    ('F28 free space voxelized on the tree grid', CPP, "voxel_side_length * float(options.free_space_scale), {}", "voxel_side_length, {}"),
    ('F29 origin x and y swapped for the rays', CPP, "cast_rays(free_runs.ids, free_runs.count, origin)", "cast_rays(free_runs.ids, free_runs.count, OctreeVoxel{ origin.y, origin.x, origin.z })"),
    ('F30 cube above the root not detected (equivalent: cannot happen)', CPP, "if (target >= kRootLevel)", "if (target > kRootLevel)"),
    ('F31 voxel side length not required', CPP, "require_side_length(\"insert_point_cloud_with_free_space\");", ";"),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=VulkanTest.OctreeSensor*'],
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
