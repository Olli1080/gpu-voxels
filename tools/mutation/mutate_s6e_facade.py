# Hand-written mutation runner (slice S6e, octree facade and storage): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6e_facade.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
POINTS = V + 'shaders/octree_points_to_ids.comp'
CPP = V + 'src/Octree.cpp'
STORE = V + 'src/OctreeStorage.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')

MUT = [
    ('F1 floor replaced by trunc (negative points near 0 become voxel 0)', POINTS, "floor(vec3(points[p]", "trunc(vec3(points[p]"),
    ('F2 upper bound inclusive (x)', POINTS, "f.x < 16384.0 &&", "f.x <= 16384.0 &&"),
    ('F3 lower bound of x missing', POINTS, "f.x >= 0.0 && f.x < 16384.0", "f.x < 16384.0"),
    ('F4 lower bound of y missing', POINTS, "f.y >= 0.0 && f.y < 16384.0", "f.y < 16384.0"),
    ('F5 upper bound of y missing', POINTS, "f.y >= 0.0 && f.y < 16384.0", "f.y >= 0.0"),
    ('F6 upper bound of z missing', POINTS, "f.z >= 0.0 && f.z < 16384.0", "f.z >= 0.0"),
    ('F7 dropped points not counted', POINTS, "        atomicAdd(dropped[0], 1u);\n", "        ;\n"),
    ('F8 multiplication instead of division', POINTS, "/ pc.voxel_side_length", "* pc.voxel_side_length"),
    ('F9 first point of the range ignored', POINTS, "3u * (pc.first + i)", "3u * i"),
    ('F10 stride of 2 floats', POINTS, "3u * (pc.first + i)", "2u * (pc.first + i)"),
    ('F11 sentinel below the valid ids', POINTS, "uint64_t id = 1ul << 42;", "uint64_t id = 1ul << 41;"),
    ('F12 y and z swapped', POINTS, "morton_code60(uvec3(f))", "morton_code60(uvec3(f.x, f.z, f.y))"),
    ('G1 sort key of 42 bits (the sentinel needs 43)', CPP, "PointsPush{ n, first, voxel_side_length });\n				batch.barrier();\n				if (n > 1)\n					primitives.record_sort_pairs64(batch, ids, values, n, 43,", "PointsPush{ n, first, voxel_side_length });\n				batch.barrier();\n				if (n > 1)\n					primitives.record_sort_pairs64(batch, ids, values, n, 42,"),
    ('G2 distinct ids over all points including the sentinels', CPP, "const uint32_t valid = n - dropped_now;", "const uint32_t valid = n;"),
    ('G3 dropped points not added to dropped_count', CPP, "			dropped += dropped_now;\n			const uint32_t valid", "			const uint32_t valid"),
    ('G4 no early return when every point is outside', CPP, "			if (valid == 0)\n				return DeviceBuffer(context, 8);\n", ""),
    ('G5 dropped counter not cleared (equivalent if the fresh buffer is zero)', CPP, "				batch.fill(dropped_buffer, 0);\n				batch.barrier();\n", ""),
    ('G6 runs of meanings not propagated in between', CPP, "if (propagate_now || end < cloud.num_clouds())", "if (propagate_now)"),
    ('G7 occupied and free swapped', CPP, "if (meaning == eBVM_OCCUPIED)\n				return OctreeDatum::Occupied;", "if (meaning == eBVM_OCCUPIED)\n				return OctreeDatum::Free;"),
    ('G8 collision accepted as occupied', CPP, "if (meaning == eBVM_OCCUPIED)\n				return OctreeDatum::Occupied;", "if (meaning == eBVM_OCCUPIED || meaning == eBVM_COLLISION)\n				return OctreeDatum::Occupied;"),
    ('G9 one meaning per sub cloud not checked', CPP, "			if (meanings.size() != cloud.num_clouds())\n				throw VulkanError(\"Octree::insert_meta_point_cloud: expected one meaning per sub cloud\");\n", ""),
    ('G10 side length not required', CPP, "			if (!(voxel_side_length > 0.0f))\n				throw", "			if (false)\n				throw"),
    ('G11 outside count of runs not summed', CPP, "outside += insert_points(cloud.points()", "outside = insert_points(cloud.points()"),
    ('G12 coordinate list returns the total dropped count', CPP, "return static_cast<uint32_t>(impl_->dropped - dropped_before);", "return static_cast<uint32_t>(impl_->dropped);"),
    ('G13 empty point list uploads a buffer of size 0', CPP, "			if (!points.empty())\n			{\n				static_assert(sizeof(Point3f) == 12);", "			if (true)\n			{\n				static_assert(sizeof(Point3f) == 12);"),
    ('G14 point limit of the point span missing (equivalent: the check of the run refuses it as well)', CPP, "			if (points.size() > kMaxVoxels)\n				throw VulkanError(\"Octree::insert_point_cloud: too many points in one insert\");\n", ""),
    ('G15 point limit of a run missing (equivalent: the sort primitive refuses it as well)', CPP, "			if (n > kMaxVoxels)\n				throw VulkanError(\"Octree::insert_point_cloud: too many points in one insert\");\n", ""),
    ('G16 point inserts do not note the pending ids (equivalent: propagate takes the exact path, only slower)', CPP, "				insert_sorted(ids, count, 0, datum, OctreeInsertMode::Hard);\n				note_pending(count, true);", "				insert_sorted(ids, count, 0, datum, OctreeInsertMode::Hard);"),
    ('G17 side length setter does not validate', CPP, "		if (!std::isfinite(voxel_side_length) || voxel_side_length < 0.0f)", "		if (false)"),
    ('G18 constructor does not store the side length', CPP, "		impl_->voxel_side_length = voxel_side_length;\n	}\n\n	Octree::~Octree()", "	}\n\n	Octree::~Octree()"),
    ('M1 magic not checked', STORE, "if (std::memcmp(header, kMagic, 8) != 0)", "if (false)"),
    ('M2 version not checked', STORE, "if (get<uint32_t>(header + 8) != kVersion)", "if (false)"),
    ('M3 kind of the target not checked', STORE, "if (kind != kind_code(tree.kind()))", "if (false)"),
    ('M4 reserved header word not checked', STORE, " || get<uint32_t>(header + 20) != 0)", ")"),
    ('M5 side length not validated', STORE, "if (!std::isfinite(side) || side < 0.0f)", "if (false)"),
    ('M6 node count not limited (equivalent: the data ends first)', STORE, "if (count > kMaxNodes)", "if (false)"),
    ('M7 checksum not compared', STORE, "if (get<uint64_t>(trailer) != hash)", "if (false)"),
    ('M8 header not part of the checksum', STORE, "uint64_t hash = fnv1a(kFnvBasis, header, kHeaderBytes);", "uint64_t hash = kFnvBasis;"),
    ('M9 reserved record word not checked', STORE, " || get<uint32_t>(record + 12) != 0)", ")"),
    ('M10 has_children byte not limited to 0 / 1', STORE, "if (record[11] > 1 ", "if (false "),
    ('M11 write does not propagate first', STORE, "		tree.propagate();\n		const std::vector<OctreeNode> nodes", "		const std::vector<OctreeNode> nodes"),
    ('M12 has_children not written', STORE, "record[11] = n.has_children ? 1 : 0;", "record[11] = 0;"),
    ('M13 occupancy not written', STORE, "record[10] = static_cast<uint8_t>(n.occupancy);", "record[10] = 0;"),
    ('M14 side length not restored', STORE, "		tree.set_voxel_side_length(side);\n", ""),
    ('M15 side length not written', STORE, "put<float>(bytes.data() + 16, tree.voxel_side_length());", "put<float>(bytes.data() + 16, 0.0f);"),
    ('M16 files written in text mode', STORE, "std::ofstream out(path, std::ios::binary | std::ios::trunc);", "std::ofstream out(path, std::ios::trunc);"),
    ('M17 level of a record read from the status byte', STORE, "node.level = record[8];", "node.level = record[9] & 15;"),
    ('M18 reading stops after the first batch', STORE, "done += batch;", "done = count;"),
    ('M19 nodes read in batches of 1 record (not a bug: equivalent)', STORE, "constexpr std::size_t kReadBatch = 65536;", "constexpr std::size_t kReadBatch = 1;"),
    ('M20 kind of the tree not written', STORE, "put<uint32_t>(bytes.data() + 12, kind_code(tree.kind()));", "put<uint32_t>(bytes.data() + 12, 0);"),
    ('M21 target replaced before the structure is checked', STORE, "		tree.load_nodes(nodes); // checks the structure; the tree stays as it was if this throws\n", "		tree.clear();\n		tree.load_nodes(nodes);\n"),
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
                                '--gtest_filter=VulkanTest.OctreePointCloud*:VulkanTest.OctreeVoxelSideLength:VulkanTest.OctreeCoordinateList*:VulkanTest.OctreeMetaPointCloud*:VulkanTest.OctreeStorage*:VulkanTest.OctreeNativeFormat*'],
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
