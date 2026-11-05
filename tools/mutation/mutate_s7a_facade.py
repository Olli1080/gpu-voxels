# Hand-written mutation runner (slice S7-1, GpuVoxels / GpuVoxelsMap: the dispatch to the seven classes, names, initialisation, the singleton): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s7a_facade.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
CPP = root + 'packages/gpu_voxels/gpu_voxels/vulkan/src/GpuVoxels.cpp'
HDR = root + 'packages/gpu_voxels/gpu_voxels/vulkan/include/gpu_voxels/vulkan/GpuVoxels.h'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

MUT = [
    ('M1 prob map built as bit map', CPP, "storage_.emplace<std::unique_ptr<ProbVoxelMap>>(std::make_unique<ProbVoxelMap>(context, dimensions, voxel_side_length));", "storage_.emplace<std::unique_ptr<BitVoxelMap>>(std::make_unique<BitVoxelMap>(context, dimensions, voxel_side_length));"),
    ('M2 prob octree built deterministic', CPP, "OctreeKind::Probabilistic, voxel_side_length", "OctreeKind::Deterministic, voxel_side_length"),
    ('M3 det octree built probabilistic', CPP, "OctreeKind::Deterministic, voxel_side_length", "OctreeKind::Probabilistic, voxel_side_length"),
    ('M4 octree without side length', CPP, "OctreeKind::Deterministic, voxel_side_length)", "OctreeKind::Deterministic)"),
    ('M5 bit map with x and y swapped', CPP, "std::make_unique<BitVoxelMap>(context, dimensions, voxel_side_length)", "std::make_unique<BitVoxelMap>(context, Dim3ui{ dimensions.y, dimensions.x, dimensions.z }, voxel_side_length)"),
    ('M6 counting list built as bit list', CPP, "storage_.emplace<std::unique_ptr<CountingVoxelList>>(std::make_unique<CountingVoxelList>(context, dimensions, voxel_side_length));", "storage_.emplace<std::unique_ptr<BitVoxelList>>(std::make_unique<BitVoxelList>(context, dimensions, voxel_side_length));"),
    ('M7 not built types without the message', CPP, "\" is not implemented\"", "\" is fine\""),
    ('M8 unknown type does not throw', CPP, "throw VulkanError(\"unknown map type \" + std::to_string(static_cast<unsigned>(type)));", "break;"),
    ('P1 points forwarded with another meaning', CPP, "return map->insert_point_cloud(points, meaning); }", "return map->insert_point_cloud(points, uint8_t(meaning ^ 1)); }"),
    ('P2 coordinates forwarded with another meaning', CPP, "return map->insert_coordinate_list(coordinates, meaning); }", "return map->insert_coordinate_list(coordinates, uint8_t(meaning ^ 1)); }"),
    ('P3 meta cloud forwarded with another meaning', CPP, "return map->insert_meta_point_cloud(cloud, meaning); }, storage_);", "return map->insert_meta_point_cloud(cloud, uint8_t(meaning ^ 1)); }, storage_);"),
    ('P4 per cloud meanings reduced to the first', CPP, "return map->insert_meta_point_cloud(cloud, meanings);", "return map->insert_meta_point_cloud(cloud, meanings[0]);"),
    ('D1 distance map: no check of the number of meanings', CPP, "if (meanings.size() != cloud.num_clouds())", "if (false)"),
    ('D2 distance map: other meanings accepted', CPP, "if (meaning != eBVM_OCCUPIED)\n\t\t\t\t\t\tthrow VulkanError(std::string(\"the map type \") + map_type_name(MT_DISTANCE_VOXELMAP)", "if (false)\n\t\t\t\t\t\tthrow VulkanError(std::string(\"the map type \") + map_type_name(MT_DISTANCE_VOXELMAP)"),
    ('C1 clear_map does not clear lists and octrees', CPP, "\t\t\t\tmap->clear();\n\t\t\telse\n\t\t\t\tmap->clear_map();", "\t\t\t\t;\n\t\t\telse\n\t\t\t\tmap->clear_map();"),
    ('C2 octree accepts every meaning for clear', CPP, "if (meaning != eBVM_OCCUPIED)\n\t\t\t\t\tthrow VulkanError(std::string(\"the map type \") + map_type_name(type_)", "if (false)\n\t\t\t\t\tthrow VulkanError(std::string(\"the map type \") + map_type_name(type_)"),
    ('C3 octree does not clear for occupied', CPP, "only supports the meaning eBVM_OCCUPIED\");\n\t\t\t\tmap->clear();", "only supports the meaning eBVM_OCCUPIED\");"),
    ('C4 clear meaning forwarded as another one', CPP, "map->clear_bit_voxel_meaning(meaning);\n\t\t\telse if", "map->clear_bit_voxel_meaning(uint8_t(meaning + 1));\n\t\t\telse if"),
    ('C5 unsupported clear does nothing', CPP, "unsupported(type_, \"clear_bit_voxel_meaning\");", ";"),
    ('C6 unsupported message without the type', CPP, "\" + map_type_name(type));", "\");"),
    ('R1 needs_rebuild true for other types', CPP, "return tree != nullptr && tree->needs_rebuild();", "return tree == nullptr || tree->needs_rebuild();"),
    ('R2 rebuild of an octree reports false', CPP, "tree->rebuild();\n\t\treturn true;", "tree->rebuild();\n\t\treturn false;"),
    ('R3 rebuild does not rebuild', CPP, "tree->rebuild();\n\t\treturn true;", "return true;"),
    ('R4 rebuild_if_needed always rebuilds', CPP, "if (!needs_rebuild())\n\t\t\treturn false;\n\t\trebuild();", "rebuild();"),
    ('R5 rebuild of other types reports true', CPP, "if (tree == nullptr)\n\t\t\treturn false;", "if (tree == nullptr)\n\t\t\treturn true;"),
    ('S1 octree memory off by one', CPP, "return map->memory_usage().bytes;", "return map->memory_usage().bytes + 1;"),
    ('S2 octree dimensions one short', CPP, "return { Octree::kCoordinateLimit, Octree::kCoordinateLimit, Octree::kCoordinateLimit };", "return { Octree::kCoordinateLimit - 1, Octree::kCoordinateLimit, Octree::kCoordinateLimit };"),
    ('S3 list dimensions as the legacy (entries, 1, 1)', CPP, "return map->reference_dimensions();", "return { map->size(), 1, 1 };"),
    ('S4 metric z from y', CPP, "static_cast<float>(dim.z) * side }", "static_cast<float>(dim.y) * side }"),
    ('S5 metric x without the side length', CPP, "{ static_cast<float>(dim.x) * side,", "{ static_cast<float>(dim.x),"),
    ('I1 empty x dimension accepted', CPP, "dimensions.x == 0 || ", ""),
    ('I2 empty y dimension accepted', CPP, "dimensions.y == 0 || ", ""),
    ('I3 empty z dimension accepted', CPP, " || dimensions.z == 0", ""),
    ('I4 side length 0 accepted', CPP, "!(voxel_side_length > 0.0f)", "!(voxel_side_length >= 0.0f)"),
    ('I5 infinite side length accepted', CPP, " || !std::isfinite(voxel_side_length)", ""),
    ('I6 size changed while maps exist', CPP, "if (!maps_.empty())\n\t\t\tthrow", "if (false)\n\t\t\tthrow"),
    ('I7 initialize does not mark the instance', CPP, "\t\tinitialized_ = true;", "\t\t;"),
    ('I8 owning context always created', CPP, "if (context_ == nullptr)\n\t\t{\n\t\t\towned_context_", "if (true)\n\t\t{\n\t\t\towned_context_"),
    ('I9 add_map before initialize allowed', CPP, "if (!initialized_)\n\t\t\tthrow VulkanError(\"initialize() has to be called before maps are added\");", ""),
    ('I10 context() without the check', CPP, "if (context_ == nullptr)\n\t\t\tthrow VulkanError(\"the Vulkan context does not exist before initialize()\");", ""),
    ('I11 map added with swapped dimensions', CPP, "*context_, type, dimensions_, voxel_side_length_", "*context_, type, Dim3ui{ dimensions_.y, dimensions_.x, dimensions_.z }, voxel_side_length_"),
    ('I12 map added with side length 1', CPP, "*context_, type, dimensions_, voxel_side_length_", "*context_, type, dimensions_, 1.0f"),
    ('N1 duplicate names replace silently', CPP, "if (maps_.contains(name))\n\t\t{\n\t\t\tGVL_LOG_ERROR(GpuVoxels, \"Map with name '\" << name << \"' already exists.\");\n\t\t\treturn {};\n\t\t}", ""),
    ('N2 del_map inverted', CPP, "if (maps_.erase(name) == 0)", "if (maps_.erase(name) != 0)"),
    ('N3 clear with meaning clears everything', CPP, "map->clear_bit_voxel_meaning(voxel_meaning);\n\t\treturn true;", "map->clear_map();\n\t\treturn true;"),
    ('N4 insert counter not stored (points)', CPP, "map->insert_point_cloud(points, voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = dropped;", "map->insert_point_cloud(points, voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = 0;"),
    ('N5 insert counter not stored (meta, meanings)', CPP, "map->insert_meta_point_cloud(cloud, voxel_meanings);\n\t\tif (outside != nullptr)\n\t\t\t*outside = dropped;", "map->insert_meta_point_cloud(cloud, voxel_meanings);\n\t\tif (outside != nullptr)\n\t\t\t*outside = 0;"),
    ('N6 insert counter not stored (meta, meaning)', CPP, "map->insert_meta_point_cloud(cloud, voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = dropped;", "map->insert_meta_point_cloud(cloud, voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = 0;"),
    ('N7 meta cloud inserted with another meaning', CPP, "map->insert_meta_point_cloud(cloud, voxel_meaning);", "map->insert_meta_point_cloud(cloud, uint8_t(voxel_meaning + 1));"),
    ('N8 points inserted with another meaning', CPP, "map->insert_point_cloud(points, voxel_meaning);", "map->insert_point_cloud(points, uint8_t(voxel_meaning + 1));"),
    ('N9 visualize reports true', CPP, "\t\treturn false;\n\t}\n}", "\t\treturn true;\n\t}\n}"),
    ('T1 singleton is always new', CPP, "GpuVoxelsSharedPtr instance = g_instance.lock();", "GpuVoxelsSharedPtr instance;"),
    ('T2 singleton not remembered', CPP, "\t\t\tg_instance = instance;\n", ""),
    ('T3 is_initialized without the flag', HDR, "return context_ != nullptr && initialized_;", "return context_ != nullptr;"),
    ('T4 is() inverted', HDR, "return as<T>() != nullptr;", "return as<T>() == nullptr;"),
    ('T5 const as() always empty', HDR, "const auto* held = std::get_if<std::unique_ptr<T>>(&storage_);\n\t\t\treturn held ? held->get() : nullptr;", "return nullptr;"),
    ('Y1 name of a type not built', CPP, "case MT_PROBAB_VOXELLIST: return \"MT_PROBAB_VOXELLIST\";", "case MT_PROBAB_VOXELLIST: return \"MT_PROBAB_VOXELMAP\";"),
    ('Y2 name of the distance map', CPP, "case MT_DISTANCE_VOXELMAP: return \"MT_DISTANCE_VOXELMAP\";", "case MT_DISTANCE_VOXELMAP: return \"MT_DISTANCE_MAP\";"),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=Facade*'],
                               env=env, capture_output=True, text=True, timeout=300)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
