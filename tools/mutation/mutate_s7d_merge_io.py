# Hand-written mutation runner (slice S7-4, merge of lists / maps and the disk formats): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s7d_merge_io.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk. Rebuild before using the executable afterwards.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
VL = V + 'src/VoxelList.cpp'
MF = V + 'src/MapFile.cpp'
GV = V + 'src/GpuVoxels.cpp'
BM = V + 'src/BitVoxelMap.cpp'
PM = V + 'src/ProbVoxelMap.cpp'
DM = V + 'src/DistanceVoxelMap.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

BOUNDS = "if (x < 0 || y < 0 || z < 0 || x >= int64_t(dimensions_.x) || y >= int64_t(dimensions_.y) || z >= int64_t(dimensions_.z))"
MAP_BRANCH = "if (other_bit_list == nullptr && other_counting_list == nullptr)\n\t\t\t\trefuse();\n\t\t\tsame_side();\n\t\t\tno_offset(\"a map\");"
WRITE_CATCH = "write_map_file(file, *map);\n\t\t\t}, storage_);\n\t\t}\n\t\tcatch (const MapFileError& e)"
READ_CATCH = "read_map_file(file, *map);\n\t\t\t}, storage_);\n\t\t}\n\t\tcatch (const MapFileError& e)"
TREE_WRITE = "OctreeNativeStorage().write_file(*tree, file.string());\n\t\t\t}\n\t\t\tcatch (const VulkanError& e)\n\t\t\t{\n\t\t\t\tGVL_LOG_ERROR(GpuVoxelsMap, e.what());\n\t\t\t\treturn false;"
TREE_READ = "OctreeNativeStorage().read_file(*tree, file.string());\n\t\t\t}\n\t\t\tcatch (const VulkanError& e)\n\t\t\t{\n\t\t\t\tGVL_LOG_ERROR(GpuVoxelsMap, e.what());\n\t\t\t\treturn false;"
SIGN_EXT = "static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(payload[i] & 0xFFu))) != payload[i]"

MUT = [
    ('L1 list merge: itself allowed', VL, "if (&other == this)\n\t\t\tthrow VulkanError(\"VoxelList: a list cannot", "if (false)\n\t\t\tthrow VulkanError(\"VoxelList: a list cannot"),
    ('L2 list merge: other voxel side length allowed', VL, "if (other.voxel_side_length() != voxel_side_length_)", "if (false)"),
    ('L3 list merge: x offset ignored', VL, "int64_t(coords[i].x) + offset.x;", "int64_t(coords[i].x);"),
    ('L4 list merge: y offset ignored', VL, "int64_t(coords[i].y) + offset.y;", "int64_t(coords[i].y);"),
    ('L5 list merge: z offset ignored', VL, "int64_t(coords[i].z) + offset.z;", "int64_t(coords[i].z);"),
    ('L6 list merge: x = -1 is inside', VL, BOUNDS, BOUNDS.replace("x < 0 ||", "x < -1 ||")),
    ('L7 list merge: y = -1 is inside', VL, BOUNDS, BOUNDS.replace("y < 0 ||", "y < -1 ||")),
    ('L8 list merge: z = -1 is inside', VL, BOUNDS, BOUNDS.replace("z < 0 ||", "z < -1 ||")),
    ('L9 list merge: x = dim is inside', VL, BOUNDS, BOUNDS.replace("x >= int64_t", "x > int64_t")),
    ('L10 list merge: y = dim is inside', VL, BOUNDS, BOUNDS.replace("y >= int64_t", "y > int64_t")),
    ('L11 list merge: z = dim is inside', VL, BOUNDS, BOUNDS.replace("z >= int64_t", "z > int64_t")),
    ('L12 list merge: meaning word index', VL, "bits[*meaning / 32] |=", "bits[*meaning / 64] |="),
    ('L13 list merge: meaning bit index', VL, "1u << (*meaning % 32);", "1u << (*meaning % 31);"),
    ('L14 list merge: bit payload taken from counting lists', VL, "else if (other.kind() == Kind::Bit)", "else if (other.kind() == Kind::Counting)"),
    ('L15 list merge: counts lost', VL, "payload.push_back(other.kind() == Kind::Counting ? source[i] : 0u);", "payload.push_back(0u);"),
    ('L16 list merge: everything dropped reports 0', VL, "if (kept == 0)\n\t\t\treturn n;", "if (kept == 0)\n\t\t\treturn 0;"),
    ('L17 list merge: dropped not reported', VL, "after the offset!\");\n\t\treturn n - kept;", "after the offset!\");\n\t\treturn 0;"),
    ('L18 list merge: ids over the old ones', VL, "uint64_t(kept) * sizeof(uint32_t), 0, uint64_t(first) * sizeof(uint32_t));", "uint64_t(kept) * sizeof(uint32_t), 0, 0);"),
    ('L19 list merge: coordinates over the old ones', VL, "0, uint64_t(first) * 3 * sizeof(uint32_t));", "0, 0);"),
    ('L20 list merge: payload over the old ones', VL, "0, uint64_t(first) * words * sizeof(uint32_t));", "0, 0);"),
    ('L21 list from map: meaning lost', VL, "return merge_entries(taken, {}, new_meaning);", "return merge_entries(taken, {}, std::nullopt);"),
    ('L22 list from map: nothing taken', VL, "taken.append_occupied_voxels(other);", ""),
    ('E1 entries: payload size not compared', VL, "payload.size() != n * words_of(kind)", "false"),
    ('E2 entries: coordinates not counted', VL, "coordinates.size() != n ||", ""),
    ('E3 entries: x = dim is inside', VL, "c.x >= dims.x || c.y >= dims.y || c.z >= dims.z", "c.x > dims.x || c.y >= dims.y || c.z >= dims.z"),
    ('E4 entries: y = dim is inside', VL, "c.x >= dims.x || c.y >= dims.y || c.z >= dims.z", "c.x >= dims.x || c.y > dims.y || c.z >= dims.z"),
    ('E5 entries: z = dim is inside', VL, "c.x >= dims.x || c.y >= dims.y || c.z >= dims.z", "c.x >= dims.x || c.y >= dims.y || c.z > dims.z"),
    ('E6 entries: ids need not match the coordinates', VL, "if (ids[i] != (uint64_t(c.z) * dims.y + c.y) * dims.x + c.x)", "if (false)"),
    ('E7 entries: equal ids are fine', VL, "ids[i] <= ids[i - 1]", "ids[i] < ids[i - 1]"),
    ('E8 entries: counts need not be sign extended', VL, "kind == Kind::Counting && " + SIGN_EXT, "false && " + SIGN_EXT),
    ('E9 entries: assign does not check', VL, "if (const auto problem = entries_problem(dimensions_, kind_, ids, coordinates, payload))", "if (const auto problem = entries_problem(dimensions_, kind_, ids, coordinates, payload); false)"),
    ('F1 file: magic not checked', MF, "if (std::memcmp(header, kMagic, 8) != 0)", "if (false)"),
    ('F2 file: version not checked', MF, "if (get<uint32_t>(header + 8) != kVersion)", "if (false)"),
    ('F3 file: type not checked', MF, "if (type != static_cast<uint32_t>(target.type))", "if (false)"),
    ('F4 file: only the x dimension is checked', MF, "if (!(dims == target.dims))", "if (dims.x != target.dims.x)"),
    ('F5 file: dimensions not checked', MF, "if (!(dims == target.dims))", "if (false)"),
    ('F6 file: side length not checked', MF, "if (get<float>(header + 28) != target.side)", "if (false)"),
    ('F7 file: first reserved word only', MF, "if (get<uint32_t>(header + 32) != 0 || get<uint32_t>(header + 36) != 0)", "if (get<uint32_t>(header + 32) != 0)"),
    ('F8 file: second reserved word only', MF, "if (get<uint32_t>(header + 32) != 0 || get<uint32_t>(header + 36) != 0)", "if (get<uint32_t>(header + 36) != 0)"),
    ('F9 file: the count of a map is not compared', MF, "if (target.is_list ? loaded.count > VoxelListBase::kMaxEntries : loaded.count != target.dims.voxel_count())", "if (loaded.count > VoxelListBase::kMaxEntries)"),
    ('F10 file: data after the end is fine', MF, "if (file_size != expected)", "if (file_size < expected)"),
    ('F11 file: the smallest file is too short', MF, "if (file_size < kHeaderBytes + kTrailerBytes)", "if (file_size <= kHeaderBytes + kTrailerBytes)"),
    ('F12 file: checksum not compared', MF, "if (get<uint64_t>(loaded.bytes.data() + body) != fnv1a(kFnvBasis, loaded.bytes.data(), body))", "if (false)"),
    ('F13 file: FNV basis', MF, "14695981039346656037ull", "14695981039346656038ull"),
    ('F14 file: FNV prime', MF, "1099511628211ull", "1099511628213ull"),
    ('F15 file: dimensions written as y x', MF, "put<uint32_t>(bytes, dims.x);\n\t\t\tput<uint32_t>(bytes, dims.y);", "put<uint32_t>(bytes, dims.y);\n\t\t\tput<uint32_t>(bytes, dims.x);"),
    ('F16 file: version written as 2', MF, "put<uint32_t>(bytes, kVersion);", "put<uint32_t>(bytes, kVersion + 1);"),
    ('F17 file: list coordinates before the ids', MF, "append(bytes, list.download_ids());\n\t\t\tappend(bytes, list.download_coordinates());", "append(bytes, list.download_coordinates());\n\t\t\tappend(bytes, list.download_ids());"),
    ('F18 file: counts read without sign extension', MF, "static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(at[loaded.count * 16 + i])))", "static_cast<uint32_t>(at[loaded.count * 16 + i])"),
    ('F19 file: distance bit 30 not refused', MF, "if ((words[i] >> 30) != 0)", "if ((words[i] >> 31) != 0)"),
    ('F20 file: bit list coordinates at the wrong offset', MF, "copy_out<Coord3ui>(at + loaded.count * 4, loaded.count);\n\t\tconst std::vector<uint32_t> bits", "copy_out<Coord3ui>(at + loaded.count * 3, loaded.count);\n\t\tconst std::vector<uint32_t> bits"),
    ('F21 file: counting list coordinates at the wrong offset', MF, "copy_out<Coord3ui>(at + loaded.count * 4, loaded.count);\n\t\tstd::vector<uint32_t> words(", "copy_out<Coord3ui>(at + loaded.count * 3, loaded.count);\n\t\tstd::vector<uint32_t> words("),
    ('F22 file: bit vectors at the wrong offset', MF, "at + loaded.count * 16, loaded.count * 8", "at + loaded.count * 12, loaded.count * 8"),
    ('F23 file: counts at the wrong offset', MF, "at[loaded.count * 16 + i]", "at[loaded.count * 12 + i]"),
    ('F24 file: prob. map written as a bit map', MF, "begin_file(MT_PROBAB_VOXELMAP, map.dimensions()", "begin_file(MT_BITVECTOR_VOXELMAP, map.dimensions()"),
    ('F25 file: checksum misses the last byte', MF, "put<uint64_t>(bytes, fnv1a(kFnvBasis, bytes.data(), bytes.size()));", "put<uint64_t>(bytes, fnv1a(kFnvBasis, bytes.data(), bytes.size() - 1));"),
    ('F26 file: a file that cannot be opened for writing is not noticed', MF, "fail(file, \"could not open the file for writing\");", ";"),
    ('F27 file: the list list is read as the wrong kind', MF, "VoxelListBase::Kind::Counting, ids, coordinates, words)", "VoxelListBase::Kind::Bit, ids, coordinates, words)"),
    ('V1 map merge: itself allowed', GV, "if (&other == this)\n\t\t\tthrow VulkanError(\"a map cannot be merged into itself\");", "if (false)\n\t\t\tthrow VulkanError(\"a map cannot be merged into itself\");"),
    ('V2 map merge: voxel side not compared (maps)', GV, "same_side();\n\t\t\tno_offset(\"a map\");", "no_offset(\"a map\");"),
    ('V3 map merge: voxel side not compared (bit list)', GV, "same_side();\n\t\t\tif (other_bit_list != nullptr)\n\t\t\t\treturn list->merge(*other_bit_list, voxel_offset, new_meaning);", "if (other_bit_list != nullptr)\n\t\t\t\treturn list->merge(*other_bit_list, voxel_offset, new_meaning);"),
    ('V4 map merge: voxel side not compared (counting list)', GV, "same_side();\n\t\t\tif (new_meaning)\n\t\t\t\tthrow VulkanError(\"merge into a counting list", "if (new_meaning)\n\t\t\t\tthrow VulkanError(\"merge into a counting list"),
    ('V5 map merge: offset into a map is fine', GV, "no_offset(\"a map\");", ""),
    ('V6 map merge: meaning into a map is fine', GV, "if (new_meaning)\n\t\t\t\tthrow VulkanError(\"merge into a map:", "if (false)\n\t\t\t\tthrow VulkanError(\"merge into a map:"),
    ('V7 map merge: meaning into a counting list is fine', GV, "if (new_meaning)\n\t\t\t\tthrow VulkanError(\"merge into a counting list", "if (false)\n\t\t\t\tthrow VulkanError(\"merge into a counting list"),
    ('V8 map merge: offset for a list from a map is fine', GV, "no_offset(\"a bit list from a map\");", ""),
    ('V9 map merge: coordinates inserted as free', GV, "insert_coordinate_list(coordinates, eBVM_OCCUPIED)", "insert_coordinate_list(coordinates, eBVM_FREE)"),
    ('V10 map merge: bit list offset lost', GV, "return list->merge(*other_bit_list, voxel_offset, new_meaning);", "return list->merge(*other_bit_list, {}, new_meaning);"),
    ('V11 map merge: counting source offset lost (bit list)', GV, "return list->merge(*other_counting_list, voxel_offset, new_meaning);", "return list->merge(*other_counting_list, {}, new_meaning);"),
    ('V12 map merge: bit list meaning lost', GV, "return list->merge(*other_bit_list, voxel_offset, new_meaning);", "return list->merge(*other_bit_list, voxel_offset, std::nullopt);"),
    ('V13 map merge: counting source meaning lost', GV, "return list->merge(*other_counting_list, voxel_offset, new_meaning);", "return list->merge(*other_counting_list, voxel_offset, std::nullopt);"),
    ('V14 map merge: counting list offset lost (bit source)', GV, "list->merge(*other_bit_list, voxel_offset) : list->merge(*other_counting_list, voxel_offset);", "list->merge(*other_bit_list, {}) : list->merge(*other_counting_list, voxel_offset);"),
    ('V15 map merge: counting list offset lost (counting source)', GV, "list->merge(*other_bit_list, voxel_offset) : list->merge(*other_counting_list, voxel_offset);", "list->merge(*other_bit_list, voxel_offset) : list->merge(*other_counting_list, {});"),
    ('V16 metric: lists truncate', GV, "const bool list_target = is<BitVoxelList>() || is<CountingVoxelList>();", "const bool list_target = false;"),
    ('V17 metric: maps floor', GV, "const bool list_target = is<BitVoxelList>() || is<CountingVoxelList>();", "const bool list_target = true;"),
    ('V18 metric: floor and trunc swapped', GV, "floor_mode ? std::floor(value / side) : std::trunc(value / side)", "floor_mode ? std::trunc(value / side) : std::floor(value / side)"),
    ('V19 metric: y from x', GV, "axis(metric.y)", "axis(metric.x)"),
    ('V20 metric: z from x', GV, "axis(metric.z)", "axis(metric.x)"),
    ('V21 metric: not finite is fine', GV, "!std::isfinite(voxels) || ", ""),
    ('V22 metric: 2^31 is fine', GV, "std::fabs(voxels) >= 2147483648.0f", "std::fabs(voxels) > 2147483648.0f"),
    ('V23 disk: write errors of maps not caught', GV, WRITE_CATCH, WRITE_CATCH.replace("MapFileError", "std::bad_alloc")),
    ('V24 disk: read errors of maps not caught', GV, READ_CATCH, READ_CATCH.replace("MapFileError", "std::bad_alloc")),
    ('V25 disk: failed octree write is success', GV, TREE_WRITE, TREE_WRITE.replace("return false;", "return true;")),
    ('V26 disk: failed octree read is success', GV, TREE_READ, TREE_READ.replace("return false;", "return true;")),
    ('V27 disk: failed map write is success', GV, "e.what());\n\t\t\treturn false;\n\t\t}\n\t\treturn true;\n\t}\n\n\tbool GpuVoxelsMap::read_from_disk", "e.what());\n\t\t\treturn true;\n\t\t}\n\t\treturn true;\n\t}\n\n\tbool GpuVoxelsMap::read_from_disk"),
    ('V28 disk: failed map read is success', GV, "e.what());\n\t\t\treturn false;\n\t\t}\n\t\treturn true;\n\t}\n\n\tuint64_t GpuVoxelsMap::memory_usage_bytes", "e.what());\n\t\t\treturn true;\n\t\t}\n\t\treturn true;\n\t}\n\n\tuint64_t GpuVoxelsMap::memory_usage_bytes"),
    ('U1 bit map upload: size not checked', BM, "if (words.size() != voxel_count() * kWordsPerVoxel)", "if (false)"),
    ('U2 prob. map upload: size not checked', PM, "if (values.size() != voxel_count())", "if (false)"),
    ('U3 prob. map upload: values zero extended', PM, "static_cast<uint32_t>(static_cast<int32_t>(values[i]))", "static_cast<uint32_t>(static_cast<uint8_t>(values[i]))"),
    ('U4 distance map upload: size not checked', DM, "if (words.size() != voxel_count())", "if (false)"),
    ('U5 distance map upload: bit 30 is fine', DM, "if ((word >> 30) != 0)", "if ((word >> 31) != 0)"),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=MergeTest.*:DiskTest.*'],
                               env=env, capture_output=True, text=True, timeout=600)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
