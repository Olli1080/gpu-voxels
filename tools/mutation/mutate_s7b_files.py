# Hand-written mutation runner (slice S7-2, the host readers .xyz / .pcd / .binvox, the load options, the box of voxels, GpuVoxels::insert_point_cloud_from_file / insert_box_into_map): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s7b_files.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
PCF = V + 'src/PointCloudFile.cpp'
GV = V + 'src/GpuVoxels.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

MUT = [
    ('X1 xyz comments are points (equivalent: they are skipped as lines without numbers, only the warning counts them)', PCF, "tokens[0].front() == '#')\n\t\t\t\treturn;", "false)\n\t\t\t\treturn;"),
    ('X2 xyz y and z from swapped columns', PCF, "parse_number(tokens[1], p.y) && parse_number(tokens[2], p.z)", "parse_number(tokens[2], p.y) && parse_number(tokens[1], p.z)"),
    ('X3 xyz plus sign not accepted', PCF, "token.front() == '+'", "token.front() == '#'"),
    ('B1 binvox: step from the depth', PCF, "const float step = scale / static_cast<float>(width);", "const float step = scale / static_cast<float>(depth);"),
    ('B2 binvox: y and z swapped', PCF, "const uint64_t y = i % static_cast<uint64_t>(width);\n\t\t\t\t\tconst uint64_t z = (i / static_cast<uint64_t>(width)) % static_cast<uint64_t>(height);", "const uint64_t z = i % static_cast<uint64_t>(width);\n\t\t\t\t\tconst uint64_t y = (i / static_cast<uint64_t>(width)) % static_cast<uint64_t>(height);"),
    ('B3 binvox: every non-zero value is a point', PCF, "if (value == 1)\n\t\t\t\tfor", "if (value != 0)\n\t\t\t\tfor"),
    ('B4 binvox: a run that ends at the volume is refused', PCF, "if (index + count > size)", "if (index + count >= size)"),
    ('B5 binvox: the line feed after data is data', PCF, "\t\t\t++pos; // the line feed after \"data\"", "\t\t\t; // the line feed after \"data\""),
    ('B6 binvox: no dim line is a size error only', PCF, "if (depth < 0)\n\t\t\tfail(file, \"the header has no dim line\");", ""),
    ('B7 binvox: width 0 accepted', PCF, "depth <= 0 || height <= 0 || width <= 0", "depth <= 0 || height <= 0"),
    ('B8 binvox: height 0 accepted', PCF, "depth <= 0 || height <= 0 || width <= 0", "depth <= 0 || width <= 0"),
    ('B9 binvox: unknown keyword lines are not skipped', PCF, "while (pos < text.size() && text[pos] != '\\n') // unknown keyword: skip the line", "while (false) // unknown keyword: skip the line"),
    ('B10 binvox: translate x not applied', PCF, "step * static_cast<float>(x) + tx", "step * static_cast<float>(x)"),
    ('B11 binvox: translate keyword reads two numbers', PCF, "number(tx);\n\t\t\t\tnumber(ty);\n\t\t\t\tnumber(tz);", "number(tx);\n\t\t\t\tnumber(ty);"),
    ('B12 binvox: truncated data accepted', PCF, "if (pos + 2 > text.size())", "if (false)"),
    ('E1 extension is case sensitive', PCF, "const std::string extension = lower(file.extension().string());", "const std::string extension = file.extension().string();"),
    ('E2 pcd read as xyz', PCF, "if (extension == \".pcd\")\n\t\t\treturn read_pcd_file(file);", "if (extension == \".pcd\")\n\t\t\treturn read_xyz_file(file);"),
    ('S1 shift to the maximum', PCF, "lo[0] = std::min(lo[0], p.x);", "lo[0] = std::max(lo[0], p.x);"),
    ('S2 shift counts points with a NaN x', PCF, "if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))", "if (std::isfinite(p.y) && std::isfinite(p.z))"),
    ('S3 shift of a cloud without finite points', PCF, "if (!any)\n\t\t\treturn;", ""),
    ('S4 shift keeps z', PCF, "p.z -= lo[2];", ""),
    ('L1 offset before the scaling', PCF, "options.scaling * p.x + options.offset.x", "options.scaling * (p.x + options.offset.x)"),
    ('L2 always shifted', PCF, "if (options.shift_to_zero)", "if (true)"),
    ('L3 model path ignored', PCF, "read_point_cloud_file(model_path / file)", "read_point_cloud_file(file)"),
    ('L4 scaling of y missing', PCF, "options.scaling * p.y + options.offset.y", "p.y + options.offset.y"),
    ('K1 box minimum not cut at 0', PCF, "const float cropped = lows[a] >= 0.0f ? lows[a] : 0.0f;", "const float cropped = lows[a];"),
    ('K2 box minimum rounded up', PCF, "std::floor(cropped / voxel_side_length)", "std::ceil(cropped / voxel_side_length)"),
    ('K3 box maximum rounded down', PCF, "std::ceil(highs[a] / voxel_side_length)", "std::floor(highs[a] / voxel_side_length)"),
    ('K4 step rounded down', PCF, "std::round(spacing / voxel_side_length)", "std::floor(spacing / voxel_side_length)"),
    ('K5 step rounded up', PCF, "std::round(spacing / voxel_side_length)", "std::ceil(spacing / voxel_side_length)"),
    ('K6 step below 1 is 0', PCF, "static_cast<int64_t>(step_f)) : 1;", "static_cast<int64_t>(step_f)) : 0;"),
    ('K7 upper voxel excluded', PCF, "count[a] = static_cast<uint64_t>((hi - lo) / step + 1);", "count[a] = static_cast<uint64_t>((hi - lo) / step);"),
    ('K8 start at the limit is inside', PCF, "lo >= int64_t(limits[a]) ? 0 :", "lo > int64_t(limits[a]) ? 0 :"),
    ('K9 limit voxel is inside', PCF, "(int64_t(limits[a]) - lo - 1) / step + 1", "(int64_t(limits[a]) - lo) / step + 1"),
    ('K10 total counts the clipped box', PCF, "box.total = saturating_mul(box.total, count[a]);", "box.total = saturating_mul(box.total, inside[a]);"),
    ('K11 empty box needs a gap of 2', PCF, "if (lo > hi)\n\t\t\t\tthrow", "if (lo > hi + 1)\n\t\t\t\tthrow"),
    ('K12 too many voxels accepted', PCF, "if (generated > kMaxBoxVoxels)", "if (false)"),
    ('K13 corners not checked', PCF, "if (!std::isfinite(lows[a]) || !std::isfinite(highs[a]))", "if (false)"),
    ('K14 side length 0 accepted', PCF, "if (!(voxel_side_length > 0.0f) || !std::isfinite(voxel_side_length))", "if (!(voxel_side_length >= 0.0f) || !std::isfinite(voxel_side_length))"),
    ('K15 infinite spacing accepted', PCF, "if (!(spacing > 0.0f) || !std::isfinite(spacing))", "if (!(spacing > 0.0f))"),
    ('K16 coordinate beyond 2^31 accepted', PCF, "std::fabs(value) > 2147483648.0f", "std::fabs(value) > 1e30f"),
    ('K17 z order outermost', PCF, "for (uint64_t x = 0; x < inside[0]; ++x)\n\t\t\tfor (uint64_t y = 0; y < inside[1]; ++y)\n\t\t\t\tfor (uint64_t z = 0; z < inside[2]; ++z)", "for (uint64_t z = 0; z < inside[2]; ++z)\n\t\t\tfor (uint64_t y = 0; y < inside[1]; ++y)\n\t\t\t\tfor (uint64_t x = 0; x < inside[0]; ++x)"),
    ('G1 file: shift flag lost', GV, "PointCloudLoadOptions{ shift_to_zero, offset, scaling }", "PointCloudLoadOptions{ false, offset, scaling }"),
    ('G2 file: offset lost', GV, "PointCloudLoadOptions{ shift_to_zero, offset, scaling }", "PointCloudLoadOptions{ shift_to_zero, Point3f{ 0, 0, 0 }, scaling }"),
    ('G3 file: scaling lost', GV, "PointCloudLoadOptions{ shift_to_zero, offset, scaling }", "PointCloudLoadOptions{ shift_to_zero, offset, 1.0f }"),
    ('G4 file: meaning changed', GV, "const uint32_t dropped = map->insert_point_cloud(points, voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = dropped;\n\t\treturn true;\n\t}\n\n\tbool GpuVoxels::insert_box_into_map", "const uint32_t dropped = map->insert_point_cloud(points, uint8_t(voxel_meaning + 1));\n\t\tif (outside != nullptr)\n\t\t\t*outside = dropped;\n\t\treturn true;\n\t}\n\n\tbool GpuVoxels::insert_box_into_map"),
    ('G5 file: counter not stored', GV, "const uint32_t dropped = map->insert_point_cloud(points, voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = dropped;\n\t\treturn true;\n\t}\n\n\tbool GpuVoxels::insert_box_into_map", "const uint32_t dropped = map->insert_point_cloud(points, voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = 0;\n\t\treturn true;\n\t}\n\n\tbool GpuVoxels::insert_box_into_map"),
    ('G6 file: file errors not caught', GV, "points = load_point_cloud(file, model_path, PointCloudLoadOptions{ shift_to_zero, offset, scaling });\n\t\t}\n\t\tcatch (const PointCloudFileError& e)", "points = load_point_cloud(file, model_path, PointCloudLoadOptions{ shift_to_zero, offset, scaling });\n\t\t}\n\t\tcatch (const std::bad_alloc& e)"),
    ('G7 file: failure reported as success', GV, "GVL_LOG_ERROR(GpuVoxels, e.what());\n\t\t\treturn false;\n\t\t}\n\t\tconst uint32_t dropped = map->insert_point_cloud(points, voxel_meaning);", "GVL_LOG_ERROR(GpuVoxels, e.what());\n\t\t\treturn true;\n\t\t}\n\t\tconst uint32_t dropped = map->insert_point_cloud(points, voxel_meaning);"),
    ('G8 box: spacing times the points per voxel', GV, "voxel_side_length_ / static_cast<float>(points_per_voxel)", "voxel_side_length_ * static_cast<float>(points_per_voxel)"),
    ('G9 box: points per voxel 0 not refused here (the box refuses it later)', GV, "if (points_per_voxel == 0)\n\t\t\tthrow", "if (false)\n\t\t\tthrow"),
    ('G10 box: meaning changed', GV, "map->insert_coordinate_list(box.coordinates, voxel_meaning)", "map->insert_coordinate_list(box.coordinates, uint8_t(voxel_meaning + 1))"),
    ('G11 box: clipped voxels not counted', GV, "box.total - box.coordinates.size() + dropped", "dropped"),
    ('G12 box: clipped to a bigger limit (equivalent: the class drops and counts them)', GV, "voxel_side_length_, map->dimensions());", "voxel_side_length_, Dim3ui{ 1u << 20, 1u << 20, 1u << 20 });"),
    ('G13 box: the class drops are not added (equivalent: with the clip to its size it drops none)', GV, "box.total - box.coordinates.size() + dropped", "box.total - box.coordinates.size()"),
    ('G14 box: counter not stored', GV, "static_cast<uint32_t>(std::min<uint64_t>(box.total - box.coordinates.size() + dropped, std::numeric_limits<uint32_t>::max()))", "0u"),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=PointCloudFiles.*:FileFacadeTest.*'],
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
