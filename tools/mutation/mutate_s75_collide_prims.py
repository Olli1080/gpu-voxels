# Hand-written mutation runner (slice S7-5, list x map collisions and primitive arrays): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s75_collide_prims.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk. Rebuild before using the executable afterwards.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
SH = V + 'shaders/list_collide.comp'
VL = V + 'src/VoxelList.cpp'
PA = V + 'src/PrimitiveArray.cpp'
GV = V + 'src/GpuVoxels.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

BOUNDS = "if (c.x < 0 || c.y < 0 || c.z < 0 || c.x >= int(pc.dim_x) || c.y >= int(pc.dim_y) || c.z >= int(pc.dim_z))"

MUT = [
    # ---- shader list_collide.comp
    ('C1 shader: x offset ignored', SH, "ivec3(pc.off_x, pc.off_y, pc.off_z)", "ivec3(0, pc.off_y, pc.off_z)"),
    ('C2 shader: y offset ignored', SH, "ivec3(pc.off_x, pc.off_y, pc.off_z)", "ivec3(pc.off_x, 0, pc.off_z)"),
    ('C3 shader: z offset ignored', SH, "ivec3(pc.off_x, pc.off_y, pc.off_z)", "ivec3(pc.off_x, pc.off_y, 0)"),
    ('C4 shader: x = -1 is inside', SH, BOUNDS, BOUNDS.replace("c.x < 0", "c.x < -1")),
    ('C5 shader: y = -1 is inside', SH, BOUNDS, BOUNDS.replace("c.y < 0", "c.y < -1")),
    ('C6 shader: z = -1 is inside', SH, BOUNDS, BOUNDS.replace("c.z < 0", "c.z < -1")),
    ('C7 shader: x = dim is inside', SH, BOUNDS, BOUNDS.replace("c.x >= int", "c.x > int")),
    ('C8 shader: y = dim is inside', SH, BOUNDS, BOUNDS.replace("c.y >= int", "c.y > int")),
    ('C9 shader: z = dim is inside', SH, BOUNDS, BOUNDS.replace("c.z >= int", "c.z > int")),
    ('C10 shader: index uses dim_x for y', SH, "(uint(c.z) * pc.dim_y + uint(c.y)) * pc.dim_x + uint(c.x)", "(uint(c.z) * pc.dim_x + uint(c.y)) * pc.dim_x + uint(c.x)"),
    ('C11 shader: y coordinate read from z', SH, "coords[3u * i + 1u]", "coords[3u * i + 2u]"),
    ('C12 shader: bit map FREE is occupied', SH, "uint map_bits = map_vox[mb] & 0xFFFFFFFEu;", "uint map_bits = map_vox[mb];"),
    ('C13 shader: bit map words 1 .. 7 ignored', SH, "map_bits |= map_vox[mb + w];", "map_bits |= 0u;"),
    ('C14 shader: default prob. threshold exclusive', SH, "int(map_vox[index]) >= pc.level", "int(map_vox[index]) > pc.level"),
    ('C15 shader: types prob. threshold exclusive', SH, "float(int(map_vox[index])) >= pc.threshold", "float(int(map_vox[index])) > pc.threshold"),
    ('C16 shader: types form uses the level', SH, "else if (pc.mode == 0u)\n        {\n            map_occupied", "else if (pc.mode != 2u)\n        {\n            map_occupied"),
    ('C17 shader: default does not look at the list voxel', SH, "if (list_bits == 0u)\n                continue;", ""),
    ('C18 shader: list FREE is occupied', SH, "uint list_bits = list_vox[lb] & 0xFFFFFFFEu;", "uint list_bits = list_vox[lb];"),
    ('C19 shader: list words 1 .. 7 ignored', SH, "list_bits |= list_vox[lb + w];", "list_bits |= 0u;"),
    ('C20 shader: mask test removed', SH, "if (shared_bits == 0u)\n                continue;", ""),
    ('C21 shader: mask word 0 ignored', SH, "uint shared_bits = 0u;\n            for (uint w = 0u;", "uint shared_bits = 0u;\n            for (uint w = 1u;"),
    ('C22 shader: mask only', SH, "shared_bits |= pc.mask[w] & list_vox[lb + w];", "shared_bits |= pc.mask[w];"),
    ('C23 shader: types of all words in word 0', SH, "atomicOr(result[1u + w], list_vox[lb + w]);", "atomicOr(result[1u], list_vox[lb + w]);"),
    ('C24 shader: types overwrite the counter', SH, "atomicOr(result[1u + w], list_vox[lb + w]);", "atomicOr(result[w], list_vox[lb + w]);"),
    ('C25 shader: types of a mask form too (equivalent: the API returns the count only)', SH, "if (pc.mode == 1u)\n        {\n            for", "if (pc.mode != 0u)\n        {\n            for"),
    ('C26 shader: no collision marker', SH, "list_vox[lb + meaning_word(BVM_COLLISION)] |= meaning_mask(BVM_COLLISION);", ""),
    ('C27 shader: counter set instead of added', SH, "atomicAdd(result[0], 1u);", "result[0] = 1u;"),
    ('C28 shader: collision bit added before the types are read', SH, "if (pc.mode == 1u)\n        {\n            for (uint w = 0u; w < WORDS_PER_VOXEL; ++w)\n                if (list_vox[lb + w] != 0u)\n                    atomicOr(result[1u + w], list_vox[lb + w]);\n        }\n        list_vox[lb + meaning_word(BVM_COLLISION)] |= meaning_mask(BVM_COLLISION);",
     "list_vox[lb + meaning_word(BVM_COLLISION)] |= meaning_mask(BVM_COLLISION);\n        if (pc.mode == 1u)\n        {\n            for (uint w = 0u; w < WORDS_PER_VOXEL; ++w)\n                if (list_vox[lb + w] != 0u)\n                    atomicOr(result[1u + w], list_vox[lb + w]);\n        }"),
    # ---- host side of the list
    ('H1 host: collide_with(bit map) in the types mode', VL, "map.dimensions(), false, MapCollisionMode::Default, offset, 0, 0.0f, nullptr).count;", "map.dimensions(), false, MapCollisionMode::Types, offset, 0, 0.0f, nullptr).count;"),
    ('H2 host: collide_with(prob. map) raw level', VL, "ProbVoxelMap::probability_from_float(threshold), 0.0f, nullptr).count;", "static_cast<int32_t>(threshold), 0.0f, nullptr).count;"),
    ('H3 host: types(bit map) in the mask mode', VL, "false, MapCollisionMode::Types, offset, 0, threshold, nullptr);", "false, MapCollisionMode::TypeMask, offset, 0, threshold, nullptr);"),
    ('H4 host: types(prob. map) in the default mode', VL, "true, MapCollisionMode::Types, offset, 0, threshold, nullptr);", "true, MapCollisionMode::Default, offset, 0, threshold, nullptr);"),
    ('H5 host: types(prob. map) threshold ignored', VL, "true, MapCollisionMode::Types, offset, 0, threshold, nullptr);", "true, MapCollisionMode::Types, offset, 0, 0.0f, nullptr);"),
    ('H6 host: mask(bit map) in the types mode', VL, "false, MapCollisionMode::TypeMask, offset, 0, threshold, &types_to_check)", "false, MapCollisionMode::Types, offset, 0, threshold, &types_to_check)"),
    ('H7 host: mask(prob. map) in the types mode', VL, "true, MapCollisionMode::TypeMask, offset, 0, threshold, &types_to_check)", "true, MapCollisionMode::Types, offset, 0, threshold, &types_to_check)"),
    ('H8 host: mask(bit map) without a mask', VL, "false, MapCollisionMode::TypeMask, offset, 0, threshold, &types_to_check)", "false, MapCollisionMode::TypeMask, offset, 0, threshold, nullptr)"),
    ('H9 host: mask(prob. map) treats the map as a bit map', VL, "true, MapCollisionMode::TypeMask, offset, 0, threshold, &types_to_check)", "false, MapCollisionMode::TypeMask, offset, 0, threshold, &types_to_check)"),
    ('H10 host: map of another context accepted', VL, "if (&map.context() != context)", "if (false)"),
    ('H11 host: other dimensions accepted', VL, "if (!(map_dimensions == dimensions_))\n\t\t\tthrow VulkanError(\"VoxelList: the reference dimensions do not match the map dimensions\");\n\t\tconst Entries& mine = impl_->entries;\n\t\tMapCollision result;", "if (false)\n\t\t\tthrow VulkanError(\"VoxelList: the reference dimensions do not match the map dimensions\");\n\t\tconst Entries& mine = impl_->entries;\n\t\tMapCollision result;"),
    ('H12 host: offset x and y swapped', VL, "offset.x, offset.y, offset.z,\n\t\t\tstatic_cast<uint32_t>(mode)", "offset.y, offset.x, offset.z,\n\t\t\tstatic_cast<uint32_t>(mode)"),
    ('H13 host: mask not copied', VL, "std::copy(mask->begin(), mask->end(), push.mask);", "(void)0;"),
    ('H14 host: types not copied back', VL, "std::copy(words.begin() + 1, words.end(), result.types.begin());", ""),
    ('H15 host: count from the first types word', VL, "result.count = words[0];", "result.count = words[1];"),
    ('H16 host: result buffer not cleared', VL, "batch.fill(impl_->collide_result, 0);", ""),
    ('H17 host: an offset beyond the map is not short cut (equivalent: the shader skips every entry)', VL, "|| beyond(offset.x, dimensions_.x) || beyond(offset.y, dimensions_.y) || beyond(offset.z, dimensions_.z))", ")"),
    ('H18 host: collide_with(prob. map) treats the map as a bit map', VL, "map.voxel_buffer(), map.dimensions(), true, MapCollisionMode::Default, offset,", "map.voxel_buffer(), map.dimensions(), false, MapCollisionMode::Default, offset,"),
    # ---- PrimitiveArray.cpp
    ('P1 primitives: metric4 radius not halved', PA, "(p.w / voxel_side_length_) / 2.f", "(p.w / voxel_side_length_)"),
    ('P2 primitives: metric3 radius not halved', PA, "(diameter / voxel_side_length_) / 2.f", "(diameter / voxel_side_length_)"),
    ('P3 primitives: metric4 x not floored', PA, "check_size(p.w, i);\n\t\t\tconverted.push_back({ std::floor(p.x / voxel_side_length_)", "check_size(p.w, i);\n\t\t\tconverted.push_back({ (p.x / voxel_side_length_)"),
    ('P4 primitives: metric3 x not floored', PA, "for (const Point3f& p : metric_points)\n\t\t\tconverted.push_back({ std::floor(p.x / voxel_side_length_)", "for (const Point3f& p : metric_points)\n\t\t\tconverted.push_back({ (p.x / voxel_side_length_)"),
    ('P5 primitives: metric4 z not floored', PA, "std::floor(p.z / voxel_side_length_) + 0.5f, (p.w", "(p.z / voxel_side_length_) + 0.5f, (p.w"),
    ('P6 primitives: metric3 z not floored', PA, "std::floor(p.z / voxel_side_length_) + 0.5f, (diameter", "(p.z / voxel_side_length_) + 0.5f, (diameter"),
    ('P7 primitives: metric4 y not floored', PA, "std::floor(p.y / voxel_side_length_) + 0.5f,\n\t\t\t\tstd::floor(p.z / voxel_side_length_) + 0.5f, (p.w", "(p.y / voxel_side_length_) + 0.5f,\n\t\t\t\tstd::floor(p.z / voxel_side_length_) + 0.5f, (p.w"),
    ('P8 primitives: metric3 y not floored', PA, "std::floor(p.y / voxel_side_length_) + 0.5f,\n\t\t\t\tstd::floor(p.z / voxel_side_length_) + 0.5f, (diameter", "(p.y / voxel_side_length_) + 0.5f,\n\t\t\t\tstd::floor(p.z / voxel_side_length_) + 0.5f, (diameter"),
    ('P9 primitives: metric4 centre without 0.5 in x', PA, "check_size(p.w, i);\n\t\t\tconverted.push_back({ std::floor(p.x / voxel_side_length_) + 0.5f", "check_size(p.w, i);\n\t\t\tconverted.push_back({ std::floor(p.x / voxel_side_length_)"),
    ('P10 primitives: voxel4 z without 0.5', PA, "static_cast<float>(p.z) + 0.5f,\n\t\t\t\tstatic_cast<float>(p.w) / 2.f", "static_cast<float>(p.z),\n\t\t\t\tstatic_cast<float>(p.w) / 2.f"),
    ('P11 primitives: voxel4 size not halved', PA, "static_cast<float>(p.w) / 2.f });", "static_cast<float>(p.w) });"),
    ('P12 primitives: voxel3 size not halved', PA, "static_cast<float>(diameter) / 2.f", "static_cast<float>(diameter)"),
    ('P13 primitives: voxel3 y without 0.5', PA, "static_cast<float>(p.y) + 0.5f, static_cast<float>(p.z) + 0.5f,\n\t\t\t\tstatic_cast<float>(diameter)", "static_cast<float>(p.y), static_cast<float>(p.z) + 0.5f,\n\t\t\t\tstatic_cast<float>(diameter)"),
    ('P14 primitives: metric4 size not checked', PA, "check_size(p.w, i);", ""),
    ('P15 primitives: voxel4 size not checked', PA, "check_size(static_cast<float>(p.w), i);", ""),
    ('P16 primitives: metric3 diameter not checked', PA, "check_diameter(diameter);", ""),
    ('P17 primitives: size zero refused', PA, "if (size < 0.0f)", "if (size <= 0.0f)"),
    ('P18 primitives: diameter zero refused', PA, "if (diameter < 0.0f)", "if (diameter <= 0.0f)"),
    ('P19 primitives: finite check removed', PA, "if (!is_finite(primitives[i]))", "if (false)"),
    ('P20 primitives: x not checked for finite', PA, "std::isfinite(p.x) && ", ""),
    ('P21 primitives: y not checked for finite', PA, "std::isfinite(p.y) && ", ""),
    ('P22 primitives: z not checked for finite', PA, "std::isfinite(p.z) && ", ""),
    ('P23 primitives: radius not checked for finite', PA, " && std::isfinite(p.radius)", ""),
    ('P24 primitives: any type accepted', PA, "if (type != ePRIM_SPHERE && type != ePRIM_CUBOID)", "if (false)"),
    ('P25 primitives: cuboid refused', PA, "if (type != ePRIM_SPHERE && type != ePRIM_CUBOID)", "if (type != ePRIM_SPHERE)"),
    ('P26 primitives: side length zero accepted', PA, "if (!(voxel_side_length > 0.0f) || !std::isfinite(voxel_side_length))", "if (!std::isfinite(voxel_side_length))"),
    ('P27 primitives: infinite side length accepted', PA, "if (!(voxel_side_length > 0.0f) || !std::isfinite(voxel_side_length))", "if (!(voxel_side_length > 0.0f))"),
    ('P28 primitives: content never stored', PA, "primitives_ = std::move(primitives);", ""),
    # ---- the façade
    ('F1 facade: primitives before initialize', GV, "if (!initialized_)\n\t\t\tthrow VulkanError(\"initialize() has to be called before primitive arrays", "if (false)\n\t\t\tthrow VulkanError(\"initialize() has to be called before primitive arrays"),
    ('F2 facade: used primitive name accepted', GV, "if (primitives_.contains(array_name))\n\t\t{\n\t\t\tLOGGING_ERROR_C(VulkanLog, GpuVoxels, \"Primitives array with name '\" << array_name << \"' already exists.\"", "if (false)\n\t\t{\n\t\t\tLOGGING_ERROR_C(VulkanLog, GpuVoxels, \"Primitives array with name '\" << array_name << \"' already exists.\""),
    ('F3 facade: array gets other dimensions', GV, "std::make_shared<PrimitiveArray>(dimensions_, voxel_side_length_, type)", "std::make_shared<PrimitiveArray>(Dim3ui{ 1, 1, 1 }, voxel_side_length_, type)"),
    ('F4 facade: array gets another side length', GV, "std::make_shared<PrimitiveArray>(dimensions_, voxel_side_length_, type)", "std::make_shared<PrimitiveArray>(dimensions_, 1.0f, type)"),
    ('F5 facade: array gets another type', GV, "std::make_shared<PrimitiveArray>(dimensions_, voxel_side_length_, type)", "std::make_shared<PrimitiveArray>(dimensions_, voxel_side_length_, ePRIM_SPHERE)"),
    ('F6 facade: unknown primitive array can be deleted', GV, "if (primitives_.erase(array_name) == 0)", "if (false)"),
    ('F7 facade: modify(metric4) does nothing', GV, "array->set_points(metric_positions);", ""),
    ('F8 facade: modify(voxel4) does nothing', GV, "array->set_points(voxel_positions);", ""),
    ('F9 facade: modify(metric3) does nothing', GV, "array->set_points(metric_positions, diameter);", ""),
    ('F10 facade: modify(voxel3) does nothing', GV, "array->set_points(voxel_positions, diameter);", ""),
    ('F11 facade: visualize primitives succeeds', GV, "\"Primitives Array with name '\" << array_name << \"' not found.\" << endl);\n\t\treturn false;", "\"Primitives Array with name '\" << array_name << \"' not found.\" << endl);\n\t\treturn true;"),
]

TESTS = 'ListMapCollisionTest.*:PrimitiveTest.*:VulkanTest.*List*:ExampleTest.*'
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=' + TESTS],
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
