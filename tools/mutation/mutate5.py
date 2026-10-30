"""Mutation check for slice S4 (voxel lists). Every mutation must be killed by the list tests."""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
VK = os.path.join(ROOT, r'packages\gpu_voxels\gpu_voxels\vulkan')
EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))
FILTER = '--gtest_filter=*List*:*Merge*:*Takes*:-*MillionPoints:*MoreThanOneUploadChunk'

S = lambda n: 'shaders\\' + n
mutations = [
    ('U1 merge: non-head group index', S('list_unique_merge.comp'), 'head ? pos[i] : pos[i] - 1u', 'pos[i]'),
    ('U2 merge: bit reduce is xor', S('list_unique_merge.comp'), 'atomicOr(out_vox', 'atomicXor(out_vox'),
    ('U3 merge: counting reduce is or', S('list_unique_merge.comp'), 'atomicAdd(out_vox', 'atomicOr(out_vox'),
    ('U4 normalize: no sign extension', S('list_normalize_counts.comp'), 'uint(((int(vox[i] & 0xFFu) ^ 0x80) - 0x80))', 'vox[i] & 0xFFu'),
    ('U5 heads: first entry is no head', S('list_unique_heads.comp'), '(i == 0u || keys[i] != keys[i - 1u])', '(keys[i] != keys[max(i, 1u) - 1u])'),
    ('U6 heads: inverted', S('list_unique_heads.comp'), '? 1u : 0u', '? 0u : 1u'),
    ('K1 compact: wrong z coordinate', S('list_compact.comp'), 'dst_coords[3u * o + 2u] = src_coords[3u * i + 2u]', 'dst_coords[3u * o + 2u] = src_coords[3u * i + 1u]'),
    ('K2 compact: ignores dst_offset', S('list_compact.comp'), 'pos[i] + pc.dst_offset', 'pos[i]'),
    ('K3 compact: voxel word offset', S('list_compact.comp'), 'dst_vox[o * pc.words + w] = src_vox[i * pc.words + w]', 'dst_vox[o * pc.words + w] = src_vox[i * pc.words + (w + 1u) % pc.words]'),
    ('E1 points: outside points kept', S('list_entries_points.comp'), 'flags[i] = inside ? 1u : 0u;', 'flags[i] = 1u;'),
    ('E2 points: counting starts at 2', S('list_entries_points.comp'), 'out_vox[i] = 1u;', 'out_vox[i] = 2u;'),
    ('E3 voxel_of_point: voxel 0 excluded', S('list_entry.glsl'), 'f.x >= 0.0 &&', 'f.x > 0.0 &&'),
    ('E4 voxel_of_point: voxel -1 accepted', S('list_entry.glsl'), 'f.y >= 0.0 &&', 'f.y >= -1.0 &&'),
    ('E4x voxel_of_point: x voxel -1 accepted', S('list_entry.glsl'), 'f.x >= 0.0 &&', 'f.x >= -1.0 &&'),
    ('E4z voxel_of_point: z voxel -1 accepted', S('list_entry.glsl'), 'f.z >= 0.0 &&', 'f.z >= -1.0 &&'),
    ('E5 voxel_of_point: upper bound inclusive', S('list_entry.glsl'), 'f.z < float(dim.z)', 'f.z <= float(dim.z)'),
    ('E6 voxel_of_point: x upper bound inclusive', S('list_entry.glsl'), 'f.x < float(dim.x)', 'f.x <= float(dim.x)'),
    ('E7 coords: upper bound inclusive', S('list_entries_coords.comp'), 'c.x < dim.x', 'c.x <= dim.x'),
    ('E8 coords: counting starts at 2', S('list_entries_coords.comp'), 'out_vox[i] = 1u;', 'out_vox[i] = 2u;'),
    ('E9 meta: sub cloud boundary', S('list_entries_meta.comp'), 'offsets[mid] <= i', 'offsets[mid] < i'),
    ('E10 meta: outside points kept', S('list_entries_meta.comp'), 'flags[i] = inside ? 1u : 0u;', 'flags[i] = 1u;'),
    ('E11 id: x and y swapped', S('list_entry.glsl'), '(c.z * dim.y + c.y) * dim.x + c.x', '(c.z * dim.x + c.y) * dim.y + c.x'),
    ('M1 membership: upper bound search', S('list_membership.comp'), 'other_ids[mid] < id', 'other_ids[mid] <= id'),
    ('M2 membership: threshold exclusive', S('list_membership.comp'), 'int(this_vox[i]) >= pc.threshold', 'int(this_vox[i]) > pc.threshold'),
    ('M3 membership: nothing is kept', S('list_membership.comp'), 'keep[i] = is_found ? 0u : 1u;', 'keep[i] = 1u;'),
    ('M4 membership: threshold ignored', S('list_membership.comp'), 'pc.use_threshold == 0u ||', 'true ||'),
    ('M5 collide_map: FREE counts as occupied', S('list_collide_map.comp'), 'map_vox[mb] & 0xFFFFFFFEu', 'map_vox[mb]'),
    ('M6 collide_map: no collision marker', S('list_collide_map.comp'), 'list_vox[lb + meaning_word(BVM_COLLISION)] |= meaning_mask(BVM_COLLISION);', ''),
    ('M7 collide_map: counter not incremented twice-safe', S('list_collide_map.comp'), 'atomicAdd(counter, 1u);', 'counter = 1u;'),
    ('F1 count_ge: exclusive', S('list_flags_count_ge.comp'), 'int(vox[i]) >= pc.threshold', 'int(vox[i]) > pc.threshold'),
    ('F2 map_occupied: FREE counts as occupied', S('list_flags_map_occupied.comp'), 'map_vox[base] & 0xFFFFFFFEu', 'map_vox[base]'),
    ('G1 gather: wrong z coordinate', S('list_gather_map.comp'), 'dst_coords[3u * o + 2u] = c.z;', 'dst_coords[3u * o + 2u] = c.y;'),
    ('G2 gather: ignores dst_offset', S('list_gather_map.comp'), 'pos[i] + pc.dst_offset', 'pos[i]'),
    ('G3 gather: payload not copied', S('list_gather_map.comp'), 'dst_vox[o * WORDS_PER_VOXEL + w] = map_vox[i * WORDS_PER_VOXEL + w];', 'dst_vox[o * WORDS_PER_VOXEL + w] = 0u;'),
    ('C1 cpp: sort key bits too small', r'src\VoxelList.cpp', 'std::max<uint32_t>(1, std::bit_width(static_cast<uint32_t>(dims.voxel_count() - 1)))', 'std::max<uint32_t>(1, std::bit_width(static_cast<uint32_t>(dims.voxel_count() - 1)) - 1)'),
    ('C2 cpp: dropped count', r'src\VoxelList.cpp', 'return n - kept;', 'return n;'),
    ('C3 cpp: coordinates of old entries not copied', r'src\VoxelList.cpp', 'batch.copy(entries.coords, combined.coords, static_cast<uint64_t>(entries.count) * 3 * sizeof(uint32_t));', ''),
    ('C4 cpp: counting merged with or', r'src\VoxelList.cpp', 'kind == Kind::Bit ? 0u : 1u', '0u'),
    ('C5 cpp: no normalisation', r'src\VoxelList.cpp', 'if (kind == Kind::Counting)\n\t\t\t{\n\t\t\t\tbatch.barrier();', 'if (false)\n\t\t\t{\n\t\t\t\tbatch.barrier();'),
    ('C6 cpp: remove_counts_below off by one', r'src\VoxelList.cpp', 'CountGePush{ n, static_cast<int32_t>(threshold) }', 'CountGePush{ n, static_cast<int32_t>(threshold) + 1 }'),
    ('C7 cpp: dimension check missing', r'src\VoxelList.cpp', 'if (!(other.reference_dimensions() == owner.reference_dimensions()))\n\t\t\t\tthrow VulkanError("VoxelList: reference dimensions differ");', ''),
    ('C8 cpp: map dimension check missing (collide)', r'src\VoxelList.cpp', 'if (!(map.dimensions() == dimensions_))\n\t\t\tthrow VulkanError("VoxelList: the reference dimensions do not match the map dimensions");\n\t\tconst Entries& mine', 'const Entries& mine'),
    ('C9 cpp: meta kind in push', r'src\VoxelList.cpp', 'EntriesMetaPush push{ dimensions_.x, dimensions_.y, dimensions_.z, n, cloud.num_clouds(),\n\t\t\tstatic_cast<uint32_t>(kind_)', 'EntriesMetaPush push{ dimensions_.x, dimensions_.y, dimensions_.z, n, cloud.num_clouds(),\n\t\t\t0u'),
    ('C10 cpp: threshold flag', r'src\VoxelList.cpp', 'threshold ? 1u : 0u', '0u'),
    ('C11 cpp: unique skips pairs', r'src\VoxelList.cpp', 'if (m <= 1)\n\t\t\t\treturn input;', 'if (m <= 2)\n\t\t\t\treturn input;'),
    ('C12 cpp: occupied voxel count', r'src\VoxelList.cpp', 'impl_->entries = impl_->make_unique(std::move(combined));\n\t\treturn found;', 'impl_->entries = impl_->make_unique(std::move(combined));\n\t\treturn found + 1;'),
    ('C13 map: merge uses collision meaning', r'src\BitVoxelMap.cpp', 'count, eBVM_OCCUPIED };', 'count, eBVM_COLLISION };'),
    ('C14 cpp: subtract keeps the common entries', r'src\VoxelList.cpp', 'impl_->keep_flagged(keep, pos, total_of_scan(keep, pos, mine.count));', 'impl_->keep_flagged(found, pos, total_of_scan(found, pos, mine.count));'),
]

only = os.environ.get('ONLY')


def read(path):
    return open(path, 'rb').read()


def build():
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', 'test_vulkan'],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0


def run_tests():
    r = subprocess.run([EXE, FILTER], capture_output=True, text=True, env=env)
    return re.findall(r'^\[  FAILED  \] VulkanTest\.(\w+)', r.stdout, re.M), r.returncode


survivors = []
touched = {}
try:
    for name, rel, old, new in mutations:
        if only and not name.startswith(only):
            continue
        path = os.path.join(VK, rel)
        original = touched.setdefault(path, read(path))
        text = original.decode('utf-8')
        crlf = '\r\n' in text
        o = old.replace('\n', '\r\n') if crlf else old
        n = new.replace('\n', '\r\n') if crlf else new
        if o not in text:
            print(f'{name}: PATTERN NOT FOUND in {rel}', flush=True)
            survivors.append(name + ' (pattern)')
            continue
        open(path, 'wb').write(text.replace(o, n, 1).encode('utf-8'))
        if not build():
            print(f'{name}: BUILD FAILED', flush=True)
            survivors.append(name + ' (build)')
        else:
            failed, code = run_tests()
            if failed or code != 0:
                print(f'{name}: KILLED by {len(failed)} test(s): {", ".join(sorted(set(failed))[:3])}', flush=True)
            else:
                print(f'{name}: SURVIVED  <-- weak tests', flush=True)
                survivors.append(name)
        open(path, 'wb').write(original)
finally:
    for path, data in touched.items():
        open(path, 'wb').write(data)
    ok = build()
    failed, code = run_tests()
    print('restored; build ok =', ok, '; failing tests at baseline:', failed, 'exit', code, flush=True)
print('SURVIVORS:', survivors, flush=True)
