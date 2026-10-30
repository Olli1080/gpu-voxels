"""Mutation check for slice S5 (DistanceVoxelMap: exact EDT and PBA). Every mutation must be killed by the distance tests."""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
VK = os.path.join(ROOT, r'packages\gpu_voxels\gpu_voxels\vulkan')
EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))
FILTER = os.environ.get('MUT_FILTER', '--gtest_filter=*Distance*:-*Performance:*LargeMap*')

S = lambda n: 'shaders\\' + n
CPP = r'src\DistanceVoxelMap.cpp'
mutations = [
    ('X1 inf too small', S('dist_common.glsl'), 'const int DIST_INF = 1 << 26;', 'const int DIST_INF = 1 << 8;'),
    ('I1 insert: bound inclusive', S('dist_insert.comp'), 'inside = c.x < dim.x && c.y < dim.y && c.z < dim.z;', 'inside = c.x <= dim.x && c.y < dim.y && c.z < dim.z;'),
    ('I2 insert: wrong stored value', S('dist_insert.comp'), 'voxels[voxel_id(c, dim)] = dist_pack(c);', 'voxels[voxel_id(c, dim)] = dist_pack(c) ^ 1u;'),
    ('I3 insert: outside not counted', S('dist_insert.comp'), 'atomicAdd(counter, 1u);', 'counter = 0u;'),
    ('I4 insert: input kind inverted', S('dist_insert.comp'), 'if (pc.input_kind == 0u)', 'if (pc.input_kind != 0u)'),
    ('M1 merge: threshold inclusive', S('dist_merge_occupied.comp'), 'int(prob[i]) > pc.level', 'int(prob[i]) >= pc.level'),
    ('M2 merge: wrong coordinates', S('dist_merge_occupied.comp'), 'dist_pack(coords_of_id(i, dim))', 'dist_pack(coords_of_id(i, uvec3(dim.y, dim.x, dim.z)))'),
    ('Z1 z pass: cost not squared', S('dist_edt_z.comp'), 'out_cost[id] = last < 0 ? DIST_INF : (int(z) - last) * (int(z) - last);', 'out_cost[id] = last < 0 ? DIST_INF : abs(int(z) - last);'),
    ('Z2 z pass: stale sites are obstacles (forward)', S('dist_edt_z.comp'), 'if (map_voxels[id] == dist_pack(uvec3(x, y, z)))', 'if (map_voxels[id] != DIST_UNINIT)'),
    ('Z3 z pass: stale sites are obstacles (backward)', S('dist_edt_z.comp'), 'if (map_voxels[id] == dist_pack(uvec3(x, y, uint(z))))', 'if (map_voxels[id] != DIST_UNINIT)'),
    ('Z4 z pass: wrong backward site', S('dist_edt_z.comp'), 'out_site[id] = dist_pack(uvec3(x, y, uint(next)));', 'out_site[id] = dist_pack(uvec3(x, y, uint(z)));'),
    ('Z5 z pass: no backward sweep', S('dist_edt_z.comp'), 'if (next >= 0 && (next - z) * (next - z) < out_cost[id])', 'if (next >= 0 && false)'),
    ('L1 line: wrong separation', S('dist_edt_line.comp'), 'cost_at(u) - cost_at(i)) / (2 * (u - i));', 'cost_at(u) - cost_at(i)) / (2 * (u - i) + 1);'),
    ('L3 line: envelope entry limit', S('dist_edt_line.comp'), 'if (w < m)', 'if (w < m - 1)'),
    ('L4 line: y step is 1', S('dist_edt_line.comp'), 'g_step = dim.x;', 'g_step = 1u;'),
    ('L5 line: y line mapping', S('dist_edt_line.comp'), 'g_base = voxel_id(uvec3(line % dim.x, 0u, line / dim.x), dim);', 'g_base = voxel_id(uvec3(line / dim.x, 0u, line % dim.x), dim);'),
    ('L6 line: line length', S('dist_edt_line.comp'), 'const int m = int(pc.axis == 0u ? dim.x : dim.y);', 'const int m = int(pc.axis == 0u ? dim.y : dim.x);'),
    ('L7 line: envelope pop', S('dist_edt_line.comp'), 'if (u == env_start[q])', 'if (u == env_start[q] + 1)'),
    ('L8 line: envelope start', S('dist_edt_line.comp'), 'const int w = 1 + separation(env_source[q], u);', 'const int w = separation(env_source[q], u);'),
    ('E1 extract: cap boundary', S('dist_extract.comp'), 'squared >= 128 * 128 ? 127 : isqrt(squared)', 'squared >= 129 * 129 ? 127 : isqrt(squared)'),
    ('E2 extract: upper border value', S('dist_extract.comp'), 'result = max(dims.x - pos.x, 0);', 'result = max(dims.x - pos.x - 1, 0);'),
    ('E3 extract: lower border inclusive', S('dist_extract.comp'), 'else if (pos.x < pc.radius) result = max(pos.x, 0);', 'else if (pos.x <= pc.radius) result = max(pos.x, 0);'),
    ('E4 extract: y border radius', S('dist_extract.comp'), 'pos.y > dims.y - pc.radius', 'pos.y > dims.y - pc.radius - 1'),
    ('E5 extract: upper border inclusive', S('dist_extract.comp'), 'if (pos.x > dims.x - pc.radius)', 'if (pos.x >= dims.x - pc.radius)'),
    ('Q1 query: wrong index', S('dist_query.comp'), 'coords_of_id(indices[i], dim)', 'coords_of_id(i, dim)'),
    ('PS1 pba sanitize: nothing reset', S('pba_sanitize.comp'), 'if (map_voxels[i] != dist_pack(coords_of_id(i, dim)))', 'if (false)'),
    ('PS2 pba sanitize: reset value', S('pba_sanitize.comp'), 'map_voxels[i] = DIST_UNINIT;', 'map_voxels[i] = 0u;'),
    ('PF1 flood: band top', S('pba_p1_flood.comp'), 'const int tz_top = tz_bottom + band_size - 1;', 'const int tz_top = tz_bottom + band_size;'),
    ('PF2 flood: uninitialised distance of the running obstacle', S('pba_p1_flood.comp'), 'obstacle.z == PBA_U ? 0x7FFFFFFF : abs(obstacle.z - (tz_top - i))', 'abs(obstacle.z - (tz_top - i))'),
    ('PF3 flood: forward result not stored', S('pba_p1_flood.comp'), 'obstacle = current;\n            data[id] = pba_store(obstacle);', 'obstacle = current;'),
    ('PP1 propagate: wrong target', S('pba_p1_propagate.comp'), 'initial[top_id] = pba_store(current);', 'initial[bottom_id] = pba_store(current);'),
    ('PP2 propagate: comparison', S('pba_p1_propagate.comp'), 'if (abs(current.z - tz_bottom) < current_dist)', 'if (abs(current.z - tz_bottom) > current_dist)'),
    ('PP3 propagate: uninitialised distance', S('pba_p1_propagate.comp'), 'int current_dist = current.z == PBA_U ? 0x7FFFFFFF : abs(current.z - tz_bottom);', 'int current_dist = abs(current.z - tz_bottom);'),
    ('PU1 update: uninitialised left used', S('pba_p1_update.comp'), 'if (left.z != PBA_U)', 'if (true)'),
    ('PU2 update: uninitialised current distance', S('pba_p1_update.comp'), 'int min_dist = current.z == PBA_U ? 0x7FFFFFFF : abs(current.z - (tz_bottom + i));', 'int min_dist = abs(current.z - (tz_bottom + i));'),
    ('PU3 update: right distance', S('pba_p1_update.comp'), 'const int d = abs(right.z - (tz_bottom + i));', 'const int d = abs(right.z - tz_bottom);'),
    ('B1 backpointers: tail pointer loses its link', S('pba_p2_backpointers.comp'), 'pba_store(ivec3(PBA_U, last_y, PBA_U));', 'pba_store(ivec3(PBA_U, PBA_U, PBA_U));'),
    ('B2 backpointers: last y', S('pba_p2_backpointers.comp'), 'last_y = ty;', 'last_y = ty + 1;'),
    ('B3 backpointers: envelope test', S('pba_p2_backpointers.comp'), 'if (intersection1 < intersection2)', 'if (intersection1 > intersection2)'),
    ('B4 intersection formula', S('pba_common.glsl'), '- float(z0))) / ny;', '- float(z0))) / (2.0 * ny);'),
    ('FW1 forward: top voxel obstacle', S('pba_p2_forward.comp'), 'next_y = ty_top;', 'next_y = ty_top - 1;'),
    ('FW2 forward: head pointer sign', S('pba_p2_forward.comp'), 'forward_ptrs[top_id - (band_size - 1) * row] = -last_y;', 'forward_ptrs[top_id - (band_size - 1) * row] = last_y;'),
    ('FW3 forward: pointer value', S('pba_p2_forward.comp'), 'forward_ptrs[id] = last_y;', 'forward_ptrs[id] = ty;'),
    ('MG1 merge: kept counter', S('pba_p2_merge.comp'), 'band2_kept = max(1, band2_kept + 1);', 'band2_kept = band2_kept + 1;'),
    ('MG2 merge: kept counter not decreased', S('pba_p2_merge.comp'), '                --band2_kept;\n', ''),
    ('MG3 merge: tail pointer', S('pba_p2_merge.comp'), 'voxel_end_band2.y = ty_end_band1;', 'voxel_end_band2.y = ty_end_band1 - 1;'),
    ('MG4 merge: head pointer sign', S('pba_p2_merge.comp'), 'forward_ptrs[tz * layer + ty_start_band1 * row + tx] = -ty_start_band2;', 'forward_ptrs[tz * layer + ty_start_band1 * row + tx] = ty_start_band2;'),
    ('MG5 merge: kept limit', S('pba_p2_merge.comp'), 'while (band2_kept < 2 && y_next_band2 != PBA_U)', 'while (band2_kept < 1 && y_next_band2 != PBA_U)'),
    ('MG6 merge: back pointer of band 2', S('pba_p2_merge.comp'), 'stack_voxels[id_next_band2] = pba_store(ivec3(voxel_next_band2.x, y_last_band1, voxel_next_band2.z));', 'stack_voxels[id_next_band2] = pba_store(ivec3(voxel_next_band2.x, PBA_U, voxel_next_band2.z));'),
    ('D1 distances: envelope walk', S('pba_p3_distances.comp'), 'if (float(ty) > intersection)', 'if (float(ty) < intersection)'),
    ('D2 distances: one round short', S('pba_p3_distances.comp'), 'for (int k = 0; k < rounds; ++k)', 'for (int k = 0; k < rounds - 1; ++k)'),
    ('D3 distances: wrong publishing row', S('pba_p3_distances.comp'), 'if (band_id == band_count - 1) // smallest y of the round: the most advanced state', 'if (band_id == 0) // smallest y of the round: the most advanced state'),
    ('D4 distances: barrier before publishing removed', S('pba_p3_distances.comp'), 'barrier(); // everybody has read the shared state of this round', ''),
    ('D5 distances: shared top not published', S('pba_p3_distances.comp'), '                s_stack_top[lane] = stack_top;\n                s_last_y[lane] = last_y;', '                s_last_y[lane] = last_y;'),
    ('T1 transpose: coordinates not swapped', S('pba_transpose.comp'), 'dist_pack(uvec3(o.y, o.x, o.z))', 'dist_pack(o)'),
    ('T2 transpose: index', S('pba_transpose.comp'), 'out_voxels[(z * pc.nx + x) * pc.ny + y]', 'out_voxels[(z * pc.nx + y) * pc.ny + x]'),
    ('C2 cpp: y line count', CPP, 'const uint32_t y_lines = dimensions_.x * dimensions_.z;', 'const uint32_t y_lines = dimensions_.x * dimensions_.y;'),
    ('C3 cpp: PBA never selected', CPP, 'if (algorithm == DistanceAlgorithm::ParallelBanding)\n\t\t\tcompute_parallel_banding(pba);', 'if (false)\n\t\t\tcompute_parallel_banding(pba);'),
    ('C4 cpp: one layer per band allowed', CPP, 'if (options.m1 > 1 && dz / options.m1 < 2)', 'if (false)'),
    ('C5 cpp: invocation cap removed', CPP, 'options.m3 == 0 || options.m3 > 16 ||', 'options.m3 == 0 ||'),
    ('C6 cpp: merge threshold conversion', CPP, 'ProbVoxelMap::probability_from_float(occupancy_threshold) };', 'static_cast<int32_t>(occupancy_threshold) };'),
    ('C7 cpp: coordinate input kind', CPP, 'insert_raw(buffer, static_cast<uint32_t>(chunk.size()), 1)', 'insert_raw(buffer, static_cast<uint32_t>(chunk.size()), 0)'),
    ('C8 cpp: meta meaning not checked', CPP, 'if (meaning != eBVM_OCCUPIED)\n\t\t\tthrow VulkanError("DistanceVoxelMap: only eBVM_OCCUPIED can be inserted");\n\t\tif (&cloud.context() != context_)', 'if (&cloud.context() != context_)'),
    ('C9 cpp: rounded root', CPP, 'return static_cast<int32_t>(std::sqrt(static_cast<double>(squared)));', 'return static_cast<int32_t>(std::round(std::sqrt(static_cast<double>(squared))));'),
    ('C10 cpp: band merging stops early', CPP, 'for (uint32_t bands = m2; bands > 1; bands /= 2)', 'for (uint32_t bands = m2; bands > 2; bands /= 2)'),
    ('C11 cpp: second pass dimensions', CPP, 'pass_along_y(transposed, transposed, dy, dx, dz);', 'pass_along_y(transposed, transposed, dx, dy, dz);'),
    ('C12 cpp: phase 3 column groups', CPP, '(nx + 15) / 16, 1, nz', '(nx + 8) / 16, 1, nz'),
    ('C13 cpp: flood threads', CPP, 'const uint32_t flood_threads = dx * dy * options.m1;', 'const uint32_t flood_threads = dx * dy;'),
    ('C14 cpp: transpose back dimensions', CPP, 'PbaVolumePush{ dy, dx, dz, count });\n\t\tbatch.submit_and_wait();', 'PbaVolumePush{ dx, dy, dz, count });\n\t\tbatch.submit_and_wait();'),
    ('C15 cpp: radius ignored', CPP, 'static_cast<uint32_t>(voxel_count()), robot_radius });', 'static_cast<uint32_t>(voxel_count()), 0 });'),
    ('C16 cpp: position check removed', CPP, 'if (position.x >= dimensions_.x || position.y >= dimensions_.y || position.z >= dimensions_.z)\n\t\t\tthrow VulkanError("DistanceVoxelMap: position outside the map");', ''),
]

only = os.environ.get('ONLY')
done = set()
if os.environ.get('DONE_LOG'):
    for line in open(os.environ['DONE_LOG'], encoding='utf-8'):
        if ': KILLED' in line:
            done.add(line.split(': KILLED')[0])


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
        if name in done:
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
