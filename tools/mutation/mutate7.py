"""Mutation check for slice S3 (ProbVoxelMap). Every mutation must be killed by the prob tests."""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
VK = os.path.join(ROOT, r'packages\gpu_voxels\gpu_voxels\vulkan')
EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))
FILTER = os.environ.get('MUT_FILTER', '--gtest_filter=*Prob*:*Probability*:-*Performance:*SplitsHugeBatches')

S = lambda n: 'shaders\\' + n
CPP = r'src\ProbVoxelMap.cpp'
mutations = [
    ('S1 apply: unknown is -128', S('prob_common.glsl'), 'old_value == PROB_UNKNOWN ? increment : clamp(old_value + increment, PROB_MIN, PROB_MAX)', 'clamp(old_value + increment, PROB_MIN, PROB_MAX)'),
    ('S2 apply: lower clamp -128', S('prob_common.glsl'), 'clamp(old_value + increment, PROB_MIN, PROB_MAX)', 'clamp(old_value + increment, -128, PROB_MAX)'),
    ('S3 apply: upper clamp 126', S('prob_common.glsl'), 'const int PROB_MAX = 127;', 'const int PROB_MAX = 126;'),
    ('S4 target: coordinate bound inclusive', S('prob_ray.glsl'), 'hit.x < dim.x && hit.y < dim.y', 'hit.x <= dim.x && hit.y < dim.y'),
    ('S5 target: coordinates y/z swapped', S('prob_ray.glsl'), 'hit = uvec3(raw[3u * i], raw[3u * i + 1u], raw[3u * i + 2u]);', 'hit = uvec3(raw[3u * i], raw[3u * i + 2u], raw[3u * i + 1u]);'),
    ('S6 robot: wrong word of the bit', S('prob_ray.glsl'), '(pc.robot_bit >> 5u)', '(pc.robot_bit >> 4u)'),
    ('S7 robot: inverted cut', S('prob_ray.glsl'), '((robot_vox[word] >> (pc.robot_bit & 31u)) & 1u) != 0u', '((robot_vox[word] >> (pc.robot_bit & 31u)) & 1u) == 0u'),
    ('S8 origin not required', S('prob_ray.glsl'), 'pc.require_origin != 0u &&', 'false &&'),
    ('S9 ray: one voxel short', S('prob_ray.glsl'), 'int n = 1 + d.x + d.y + d.z;', 'int n = d.x + d.y + d.z;'),
    ('S10 ray: x branch condition', S('prob_ray.glsl'), 'if (error_xy > 0 && error_xz > 0)', 'if (error_xy >= 0 && error_xz > 0)'),
    ('S11 ray: y branch error update', S('prob_ray.glsl'), 'error_xy += dd.x;\n                error_yz -= dd.z;', 'error_xy += dd.x;\n                error_yz += dd.z;'),
    ('S12 ray: z branch error update', S('prob_ray.glsl'), 'error_xz += dd.x;\n                error_yz += dd.y;', 'error_xz += dd.x;\n                error_yz -= dd.y;'),
    ('S13 ray: x step uses y error', S('prob_ray.glsl'), 'error_xy -= dd.y;\n                error_xz -= dd.z;', 'error_xy -= dd.z;\n                error_xz -= dd.z;'),
    ('S14 ray: free update on the hit voxel only', S('prob_ray.glsl'), 'prob_emit(voxel_id(uvec3(p), dim), pc.free_inc);', 'prob_emit(voxel_id(uvec3(p), dim), pc.hit_inc);'),
    ('S15 ray: hit update lost', S('prob_ray.glsl'), 'prob_emit(voxel_id(hit, dim), pc.hit_inc);\n}', '}'),
    ('A1 apply: unknown assigns zero', S('prob_points_apply.comp'), 'atomicExchange(voxels[id], PROB_UNKNOWN_WORD);', 'atomicExchange(voxels[id], 0u);'),
    ('A2 accumulate: sum replaced by store', S('prob_points_apply.comp'), 'atomicAdd(acc[id], increment);', 'acc[id] = increment;'),
    ('A3 accumulate: touched flag missing', S('prob_points_apply.comp'), 'atomicOr(touched[id >> 5u], 1u << (id & 31u));', ''),
    ('A4 atomic: plain read-modify-write', S('prob_points_apply.comp'), 'uint old_word = voxels[id];\n    for (;;)', 'uint old_word = voxels[id];\n    voxels[id] = uint(prob_apply(int(old_word), increment));\n    for (;false;)'),
    ('A5 accumulate mode never selected', S('prob_points_apply.comp'), 'pc.mode == 1u', 'pc.mode == 2u'),
    ('A6 counters: outside not counted', S('prob_points_apply.comp'), 'void prob_skip_outside() { atomicAdd(counters[0], 1u); }', 'void prob_skip_outside() { }'),
    ('A7 counters: cut not counted', S('prob_points_apply.comp'), 'void prob_skip_cut() { atomicAdd(counters[1], 1u); }', 'void prob_skip_cut() { }'),
    ('P1 accumulated: unknown base -128', S('prob_apply_accumulated.comp'), 'old_value == PROB_UNKNOWN ? 0 : old_value', 'old_value'),
    ('P2 accumulated: accumulator not reset', S('prob_apply_accumulated.comp'), '        acc[i] = 0;\n', ''),
    ('P3 accumulated: touched not cleared', S('prob_apply_accumulated.comp'), '        atomicAnd(touched[word], ~bit);\n', ''),
    ('P4 accumulated: no clamp', S('prob_apply_accumulated.comp'), 'clamp(base + acc[i], PROB_MIN, PROB_MAX)', 'base + acc[i]'),
    ('F1 fold: replays only the first update', S('prob_events_fold.comp'), 'for (uint j = i; j < pc.n && keys[j] == id; ++j)', 'for (uint j = i; j < i + 1u; ++j)'),
    ('F2 fold: every entry starts a replay', S('prob_events_fold.comp'), 'if (i != 0u && keys[i] == keys[i - 1u])\n            continue;', ''),
    ('F3 fold: value order reversed', S('prob_events_fold.comp'), 'value = prob_apply(value, int(vals[j]));', 'value = prob_apply(value, int(vals[pc.n - 1u - (j - i)]));'),
    ('W1 write: chunk base ignored', S('prob_events_write.comp'), 'offsets[i - pc.batch_first] - pc.base', 'offsets[i - pc.batch_first]'),
    ('W2 write: increment sign lost', S('prob_events_write.comp'), 'vals[g_position] = uint(increment);', 'vals[g_position] = uint(abs(increment));'),
    ('W3 write: id off by one', S('prob_events_write.comp'), 'keys[g_position] = id;', 'keys[g_position] = id ^ 1u;'),
    ('N1 count: one event too many', S('prob_events_count.comp'), 'counts[k] = g_events;', 'counts[k] = g_events + 1u;'),
    ('N2 count: skipped not counted', S('prob_events_count.comp'), 'void prob_skip_outside() { atomicAdd(counters[0], 1u); }', 'void prob_skip_outside() { }'),
    ('L1 collide: threshold exclusive', S('prob_collide.comp'), 'value < pc.threshold_this', 'value <= pc.threshold_this'),
    ('L2 collide: FREE bit counts as occupied', S('prob_collide.comp'), 'other[i * WORDS_PER_VOXEL] & 0xFFFFFFFEu', 'other[i * WORDS_PER_VOXEL]'),
    ('L3 collide: fixed marker increment', S('prob_collide.comp'), 'prob_apply(value, pc.mark_increment)', 'prob_apply(value, 72)'),
    ('L4 collide: other threshold ignored', S('prob_collide.comp'), 'int(other[i]) >= pc.threshold_other', 'true'),
    ('L5 collide: no marker', S('prob_collide.comp'), 'voxels[i] = uint(prob_apply(value, pc.mark_increment));', ''),
    ('B1 bit collide: threshold exclusive', S('bit_voxel_collide_prob.comp'), 'int(other[i]) < pc.threshold', 'int(other[i]) <= pc.threshold'),
    ('B2 bit collide: no marker', S('bit_voxel_collide_prob.comp'), 'voxels[base + meaning_word(BVM_COLLISION)] |= meaning_mask(BVM_COLLISION);', ''),
    ('B3 bit collide: FREE counts', S('bit_voxel_collide_prob.comp'), 'voxels[base] & 0xFFFFFFFEu', 'voxels[base]'),
    ('C1 cpp: float conversion order', CPP, '(std::max)((std::min)(1.0f, value), 0.0f)', '(std::max)((std::min)(value, 1.0f), 0.0f)'),
    ('C2 cpp: swept volume offset', CPP, 'static_cast<int32_t>(meaning) - eBVM_UNCERTAIN_OCC_PROB', 'static_cast<int32_t>(meaning) - 128'),
    ('C3 cpp: collision meaning is a swept volume', CPP, '\t\t\tcase eBVM_COLLISION: return', '\t\t\tcase 254: return'),
    ('C4 cpp: accumulate flag in push', CPP, 'params.free_inc, params.hit_inc, accumulate ? 1u : 0u, params.assign_unknown);', 'params.free_inc, params.hit_inc, 0u, params.assign_unknown);'),
    ('C5 cpp: unknown goes through the event path', CPP, 'if (options.mode == UpdateMode::Sequential && !params.assign_unknown)', 'if (options.mode == UpdateMode::Sequential)'),
    ('C6 cpp: origin not required', CPP, 'params.require_origin = true;', 'params.require_origin = false;'),
    ('C7 cpp: robot dimensions unchecked', CPP, 'if (!(robot_map->dimensions() == dimensions_))\n\t\t\t\tthrow VulkanError("ProbVoxelMap::insert_sensor_data: the robot map must have the same dimensions");', ''),
    ('C8 cpp: robot bit fixed', CPP, 'params.robot_bit = robot_voxel_meaning;', 'params.robot_bit = eBVM_OCCUPIED;'),
    ('C9 cpp: collide other kind', CPP, 'other_is_bit ? 1u : 0u, level, level', '0u, level, level'),
    ('C10 cpp: meta batches not grouped', CPP, 'update_of_meaning(meanings[end], options.model) == update', 'false'),
    ('C11 cpp: clear meaning condition', CPP, 'if (meaning != eBVM_OCCUPIED)\n\t\t\tthrow', 'if (meaning == eBVM_OCCUPIED)\n\t\t\tthrow'),
    ('C12 cpp: clear value', CPP, 'voxels_.fill(kUnknownWord);', 'voxels_.fill(0);'),
    ('C13 cpp: download drops the sign', CPP, 'static_cast<int8_t>(words[i] & 0xFFu)', 'static_cast<int8_t>(words[i] & 0x7Fu)'),
    ('C14 cpp: sort key bits too small', CPP, 'std::max<uint32_t>(1, std::bit_width(static_cast<uint32_t>(voxel_count() - 1)))', 'std::max<uint32_t>(1, std::bit_width(static_cast<uint32_t>(voxel_count() - 1)) - 1)'),
    ('C15 cpp: chunk base of the events', CPP, 'write_push.base = static_cast<uint32_t>(base);', 'write_push.base = 0;'),
    ('C16 cpp: chunk start measurement', CPP, 'write_push.first = source.first + a;', 'write_push.first = source.first;'),
    ('C19 cpp: chunk limit ignored (total)', CPP, 'if (total > max_events)', 'if (total > kMaxSequentialEvents)'),
    ('C20 cpp: chunk limit ignored (search)', CPP, 'if (offset_of(mid) - offset_of(a) <= max_events)', 'if (offset_of(mid) - offset_of(a) <= kMaxSequentialEvents)'),
    ('C21 cpp: single ray limit ignored', CPP, 'if (offset_of(a + 1) - offset_of(a) > max_events)', 'if (false)'),
    ('C17 cpp: model not validated for hits', CPP, 'model.free_update < ProbVoxelMap::kMinProbability || model.occupied_update < ProbVoxelMap::kMinProbability', 'model.free_update < ProbVoxelMap::kMinProbability'),
    ('C18 bit map: threshold not converted', r'src\BitVoxelMap.cpp', 'ProbVoxelMap::probability_from_float(threshold) };', 'static_cast<int32_t>(threshold) };'),
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
    return re.findall(r'^\[  FAILED  \] (?:VulkanTest|RayWalk)\.(\w+)', r.stdout, re.M), r.returncode


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
