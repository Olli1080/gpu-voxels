"""Mutation check for slice S6c part 1 (collisions with maps / lists / ids; derived from the S6b runner). Original docstring: S6b (64-bit sort, octree node formats, build/find/extract kernels, Octree host code).
kind 'gpu'  : test_vulkan with the octree filter (fail fast)
kind 'sort' : test_vulkan with the 64-bit sort filter
kind 'fmt'  : test_octree_reference (OctreeFormat tests) first, test_vulkan octree filter when that survives
kind 'cpu'  : test_octree_reference only"""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
V = r'packages\gpu_voxels\gpu_voxels\vulkan'
SH = V + r'\shaders'
NODES = SH + r'\octree_nodes.glsl'
BUILD = SH + r'\octree_build.glsl'
FIND = SH + r'\octree_find.glsl'
EXTRACT = SH + r'\octree_extract.glsl'
HEADS = SH + r'\octree_heads.comp'
COMPACT = SH + r'\octree_compact_ids.comp'
MORTON_IDS = SH + r'\octree_morton_ids.comp'
GID = SH + r'\octree_gid.glsl'
COLLIDE = SH + r'\octree_collide.glsl'
CLEAR = SH + r'\octree_clear_collision.comp'
TREES = SH + r'\octree_collide_trees.glsl'
LB = SH + r'\octree_lb.glsl'
LBT = SH + r'\octree_lb_trees.glsl'
LBE = SH + r'\octree_lb_extract.glsl'
LBC = SH + r'\octree_lb_count.comp'
LBM = SH + r'\octree_lb_move.comp'
HIST64 = SH + r'\radix_histogram64.comp'
SCATTER64 = SH + r'\radix_scatter64.comp'
CPP = V + r'\src\Octree.cpp'
PRIM = V + r'\src\Primitives.cpp'
FORMAT = V + r'\include\gpu_voxels\vulkan\OctreeFormat.h'
REF = r'tests\reference\octree_reference.h'
CPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\Debug\test_octree_reference.exe')
GPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
OCTREE_FILTER = '--gtest_filter=*Balance*:-*LargeTrees*'
SORT_FILTER = '--gtest_filter=*SortPairs64*'
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

mutations = [
    # ---- framework
    ('F1 full stack', 'gpu', LB, 'if (lb_num == 0u || lb_num >= pc.stack_threshold)', 'if (lb_num == 0u || lb_num > pc.stack_threshold)'),
    ('F2 idle threshold', 'gpu', LB, 'atomicAdd(idle_count, 0u) >= pc.idle_threshold ? 1u : 0u', 'atomicAdd(idle_count, 0u) > pc.idle_threshold ? 1u : 0u'),
    ('F3 idle counting', 'gpu', LB, 'if (tid == 0u)\n            atomicAdd(idle_count, 1u);', 'if (tid == 0u)\n            atomicAdd(idle_count, 0u);'),
    ('F4 items per iteration', 'gpu', LB, 'const uint items = min(LB_ITEMS, lb_num);', 'const uint items = LB_ITEMS;'),
    ('F5 stack read', 'gpu', LB, 'lb_cache[w] = stacks[lb_base() + first * LB_WORDS + w];', 'lb_cache[w] = stacks[lb_base() + first * LB_WORDS + w + 1u];'),
    ('F6 pop count', 'gpu', LB, 'lb_num -= items;', 'lb_num -= 1u;'),
    ('F7 scan rounds', 'gpu', LB, 'offset < LB_THREADS; offset <<= 1u', 'offset < 32u; offset <<= 1u'),
    ('F8 exclusive prefix', 'gpu', LB, 'const uint exclusive = lb_scan[tid] - (insert ? 1u : 0u);', 'const uint exclusive = lb_scan[tid];'),
    ('F9 push count', 'gpu', LB, 'lb_num += total;', 'lb_num += total + 1u;'),
    ('F10 stack base', 'gpu', LB, 'return gl_WorkGroupID.x * pc.stack_size * LB_WORDS;', 'return gl_WorkGroupID.x * pc.stack_size;'),
    ('F11 total of the scan', 'gpu', LB, 'total = lb_scan[LB_THREADS - 1u];', 'total = lb_scan[LB_THREADS - 2u];'),
    ('F12 lane count', 'gpu', LB, 'const uint LB_LANES = 8u;', 'const uint LB_LANES = 4u;'),
    # ---- tree vs tree on the framework
    ('T1 item words', 'gpu', LBT, 'packUint2x32(uvec2(lb_cache[first + 2u], lb_cache[first + 3u]))', 'packUint2x32(uvec2(lb_cache[first + 3u], lb_cache[first + 2u]))'),
    ('T2 level of the item', 'gpu', LBT, 'const uint level = (flags >> 8u) & 255u;', 'const uint level = (flags >> 7u) & 255u;'),
    ('T3 active flags', 'gpu', LBT, 'const bool b_active = (flags & 2u) != 0u;', 'const bool b_active = (flags & 1u) != 0u;'),
    ('T4 end here', 'gpu', LBT, 'const bool end_here = pc.min_level >= level;', 'const bool end_here = pc.min_level > level;'),
    ('T5 count condition', 'gpu', LBT, 'if ((!a_part && !b_part) || end_here)', 'if ((!a_part && !b_part) && end_here)'),
    ('T6 counter index', 'gpu', LBT, 'atomicAdd(counters[1u + level], 1u);', 'atomicAdd(counters[level], 1u);'),
    ('T7 marked node', 'gpu', LBT, 'OctInnerBlock(a_active ? a_address : b_address).w[0]', 'OctInnerBlock(a_active ? b_address : a_address).w[0]'),
    ('T8 refine', 'gpu', LBT, 'if (!is_last_level && (a_part || b_part) && !end_here)', 'if (!is_last_level && (a_part && b_part) && !end_here)'),
    ('T9 next level', 'gpu', LBT, '((level - 1u) << 8u);', '((level) << 8u);'),
    ('T10 next address', 'gpu', LBT, 'a_next = a_part ? a.child : a_address;', 'a_next = a.child;'),
    ('T11 leaf stage', 'gpu', LBT, 'if (level == 1u && pc.min_level == 0u && (a_part || b_part))', 'if (level == 2u && pc.min_level == 0u && (a_part || b_part))'),
    ('T12 leaves of b', 'gpu', LBT, 'else if (node_occupied(oct_load_leaf(OctLeafBlock(b.child), j)))', 'else if (true)'),
    ('T13 leaves of a', 'gpu', LBT, '                            if (node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)))\n                            {\n                                atomicAdd(counters[1], 1u);', '                            if (true)\n                            {\n                                atomicAdd(counters[1], 1u);'),
    ('T14 pushed words', 'gpu', LBT, 'stacks[at + 2u] = uint(b_next & 0xFFFFFFFFul);', 'stacks[at + 2u] = uint(a_next & 0xFFFFFFFFul);'),
    ('T15 push position', 'gpu', LBT, 'const uint at = lb_base() + (lb_num + exclusive) * LB_WORDS;\n            stacks[at] = uint(a_next', 'const uint at = lb_base() + (lb_num) * LB_WORDS;\n            stacks[at] = uint(a_next'),
    ('T16 flags of the pushed item', 'gpu', LBT, 'next_flags = (a_part ? 1u : 0u) | (b_part ? 2u : 0u)', 'next_flags = (a_part ? 2u : 0u) | (b_part ? 1u : 0u)'),
    # ---- extract on the framework
    ('E1 level of the item', 'gpu', LBE, 'const uint level = (lb_cache[first + 4u] >> 8u) & 255u;', 'const uint level = (lb_cache[first + 4u] >> 7u) & 255u;'),
    ('E2 id words', 'gpu', LBE, 'packUint2x32(uvec2(lb_cache[first + 2u], lb_cache[first + 3u]))', 'packUint2x32(uvec2(lb_cache[first + 3u], lb_cache[first + 2u]))'),
    ('E3 emit rule', 'gpu', LBE, 'if (selection[node.status] != 0u && (!part || at_min_level))', 'if (selection[node.status] != 0u && (!part))'),
    ('E4 descent rule', 'gpu', LBE, 'if (part && !at_min_level && node.child != 0ul)', 'if (part && node.child != 0ul)'),
    ('E5 last level', 'gpu', LBE, '                if (!last_level)\n                {\n                    insert = true;', '                if (last_level)\n                {\n                    insert = true;'),
    ('E6 child id', 'gpu', LBE, 'next_id = id;', 'next_id = block_id;'),
    ('E7 leaf id', 'gpu', LBE, 'emit(id * 8ul + uint64_t(c), 0u, leaf.status, leaf.occupancy);', 'emit(id * 8ul, 0u, leaf.status, leaf.occupancy);'),
    ('E8 leaf selection', 'gpu', LBE, 'if (selection[leaf.status] != 0u)', 'if (true)'),
    ('E9 pushed level', 'gpu', LBE, 'next_flags = (level - 1u) << 8u;', 'next_flags = level << 8u;'),
    ('E10 pushed address', 'gpu', LBE, 'next_address = node.child;', 'next_address = address;'),
    # ---- balance kernels
    ('C1 level of an item', 'gpu', LBC, 'atomicAdd(level_sums[(stacks[(task * pc.stack_size + i) * WORDS + 4u] >> 8u) & 255u], 1u);', 'atomicAdd(level_sums[(stacks[(task * pc.stack_size + i) * WORDS + 4u] >> 7u) & 255u], 1u);'),
    ('C2 sums layout', 'gpu', LBC, 'item_sums[pc.tasks * (LEVELS - 1u - l) + task] = level_sums[l];', 'item_sums[pc.tasks * l + task] = level_sums[l];'),
    ('C3 offsets from the highest level', 'gpu', LBC, 'for (int l = int(LEVELS) - 1; l >= 0; --l)', 'for (int l = 0; l < int(LEVELS); ++l)'),
    ('C4 inter offsets layout', 'gpu', LBC, 'inter_offsets[task * LEVELS + l] = level_sums[l];', 'inter_offsets[l * pc.tasks + task] = level_sums[l];'),
    ('M1 sums read', 'gpu', LBM, 'level_offsets[l] = sums[(LEVELS - 1u - l) * pc.tasks + task];', 'level_offsets[l] = sums[l * pc.tasks + task];'),
    ('M2 position in the level', 'gpu', LBM, 'const uint offset = level_offsets[level] + (i - level_offsets[LEVELS + level]);', 'const uint offset = level_offsets[level] + i;'),
    ('M3 target stack', 'gpu', LBM, 'const uint target = offset % pc.tasks;', 'const uint target = offset / pc.tasks;'),
    ('M4 target position', 'gpu', LBM, 'const uint position = offset / pc.tasks;', 'const uint position = offset % pc.tasks;'),
    ('M5 overflow test', 'gpu', LBM, 'if (position >= pc.stack_size)', 'if (position > pc.stack_size)'),
    ('M6 new counts', 'gpu', LBM, 'stack_counts[task] = total / pc.tasks + ((total % pc.tasks) >= (task + 1u) ? 1u : 0u);', 'stack_counts[task] = total / pc.tasks + ((total % pc.tasks) > (task + 1u) ? 1u : 0u);'),
    ('M7 total', 'gpu', LBM, 'result[0] = total;', 'result[0] = total + 1u;'),
    ('M8 item copy', 'gpu', LBM, 'for (uint w = 0u; w < WORDS; ++w)\n            stacks_out[out_base + w] = stacks_in[base + w];', 'for (uint w = 0u; w < WORDS - 1u; ++w)\n            stacks_out[out_base + w] = stacks_in[base + w];'),
    # ---- host driver
    ('H1 stack threshold', 'gpu', CPP, 'const uint32_t stack_threshold = stack + kBalanceItems - kBalanceThreads;', 'const uint32_t stack_threshold = stack + kBalanceItems - kBalanceThreads + 1;'),
    ('H2 idle threshold', 'gpu', CPP, 'static_cast<uint32_t>(balancer.idle_fraction * float(tasks)));', 'static_cast<uint32_t>(balancer.idle_fraction * float(tasks)) + 1);'),
    ('H3 first count', 'gpu', CPP, 'zero_counts[0] = 1;', 'zero_counts[0] = 0;'),
    ('H4 swap', 'gpu', CPP, '				std::swap(current, other);\n				if (total > 0 && (total + tasks - 1)', '				if (total > 0 && (total + tasks - 1)'),
    ('H5 stack overflow check', 'gpu', CPP, 'if (read_word(balance_result, 1) != 0)', 'if (false)'),
    ('H6 capacity check', 'gpu', CPP, '(total + tasks - 1) / tasks >= stack_threshold', '(total + tasks - 1) / tasks > stack_threshold'),
    ('H7 scan size', 'gpu', CPP, 'primitives.record_exclusive_scan(batch, item_sums, scanned, tasks * kBalanceLevels + 1, keep_alive);', 'primitives.record_exclusive_scan(batch, item_sums, scanned, tasks * kBalanceLevels, keep_alive);'),
    ('H8 total entry', 'gpu', CPP, 'item_sums.upload(&zero, 4, uint64_t(tasks) * kBalanceLevels * 4);', 'item_sums.upload(&zero, 4, 0);'),
    ('H9 stats', 'gpu', CPP, '				++balancer_stats.balance_steps;\n', ''),
    ('H10 config validation of tasks', 'gpu', CPP, 'config.tasks == 0 || config.tasks > 65535 ||', 'config.tasks > 65535 ||'),
    ('H11 config validation of the stack', 'gpu', CPP, 'config.stack_size < kBalanceThreads ||', 'config.stack_size < 1 ||'),
    ('H12 config validation of the fraction', 'gpu', CPP, '!(config.idle_fraction > 0.0f && config.idle_fraction <= 1.0f)', '!(config.idle_fraction >= 0.0f && config.idle_fraction <= 1.0f)'),
    ('H13 balanced extract start', 'gpu', CPP, '(kRootLevel - 1) << 8, 0u };\n					run_balanced(kernels.lb_extract', '(kRootLevel - 2) << 8, 0u };\n					run_balanced(kernels.lb_extract'),
    ('H14 balanced trees start flags', 'gpu', CPP, '(a_part ? 1u : 0u) | (b_part ? 2u : 0u) | ((kRootLevel - 1) << 8), 0u };', '(a_part ? 1u : 0u) | (b_part ? 2u : 0u) | ((kRootLevel - 2) << 8), 0u };'),
    ('H15 balanced trees params', 'gpu', CPP, '{ options.min_level, options.mark_collisions ? 1u : 0u, uint32_t(options.occupancy_threshold) });', '{ options.min_level, 1u, uint32_t(options.occupancy_threshold) });'),
    ('H16 balanced extract params', 'gpu', CPP, '{ options.min_level, write_mode, capacity });', '{ options.min_level, 1u, capacity });'),
]

only = os.environ.get('ONLY')
done = set()
if os.environ.get('DONE_LOG'):
    for line in open(os.environ['DONE_LOG'], encoding='utf-8'):
        if ': KILLED' in line:
            done.add(line.split(': KILLED')[0])


def read(path):
    return open(path, 'rb').read()


def build(target):
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', target],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    except subprocess.TimeoutExpired:
        return ['TIMEOUT'], 1
    failed = re.findall(r'^\[  FAILED  \] (\S+)', r.stdout, re.M)
    return failed, r.returncode


def evaluate(kind):
    """returns (status, detail): status in KILLED / SURVIVED / BUILD"""
    if kind in ('fmt', 'cpu'):
        if not build('test_octree_reference'):
            return 'BUILD', ''
        failed, code = run([CPU_EXE, '--gtest_fail_fast'])
        if failed or code != 0:
            return 'KILLED', 'cpu: ' + ', '.join(sorted(set(failed))[:2])
        if kind == 'cpu':
            return 'SURVIVED', ''
    if not build('test_vulkan'):
        return 'BUILD', ''
    flt = SORT_FILTER if kind == 'sort' else OCTREE_FILTER
    failed, code = run([GPU_EXE, flt, '--gtest_fail_fast'])
    if failed or code != 0:
        return 'KILLED', ', '.join(sorted(set(failed))[:2]) + (f' (exit {code})' if not failed else '')
    return 'SURVIVED', ''


survivors = []
touched = {}
try:
    for name, kind, rel, old, new in mutations:
        if only and not name.startswith(only):
            continue
        if name in done:
            continue
        path = os.path.join(ROOT, rel)
        original = touched.setdefault(path, read(path))
        text = original.decode('utf-8')
        crlf = '\r\n' in text
        o = old.replace('\n', '\r\n') if crlf else old
        n = new.replace('\n', '\r\n') if crlf else new
        if text.count(o) != 1:
            print(f'{name}: PATTERN {"NOT FOUND" if text.count(o) == 0 else "AMBIGUOUS"} in {rel}', flush=True)
            survivors.append(name + ' (pattern)')
            continue
        open(path, 'wb').write(text.replace(o, n, 1).encode('utf-8'))
        status, detail = evaluate(kind)
        if status == 'KILLED':
            print(f'{name}: KILLED by {detail}', flush=True)
        elif status == 'BUILD':
            print(f'{name}: BUILD FAILED', flush=True)
            survivors.append(name + ' (build)')
        else:
            print(f'{name}: SURVIVED  <-- weak tests', flush=True)
            survivors.append(name)
        open(path, 'wb').write(original)
finally:
    for path, data in touched.items():
        open(path, 'wb').write(data)
    ok1 = build('test_octree_reference')
    ok2 = build('test_vulkan')
    f1, c1 = run([CPU_EXE])
    f2, c2 = run([GPU_EXE, OCTREE_FILTER])
    f3, c3 = run([GPU_EXE, SORT_FILTER])
    print('restored; build ok =', ok1, ok2, '; failing at baseline:', f1, f2, f3, 'exit', c1, c2, c3, flush=True)
print('SURVIVORS:', survivors, flush=True)
