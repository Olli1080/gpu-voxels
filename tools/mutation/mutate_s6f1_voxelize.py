# Hand-written mutation runner (slice S6f-1, Octree::voxelize_points: pose, voxel grid, distinct voxels with counts and occupancy): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6f1_voxelize.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
SENSOR = V + 'shaders/octree_sensor_voxels.comp'
STARTS = V + 'shaders/octree_run_starts.comp'
COUNTS = V + 'shaders/octree_run_counts.comp'
CPP = V + 'src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')

SORT_BLOCK = "batch.barrier();\n\t\t\t\tif (n > 1)\n\t\t\t\t\tprimitives.record_sort_pairs64(batch, ids, values, n, 43, keep_alive); // sentinel 2^42 needs 43 bits\n\t\t\t\tbatch.submit_and_wait();\n\t\t\t}\n\t\t\truns.outside"

MUT = [
    ('V1 translation of x ignored', SENSOR, "dot(pc.row0.xyz, p) + pc.row0.w", "dot(pc.row0.xyz, p)"),
    ('V2 y takes the rotation of z', SENSOR, "dot(pc.row1.xyz, p) + pc.row1.w", "dot(pc.row2.xyz, p) + pc.row1.w"),
    ('V3 z takes the rotation of x', SENSOR, "dot(pc.row2.xyz, p) + pc.row2.w", "dot(pc.row0.xyz, p) + pc.row2.w"),
    ('V4 trunc instead of floor', SENSOR, "floor(world / pc.voxel_side_length)", "trunc(world / pc.voxel_side_length)"),
    ('V5 x limit inclusive', SENSOR, "f.x < 16384.0", "f.x <= 16384.0"),
    ('V6 y limit inclusive', SENSOR, "f.y < 16384.0", "f.y <= 16384.0"),
    ('V7 z limit inclusive', SENSOR, "f.z < 16384.0", "f.z <= 16384.0"),
    ('V8 x zero excluded', SENSOR, "f.x >= 0.0", "f.x > 0.0"),
    ('V9 y zero excluded', SENSOR, "f.y >= 0.0", "f.y > 0.0"),
    ('V10 z zero excluded', SENSOR, "f.z >= 0.0", "f.z > 0.0"),
    ('V11 dropped points not counted', SENSOR, "atomicAdd(dropped[0], 1u);", "atomicAdd(dropped[0], 0u);"),
    ('V12 multiply by the side length', SENSOR, "world / pc.voxel_side_length", "world * pc.voxel_side_length"),
    ('V13 sentinel inside the valid ids', SENSOR, "1ul << 42", "1ul << 41"),
    ('V14 high word of the id', SENSOR, "ids[2u * i + 1u] = uint(id >> 32);", "ids[2u * i + 1u] = uint(id >> 31);"),
    ('R1 heads inverted', STARTS, "heads[i] == 0u", "heads[i] != 0u"),
    ('R2 start of a run off by one', STARTS, "starts[offsets[i]] = i;", "starts[offsets[i]] = i + 1u;"),
    ('C1 last run one too short', COUNTS, ": pc.n;", ": pc.n - 1u;"),
    ('C2 last run reads past the starts', COUNTS, "(k + 1u < m)", "(k + 1u <= m)"),
    ('C3 count limit 254', COUNTS, "min(count, 255u)", "min(count, 254u)"),
    ('C4 lower clamp -128', COUNTS, "int(min(count, 255u)), -127, 127)", "int(min(count, 255u)), -128, 127)"),
    ('C5 occupancy without the count', COUNTS, "pc.update_probability * int(min(count, 255u))", "pc.update_probability"),
    ('C6 number of runs without the last head', COUNTS, "offsets[pc.n - 1u] + heads[pc.n - 1u]", "offsets[pc.n - 1u]"),
    ('C7 occupancy without the initial value', COUNTS, "clamp(pc.initial_probability + ", "clamp("),
    ('P1 pose transposed', CPP, "push.rows[4 * r + c] = pose(r, c);", "push.rows[4 * r + c] = pose(c, r);"),
    ('P2 sentinels counted as voxels', CPP, "const uint32_t valid = n - runs.outside;", "const uint32_t valid = n;"),
    ('P3 sort over 42 bits only', CPP, SORT_BLOCK, SORT_BLOCK.replace("n, 43, keep_alive", "n, 42, keep_alive")),
    ('P4 no sort', CPP, SORT_BLOCK, SORT_BLOCK.replace("if (n > 1)", "if (false)")),
    ('P5 model values swapped', CPP, "RunCountsPush{ valid, model.initial_probability, model.update_probability }", "RunCountsPush{ valid, model.update_probability, model.initial_probability }"),
    ('P6 side length 0 accepted', CPP, "if (!(side > 0.0f) || !std::isfinite(side))", "if (side < 0.0f || !std::isfinite(side))"),
    ('P7 infinite side length accepted', CPP, "if (!(side > 0.0f) || !std::isfinite(side))", "if (!(side > 0.0f))"),
    ('P8 number of voxels is the number of valid points', CPP, "runs.count = distinct_count(heads, offsets, valid);", "runs.count = valid;"),
    ('P9 outside points not reported', CPP, "dropped_now = runs.outside;", "dropped_now = 0;"),
    ('P10 empty cloud skips the check of the side length', CPP, "voxelize(DeviceBuffer(context, 12), 0, pose, side, model); // validates the side length", ";"),
    ('P11 occupancy read as the wrong width', CPP, "download_vector<int32_t>(runs.count)", "download_vector<int32_t>(runs.count > 1 ? runs.count - 1 : 1)"),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=VulkanTest.OctreeVoxelize*'],
                               env=test_env, capture_output=True, text=True, timeout=900)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
