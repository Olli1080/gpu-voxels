# Hand-written mutation runner (slice S6f-2, Octree::cast_free_space_rays: bounding box, 6-connected walk to the origin, exact hit counts, chunks, limits): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s6f2_ray_cast.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
V = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
BOUNDS = V + 'shaders/octree_ray_bounds.comp'
RAY = V + 'shaders/octree_ray_cast.comp'
CPP = V + 'src/Octree.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
# GPU assisted validation in safe mode keeps mutants with wrong addresses from faulting the GPU (see mutate_s6d4_rebuild.py)
test_env = dict(env, VK_KHRONOS_VALIDATION_GPUAV_ENABLE='true', VK_KHRONOS_VALIDATION_GPUAV_SAFE_MODE='true',
                VK_KHRONOS_VALIDATION_VALIDATE_SYNC='false', VK_KHRONOS_VALIDATION_VALIDATE_CORE='false')

SWAP_ENDS_OLD = "    ivec3 p = ivec3(morton_decode60(packUint2x32(uvec2(ids[2u * k], ids[2u * k + 1u])))) - low;\n    const ivec3 target = ivec3(pc.origin_x, pc.origin_y, pc.origin_z) - low;"
SWAP_ENDS_NEW = "    const ivec3 target = ivec3(morton_decode60(packUint2x32(uvec2(ids[2u * k], ids[2u * k + 1u])))) - low;\n    ivec3 p = ivec3(pc.origin_x, pc.origin_y, pc.origin_z) - low;"

MUT = [
    ('B1 y minimum from x', BOUNDS, "atomicMin(bounds[1], c.y);", "atomicMin(bounds[1], c.x);"),
    ('B2 x maximum from y', BOUNDS, "atomicMax(bounds[3], c.x);", "atomicMax(bounds[3], c.y);"),
    ('B3 z maximum missing', BOUNDS, "atomicMax(bounds[5], c.z);", "atomicMax(bounds[5], 0u);"),
    ('B4 x minimum missing', BOUNDS, "atomicMin(bounds[0], c.x);", "atomicMin(bounds[0], 0u);"),
    ('B5 z minimum from y', BOUNDS, "atomicMin(bounds[2], c.z);", "atomicMin(bounds[2], c.y);"),
    ('K1 walk from the origin to the voxel', RAY, SWAP_ENDS_OLD, SWAP_ENDS_NEW),
    ('K2 last cell missing', RAY, "int n = 1 + d.x + d.y + d.z;", "int n = d.x + d.y + d.z;"),
    ('K3 x branch also on equal error', RAY, "error_xy > 0 && error_xz > 0", "error_xy >= 0 && error_xz > 0"),
    ('K4 x branch needs only the first error', RAY, "error_xy > 0 && error_xz > 0", "error_xy > 0"),
    ('K5 y branch also on equal error', RAY, "else if (error_yz > 0)", "else if (error_yz >= 0)"),
    ('K6 error_xy after x step', RAY, "error_xy -= dd.y;", "error_xy += dd.y;"),
    ('K7 error_xz after x step', RAY, "error_xz -= dd.z;", "error_xz += dd.z;"),
    ('K8 error_xy after y step', RAY, "error_xy += dd.x;", "error_xy -= dd.x;"),
    ('K9 error_yz after y step', RAY, "error_yz -= dd.z;", "error_yz += dd.z;"),
    ('K10 error_xz after z step', RAY, "error_xz += dd.x;", "error_xz -= dd.x;"),
    ('K11 error_yz after z step', RAY, "error_yz += dd.y;", "error_yz -= dd.y;"),
    ('K12 step x on equal coordinates (equivalent: no step happens on a zero difference)', RAY, "target.x > p.x ? 1 : -1", "target.x >= p.x ? 1 : -1"),
    ('K13 step y direction inverted', RAY, "target.y > p.y ? 1 : -1", "target.y < p.y ? 1 : -1"),
    ('K14 no guard (equivalent: the walk stays in its box)', RAY, "if (all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, size)))", "if (true)"),
    ('K15 plain add instead of atomic', RAY, "atomicAdd(counts[uint(p.x) + uint(size.x) * (uint(p.y) + uint(size.y) * uint(p.z))], 1u);", "counts[uint(p.x) + uint(size.x) * (uint(p.y) + uint(size.y) * uint(p.z))] += 1u;"),
    ('K16 y and z swapped in the index', RAY, "(uint(p.y) + uint(size.y) * uint(p.z))", "(uint(p.z) + uint(size.z) * uint(p.y))"),
    ('K17 chunk offset ignored', RAY, "const uint k = pc.first + i;", "const uint k = i;"),
    ('K18 one ray too many', RAY, "if (i >= pc.count)", "if (i > pc.count)"),
    ('K19 box size of x taken from y', RAY, "const ivec3 size = ivec3(pc.size_x, pc.size_y, pc.size_z);", "const ivec3 size = ivec3(pc.size_y, pc.size_y, pc.size_z);"),
    ('P1 origin x not checked', CPP, "if (origin.x >= Octree::kCoordinateLimit || origin.y", "if (origin.y"),
    ('P2 limit exclusive', CPP, "if (cells > ray_map_limit)", "if (cells >= ray_map_limit)"),
    ('P3 box without the origin (minimum)', CPP, "rays.box_min[a] = std::min(bounds[a], o[a]);", "rays.box_min[a] = bounds[a];"),
    ('P4 box without the origin (maximum)', CPP, "std::max(bounds[3 + a], o[a]) - rays.box_min[a] + 1;", "bounds[3 + a] - rays.box_min[a] + 1;"),
    ('P5 longest ray too short', CPP, "longest += rays.box_size[a] - 1;", "longest += 0;"),
    ('P6 longest ray without the end cell', CPP, "longest += 1;", "longest += 0;"),
    ('P7 map filled with 1', CPP, "batch.fill(rays.counts, 0);", "batch.fill(rays.counts, 1);"),
    ('P8 map too small', CPP, "rays.counts = DeviceBuffer(context, cells * 4);", "rays.counts = DeviceBuffer(context, cells * 2);"),
    ('P9 voxel x not checked', CPP, "if (v.x >= Octree::kCoordinateLimit || v.y", "if (v.y"),
    ('P10 chunk offset not passed', CPP, "RayPush{ static_cast<uint32_t>(first), count,", "RayPush{ 0, count,"),
    ('P11 origin x and y swapped in the push', CPP, "{ o[0], o[1], o[2] } });", "{ o[1], o[0], o[2] } });"),
    ('P12 box minimum x and y swapped in the push', CPP, "count, { rays.box_min[0], rays.box_min[1], rays.box_min[2] },", "count, { rays.box_min[1], rays.box_min[0], rays.box_min[2] },"),
    ('P13 returned minimum y and z swapped', CPP, "map.min = { rays.box_min[0], rays.box_min[1], rays.box_min[2] };", "map.min = { rays.box_min[0], rays.box_min[2], rays.box_min[1] };"),
    ('P14 returned size x and z swapped', CPP, "map.size = { rays.box_size[0], rays.box_size[1], rays.box_size[2] };", "map.size = { rays.box_size[2], rays.box_size[1], rays.box_size[0] };"),
    ('P15 chunks of the wrong size', CPP, "const uint64_t per_chunk = std::max<uint64_t>(1, ray_chunk_steps / longest);", "const uint64_t per_chunk = std::max<uint64_t>(1, ray_chunk_steps / (longest + 1));"),
    ('P16 submission not counted', CPP, "++ray_submissions;", ";"),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=VulkanTest.OctreeRayCast*'],
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
