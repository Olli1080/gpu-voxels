# Hand-written mutation runner for the in-process viewer core (Viewer.h): one mutation at a time, build, run only the viewer tests, restore.
# The viewer tests check the extraction against host models and the rendered pixels against the projection computed on the host, so every change of
# the extraction kernels, the camera matrix, the pipeline state or the cube shaders must be noticed.
# Usage: python mutate_viewer.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk. Rebuild before using the executable afterwards.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
VK = root + 'packages/gpu_voxels/gpu_voxels/vulkan/'
VW = VK + 'src/Viewer.cpp'
VERT = VK + 'shaders/viewer_cube.vert'
FRAG = VK + 'shaders/viewer_cube.frag'
EB = VK + 'shaders/viewer_extract_bit.comp'
EP = VK + 'shaders/viewer_extract_prob.comp'
EL = VK + 'shaders/viewer_extract_list.comp'
FIN = VK + 'shaders/viewer_finalize.comp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

MUT = [
    ('V1 pipeline: depth test always passes', VW, "setDepthCompareOp(vk::CompareOp::eLess)", "setDepthCompareOp(vk::CompareOp::eAlways)"),
    ('V2 camera: y axis not flipped for Vulkan', VW, "0.0f, -f, 0.0f, 0.0f,", "0.0f, f, 0.0f, 0.0f,"),
    ('V3 camera: depth range halved', VW, "z_far / (z_near - z_far), -1.0f,", "0.5f * z_far / (z_near - z_far), -1.0f,"),
    ('V4 renderer: always shaded', VW, "push.shaded = options.shaded ? 1u : 0u;", "push.shaded = 1u;"),
    ('V5 finalize: instanceCount not limited by the capacity', FIN, "min(args[ARG_FOUND], pc.capacity)", "args[ARG_FOUND]"),
    ('V6 bit map: only the first word of a voxel is read', EB, "uint bits = voxels[i * 8u + w];", "uint bits = voxels[i * 8u];"),
    ('V7 bit map: y taken modulo dim_z', EB, "const uint y = (i / pc.dim_x) % pc.dim_y;", "const uint y = (i / pc.dim_x) % pc.dim_z;"),
    ('V8 bit map: palette index shifted by one', EB, "palette[w * 32u + uint(findLSB(bits))]", "palette[w * 32u + uint(findLSB(bits)) + 1u]"),
    ('V9 prob map: threshold exclusive', EP, "value < pc.min_value", "value <= pc.min_value"),
    ('V10 prob map: unknown voxels are drawn', EP, "value == -128 || value < pc.min_value", "value < pc.min_value"),
    ('V11 list: coordinates not scaled by the side length', EL, "coordinates[3u * i + 2u]) * pc.side", "coordinates[3u * i + 2u])"),
    ('V12 cube vertex: normal sign inverted', VERT, "normal[axis] = sign_;", "normal[axis] = -sign_;"),
    ('V13 cube vertex: the face has no extent along its axis', VERT, "local[axis] = sign_ > 0.0 ? 1.0 : 0.0;", "local[axis] = 0.0;"),
    ('V14 cube fragment: ambient light 0.30', FRAG, "0.35 + 0.65 * diffuse", "0.30 + 0.65 * diffuse"),
]

TESTS = 'Viewer*'
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_filter=' + TESTS],
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
