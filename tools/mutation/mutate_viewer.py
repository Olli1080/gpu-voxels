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
EO = VK + 'shaders/viewer_extract_octree.comp'
CMN = VK + 'shaders/viewer_cube_common.glsl'
FIN = VK + 'shaders/viewer_finalize.comp'
SPF = VK + 'shaders/viewer_sphere.frag'
SCN = VK + 'src/ViewerScene.cpp'
GV = VK + 'src/GpuVoxels.cpp'
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
    # octree path (2026-11-04): kernel viewer_extract_octree.comp and CubeExtractor::extract(Octree)
    ('V15 octree kernel: the position is not scaled by the level', EO, "vec4(vec3(c) * edge, edge)", "vec4(vec3(c) * pc.side, edge)"),
    ('V16 octree kernel: the edge ignores the level', EO, "const float edge = float(1u << level) * pc.side;", "const float edge = pc.side;"),
    ('V17 octree kernel: a node with occupied and free bits is drawn free', EO, "else if ((status & STATUS_OCCUPIED) != 0u)",
     "else if ((status & STATUS_OCCUPIED) != 0u && (status & STATUS_FREE) == 0u)"),
    ('V18 octree kernel: unknown nodes are drawn', EO, "if (occupancy == -128 || occupancy < pc.min_value)", "if (occupancy < pc.min_value)"),
    ('V19 octree kernel: threshold exclusive', EO, "if (occupancy == -128 || occupancy < pc.min_value)", "if (occupancy == -128 || occupancy <= pc.min_value)"),
    ('V20 octree kernel: the Morton id is cut to 32 bits', EO, "morton_decode60(id)", "morton_decode60(id & 0xFFFFFFFFul)"),
    ('V21 octree kernel: the occupancy is not sign extended', EO, "const int occupancy = bitfieldExtract(int(word), 16, 8);", "const int occupancy = int((word >> 16u) & 255u);"),
    ('V22 octree host: free nodes are never selected', VW, "(status & octree_format::kFree) ? display.free_colour : 0u", "(status & octree_format::kFree) ? 0u : 0u"),
    ('V23 octree host: occupied nodes get the free colour in the selection', VW, "(status & octree_format::kOccupied) ? display.occupied_colour", "(status & octree_format::kOccupied) ? display.free_colour"),
    ('V24 octree host: the minimum level is not passed on', VW, "options.min_level = display.min_level;", "options.min_level = 0;"),
    ('V25 octree host: the side length of the tree wins over the display', VW, "display.side_length != 0.0f ? display.side_length : tree.voxel_side_length()",
     "tree.voxel_side_length() > 0.0f ? tree.voxel_side_length() : display.side_length"),
    ('V26 octree host: a probabilistic tree is processed as a deterministic one', VW, "probabilistic ? 1u : 0u, display.min_value", "0u, display.min_value"),
    ('V27 octree host: the last cube is not converted', VW, "OctreePush push{ cubes.count, out.capacity()", "OctreePush push{ cubes.count - 1, out.capacity()"),
    ('V28 probabilistic gradient: scale 255 instead of 254', CMN, "float(value + 127) / 254.0", "float(value + 127) / 255.0"),
    # primitives, spheres, scene (2026-11-04)
    ('V29 sphere shader: no discard when the ray misses', SPF, "    if (discriminant < 0.0)", "    if (false)"),
    ('V30 sphere shader: the depth of the hit point is not written', SPF, "    gl_FragDepth = clip.z / clip.w;\n", ""),
    ('V31 sphere shader: radius twice as large', SPF, "v_sphere.w * v_sphere.w)", "4.0 * v_sphere.w * v_sphere.w)"),
    ('V32 cube vertex: the sphere centre is the minimum corner', VERT, "v_sphere = vec4(inst.pos_size.xyz + 0.5 * inst.pos_size.w, 0.5 * inst.pos_size.w);", "v_sphere = vec4(inst.pos_size.xyz, 0.5 * inst.pos_size.w);"),
    ('V33 renderer: sphere buffers use the cube pipeline', VW, "buffer->kind() == CubeKind::Sphere ? *sphere_pipeline_ : *pipeline_", "*pipeline_"),
    ('V34 renderer: the pipeline is bound once', VW, "if (!bound || *bound != buffer->kind())", "if (!bound)"),
    ('V35 renderer: the eye is the camera target', VW, "push.eye[i] = camera.eye[i];", "push.eye[i] = camera.target[i];"),
    ('V36 primitives: the cube starts at the centre', VW, "cubes.push_back({ (p.x - p.radius) * side,", "cubes.push_back({ p.x * side,"),
    ('V37 primitives: spheres get the cube kind', VW, "array.type() == ePRIM_SPHERE ? CubeKind::Sphere : CubeKind::Cube", "CubeKind::Cube"),
    ('V38 assign: found counts only the stored cubes', VW, "static_cast<uint32_t>(cubes.size()) };", "stored };"),
    ('V39 primitives: the edge is the radius', VW, "2.0f * p.radius * side, colour", "p.radius * side, colour"),
    ('V40 scene hook: force_repaint is ignored', SCN, "if (force_repaint && layer->visible)\n				extract(*layer);\n			return true;\n		}\n		return show_map(name);", "if (layer->visible)\n				extract(*layer);\n			return true;\n		}\n		return show_map(name);"),
    ('V41 scene: a layer shown again is not extracted', SCN, "if (visible && !layer.visible)", "if (false)"),
    ('V42 scene refresh: hidden layers are extracted too', SCN, "		for (auto& layer : layers_)\n			if (layer->visible)\n				extract(*layer);", "		for (auto& layer : layers_)\n			extract(*layer);"),
    ('V43 scene: buffers() includes hidden layers', SCN, "			if (layer->visible)\n				result.push_back(layer->cubes.get());", "			if (true)\n				result.push_back(layer->cubes.get());"),
    ('V44 scene: the hooks stay after the scene is gone', SCN, "		voxels_.set_visualization_hooks({}, {});", "		(void)0;"),
    ('V45 facade: the hook result is ignored', GV, "return visualize_map_hook_ ? visualize_map_hook_(map_name, force_repaint) : false;", "return false;"),
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
