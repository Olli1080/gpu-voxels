"""Mutation check for slice S2 (meta cloud transform / insertion / self collision / DH chain)."""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
VK = os.path.join(ROOT, r'packages\gpu_voxels\gpu_voxels\vulkan')
EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

mutations = [
    ('Q1 transposed matrix element in transform shader', r'shaders\meta_transform.comp', 'entries[lo].m[1] * y', 'entries[lo].m[4] * y'),
    ('Q2 translation dropped in transform shader', r'shaders\meta_transform.comp', ' + entries[lo].m[3];', ';'),
    ('Q3 sub cloud search uses < instead of <=', r'shaders\meta_transform.comp', 'if (entries[mid].begin <= i)', 'if (entries[mid].begin < i)'),
    ('Q4 in-place shader transforms disabled clouds too', r'shaders\meta_transform_inplace.comp', 'if (entries[lo].enabled == 0u || i >= entries[lo].end)', 'if (i >= entries[lo].end)'),
    ('Q5 meta insert always uses meaning of cloud 0', r'shaders\bit_voxel_insert_meta.comp', 'const uint meaning = meanings[lo];', 'const uint meaning = meanings[0];'),
    ('Q6 self collision: own meaning not inserted', r'shaders\bit_voxel_insert_meta_selfcoll.comp', '        atomicOr(words[base + meaning_word(pc.meaning)], meaning_mask(pc.meaning));\n    }\n}', '    }\n}'),
    ('Q7 self collision: mask not applied', r'shaders\bit_voxel_insert_meta_selfcoll.comp', ' & masks[pc.sub_cloud * WORDS_PER_VOXEL + w]', ''),
    ('Q8 self collision: own meaning not reported', r'shaders\bit_voxel_insert_meta_selfcoll.comp', '            atomicOr(colliding[meaning_word(pc.meaning)], meaning_mask(pc.meaning));\n', ''),
    ('Q9 self collision: voxel not marked as collision', r'shaders\bit_voxel_insert_meta_selfcoll.comp', '            atomicOr(words[base + meaning_word(BVM_COLLISION)], meaning_mask(BVM_COLLISION));\n', ''),
    ('Q10 self collision: FREE bit counts as collision partner', r'shaders\bit_voxel_insert_meta_selfcoll.comp', '(masked[w] & 0xFFFFFFFEu)', '(masked[w] & 0xFFFFFFFFu)'),
    ('K1 classic DH: missing cos(alpha)', r'src\KinematicChain.cpp', 'm(0, 1) = -st * ca;', 'm(0, 1) = -st;'),
    ('K2 Craig DH: sign of d term', r'src\KinematicChain.cpp', 'm(1, 3) = -d * sa;', 'm(1, 3) = d * sa;'),
    ('K3 chain moves link i with T_{i+1}', r'src\KinematicChain.cpp', 'transformations.push_back({ static_cast<uint16_t>(*cloud_of_link_[i]), transforms_[i] });', 'transformations.push_back({ static_cast<uint16_t>(*cloud_of_link_[i]), transforms_[i + 1] });'),
    ('K4 prismatic joint value added to theta', r'src\KinematicChain.cpp', 'case DHJointType::Prismatic: d = p.d + p.value; break;', 'case DHJointType::Prismatic: theta = p.theta + p.value; break;'),
    ('K5 default masks keep the own bit', r'src\BitVoxelMap.cpp', 'mask[meaning_list[k] / 32] &= ~(1u << (meaning_list[k] % 32));', ''),
    ('K6 default meanings shifted by one', r'src\BitVoxelMap.cpp', 'static_cast<uint8_t>(eBVM_SWEPT_VOLUME_START + k)', 'static_cast<uint8_t>(eBVM_SWEPT_VOLUME_START + k + 1)'),
    ('K7 transform_sub_clouds ignores matrix translation', r'src\MetaPointCloud.cpp', 'e.m[row * 4 + col] = t.transform(row, col);', 'e.m[row * 4 + col] = (col == 3) ? 0.0f : t.transform(row, col);'),
]

def read(path):
    return open(path, 'rb').read()

def build():
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', 'test_vulkan'],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0

def run_tests():
    r = subprocess.run([EXE], capture_output=True, text=True, env=env)
    return re.findall(r'^\[  FAILED  \] VulkanTest\.(\w+) \(', r.stdout, re.M)

touched = {}
try:
    for name, rel, old, new in mutations:
        path = os.path.join(VK, rel)
        original = touched.setdefault(path, read(path))
        text = original.decode('utf-8')
        crlf = '\r\n' in text
        o = old.replace('\n', '\r\n') if crlf else old
        n = new.replace('\n', '\r\n') if crlf else new
        if o not in text:
            print(f'{name}: PATTERN NOT FOUND in {rel}', flush=True)
            continue
        open(path, 'wb').write(text.replace(o, n, 1).encode('utf-8'))
        ok = build()
        if not ok:
            print(f'{name}: BUILD FAILED', flush=True)
        else:
            failed = run_tests()
            print(f'{name}: {"KILLED" if failed else "SURVIVED  <-- weak tests"}; failing: {", ".join(failed) if failed else "-"}', flush=True)
        open(path, 'wb').write(original)
finally:
    for path, data in touched.items():
        open(path, 'wb').write(data)
    ok = build()
    failed = run_tests()
    print('restored; build ok =', ok, '; failing tests at baseline:', failed, flush=True)
