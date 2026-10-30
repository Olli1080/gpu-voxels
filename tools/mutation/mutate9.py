"""Mutation check for slice S6a (BDA runtime, Morton host/GLSL, octree reference). Every mutation must be killed.
kind 'cpu': built as test_octree_reference and killed by it; kind 'gpu': built as test_vulkan, killed by the filtered GPU tests."""
import os, re, subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
V = r'packages\gpu_voxels\gpu_voxels\vulkan'
MORTON_H = V + r'\include\gpu_voxels\vulkan\Morton.h'
MORTON_GLSL = V + r'\shaders\morton.glsl'
CONTEXT = V + r'\src\Context.cpp'
BUFFER_CPP = V + r'\src\DeviceBuffer.cpp'
REF = r'tests\reference\octree_reference.h'
CPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\Debug\test_octree_reference.exe')
GPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
GPU_FILTER = '--gtest_filter=*DeviceAddress*:*Pointers*:*Morton*'
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

mutations = [
    # ---- host Morton.h
    ('H1 spread: mask removed', 'cpu', MORTON_H, 'x &= 1023u;\n\t\tx = (x | (x << 16))', 'x = (x | (x << 16))'),
    ('H2 spread: constant 1', 'cpu', MORTON_H, '0x030000FFu;\n\t\tx = (x | (x << 8))', '0x030000FEu;\n\t\tx = (x | (x << 8))'),
    ('H3 spread: constant 2', 'cpu', MORTON_H, '(x | (x << 8)) & 0x0300F00Fu', '(x | (x << 8)) & 0x0300F00Eu'),
    ('H4 spread: constant 3', 'cpu', MORTON_H, '(x | (x << 4)) & 0x030C30C3u', '(x | (x << 4)) & 0x030C30C2u'),
    ('H5 spread: constant 4', 'cpu', MORTON_H, '(x | (x << 2)) & 0x09249249u', '(x | (x << 2)) & 0x09249248u'),
    ('H6 spread: shift 16', 'cpu', MORTON_H, '(x | (x << 16))', '(x | (x << 15))'),
    ('H7 compact: mask removed', 'cpu', MORTON_H, 'x &= 0x09249249u;\n\t\tx = (x ^ (x >> 2))', 'x = (x ^ (x >> 2))'),
    ('H8 compact: step 2', 'cpu', MORTON_H, '(x ^ (x >> 2)) & 0x030C30C3u', '(x ^ (x >> 2)) & 0x030C30C2u'),
    ('H9 compact: step 4', 'cpu', MORTON_H, '(x ^ (x >> 4)) & 0x0300F00Fu', '(x ^ (x >> 4)) & 0x0300F00Eu'),
    ('H10 compact: step 8', 'cpu', MORTON_H, '(x ^ (x >> 8)) & 0xFF0000FFu', '(x ^ (x >> 8)) & 0xFF0000FEu'),
    ('H11 compact: final mask', 'cpu', MORTON_H, '(x ^ (x >> 16)) & 0x000003FFu', '(x ^ (x >> 16)) & 0x000001FFu'),
    ('H12 code30: y shift', 'cpu', MORTON_H, '(spread10(y) << 1)', '(spread10(y) << 2)'),
    ('H13 code30: z shift', 'cpu', MORTON_H, '(spread10(z) << 2)', '(spread10(z) << 1)'),
    ('H14 code60: high part shift', 'cpu', MORTON_H, ')) << 30) | uint64_t(code30', ')) << 29) | uint64_t(code30'),
    ('H15 code60: high x', 'cpu', MORTON_H, 'code30(x >> 10,', 'code30(x >> 9,'),
    ('H16 code60: high z', 'cpu', MORTON_H, 'z >> 10)) << 30)', 'z >> 11)) << 30)'),
    ('H17 decode: low part', 'cpu', MORTON_H, 'const uint32_t lo = uint32_t(code);', 'const uint32_t lo = uint32_t(code >> 1);'),
    ('H18 decode: high part', 'cpu', MORTON_H, 'const uint32_t hi = uint32_t(code >> 30);', 'const uint32_t hi = uint32_t(code >> 31);'),
    ('H19 decode: y from wrong bit', 'cpu', MORTON_H, '(compact10(hi >> 1) << 10) | compact10(lo >> 1)', '(compact10(hi >> 1) << 10) | compact10(lo >> 2)'),
    ('H20 decode: z high shift', 'cpu', MORTON_H, '(compact10(hi >> 2) << 10)', '(compact10(hi >> 2) << 9)'),
    ('H21 prefix: level + 1', 'cpu', MORTON_H, 'id >> (3u * (level + 1u))', 'id >> (3u * level)'),
    ('H22 node index: mask', 'cpu', MORTON_H, '(id >> (3u * level)) & 7u)', '(id >> (3u * level)) & 3u)'),
    ('H23 common level: rounding', 'cpu', MORTON_H, '+ 2u) / 3u', '+ 1u) / 3u'),
    ('H24 common level: divisor', 'cpu', MORTON_H, '+ 2u) / 3u', '+ 2u) / 4u'),
    # ---- reference
    ('R1 morton: y bit', 'cpu', REF, 'code |= uint64_t((y >> b) & 1u) << (3 * b + 1);', 'code |= uint64_t((y >> b) & 1u) << (3 * b + 2);'),
    ('R2 morton: bit count', 'cpu', REF, 'for (uint32_t b = 0; b < 20; ++b)\n\t\t{\n\t\t\tcode |=', 'for (uint32_t b = 0; b < 19; ++b)\n\t\t{\n\t\t\tcode |='),
    ('R3 unmorton: z bit', 'cpu', REF, 'v.z |= uint32_t((code >> (3 * b + 2)) & 1u) << b;', 'v.z |= uint32_t((code >> (3 * b + 1)) & 1u) << b;'),
    ('R4 common level: shift', 'cpu', REF, 'a >>= 3;\n\t\t\tb >>= 3;', 'a >>= 2;\n\t\t\tb >>= 2;'),
    ('R5 status: single bit equality', 'cpu', REF, '(single_bit && o == a) ? a :', 'single_bit ? a :'),
    ('R6 status: PART', 'cpu', REF, 'uint8_t(o | status::PART)', 'o'),
    ('R7 status: LAST_LEVEL level', 'cpu', REF, 'if (level == 1)\n\t\t\tresult |= status::LAST_LEVEL;', 'if (level == 2)\n\t\t\tresult |= status::LAST_LEVEL;'),
    ('R8 status: dynamic bit dropped', 'cpu', REF, 'static_dynamic_or & (status::STATIC | status::DYNAMIC)', 'static_dynamic_or & status::DYNAMIC'),
    ('R9 status: single bit FREE', 'cpu', REF, 'a == status::FREE || a == status::UNKNOWN || a == status::OCCUPIED', 'a == status::FREE || a == status::OCCUPIED'),
    ('R10 dropped: limit', 'cpu', REF, 'v.x >= kCoordinateLimit', 'v.x > kCoordinateLimit'),
    ('R11 dropped: y checked', 'cpu', REF, '|| v.y >= kCoordinateLimit', ''),
    ('R12 unique removed', 'cpu', REF, 'ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());', ''),
    ('R13 sort removed', 'cpu', REF, 'std::sort(ids_.begin(), ids_.end());', ''),
    ('R14 empty root status', 'cpu', REF, 'root.status = status::UNKNOWN;\n\t\t\t\troot.occupancy', 'root.status = 0;\n\t\t\t\troot.occupancy'),
    ('R15 empty root occupancy', 'cpu', REF, 'kind_ == Kind::Probabilistic ? kUnknownOccupancy : int8_t(0);\n\t\t\t\tnodes_', 'int8_t(0);\n\t\t\t\tnodes_'),
    ('R16 find: level index', 'cpu', REF, 'id >> (3 * (node->level - 1))', 'id >> (3 * node->level)'),
    ('R17 filler: STATIC', 'cpu', REF, 'n.status = status::UNKNOWN | status::STATIC;\n\t\t\tif (kind_', 'n.status = status::UNKNOWN;\n\t\t\tif (kind_'),
    ('R18 filler: LAST_LEVEL', 'cpu', REF, 'if (level == 1)\n\t\t\t\tn.status |= status::LAST_LEVEL;', ''),
    ('R19 filler: probabilistic occupancy', 'cpu', REF, 'n.occupancy = kUnknownOccupancy;\n\t\t\tif (level == 1)', 'n.occupancy = 0;\n\t\t\tif (level == 1)'),
    ('R20 leaf: STATIC', 'cpu', REF, 'n.status = status::STATIC | (kind_', 'n.status = (kind_'),
    ('R21 leaf: probabilistic occupancy', 'cpu', REF, 'kind_ == Kind::Probabilistic ? kOccupied : int8_t(0);\n\t\t\t}', 'kind_ == Kind::Probabilistic ? int8_t(126) : int8_t(0);\n\t\t\t}'),
    ('R22 children: id', 'cpu', REF, 'const uint64_t child_id = id * 8 + c;', 'const uint64_t child_id = id * 8;'),
    ('R23 children: range shift', 'cpu', REF, '(ids_[child_end] >> (3 * (level - 1))) == child_id', '(ids_[child_end] >> (3 * level)) == child_id'),
    ('R24 prob: all non part', 'cpu', REF, 'if (!(pmax == pmin && all_non_part))', 'if (!(pmax == pmin))'),
    ('R25 prob: max', 'cpu', REF, 'pmax = std::max<int>(pmax, c.occupancy);', 'pmax = std::min<int>(pmax, c.occupancy);'),
    ('R26 prob: min initial', 'cpu', REF, 'pmin = children[0].occupancy;', 'pmin = 126;'),
    ('R27 prob: min', 'cpu', REF, 'pmin = std::min<int>(pmin, c.occupancy);', 'pmin = std::max<int>(pmin, c.occupancy);'),
    ('R28 prob: LAST_LEVEL', 'cpu', REF, 'if (level == 1)\n\t\t\t\ts |= status::LAST_LEVEL;', 'if (level == 2)\n\t\t\t\ts |= status::LAST_LEVEL;'),
    ('R29 prob: UNKNOWN dropped', 'cpu', REF, 'uint8_t s = status::UNKNOWN | status::STATIC;', 'uint8_t s = status::STATIC;'),
    ('R30 det: and initial', 'cpu', REF, 'uint8_t o = 0, a = 0xFF, sd = 0;', 'uint8_t o = 0, a = 0, sd = 0;'),
    ('R31 det: static or', 'cpu', REF, 'sd |= c.status;', 'sd |= c.status & status::DYNAMIC;'),
    ('R33 status table: dynamic mask', 'cpu', REF, 'static_dynamic_or & (status::STATIC | status::DYNAMIC)', 'static_dynamic_or'),
    ('R34 status: FREE bit mask', 'cpu', REF, 'const uint8_t o = child_or & status::OCCUPANCY_MASK;', 'const uint8_t o = child_or & (status::UNKNOWN | status::OCCUPIED);'),
    ('R32 node_count level', 'cpu', REF, 'return size_t(std::count_if(nodes_.begin(), nodes_.end(), [level](const auto& kv) { return kv.first.first == level; }));', 'return size_t(std::count_if(nodes_.begin(), nodes_.end(), [level](const auto& kv) { return kv.first.first <= level; }));'),
    # ---- GLSL
    ('G1 spread: mask removed', 'gpu', MORTON_GLSL, '    x &= 1023u;\n    x = (x | (x << 16))', '    x = (x | (x << 16))'),
    ('G2 spread: constant', 'gpu', MORTON_GLSL, '0x0300F00Fu;\n    x = (x | (x << 4))', '0x0300F00Eu;\n    x = (x | (x << 4))'),
    ('G3 compact: mask removed', 'gpu', MORTON_GLSL, '    x &= 0x09249249u;\n    x = (x ^ (x >> 2))', '    x = (x ^ (x >> 2))'),
    ('G4 compact: constant', 'gpu', MORTON_GLSL, '(x ^ (x >> 8)) & 0xFF0000FFu', '(x ^ (x >> 8)) & 0xFF0000FEu'),
    ('G5 code30: z shift', 'gpu', MORTON_GLSL, '(morton_spread10(c.z) << 2)', '(morton_spread10(c.z) << 1)'),
    ('G6 code60: high shift', 'gpu', MORTON_GLSL, 'morton_code30(c >> 10u)', 'morton_code30(c >> 9u)'),
    ('G7 code60: high shift', 'gpu', MORTON_GLSL, ') << 30) | uint64_t(morton_code30(c))', ') << 29) | uint64_t(morton_code30(c))'),
    ('G8 decode: low part', 'gpu', MORTON_GLSL, 'const uint lo = uint(code);', 'const uint lo = uint(code >> 1);'),
    ('G9 decode: high part', 'gpu', MORTON_GLSL, 'const uint hi = uint(code >> 30);', 'const uint hi = uint(code >> 31);'),
    ('G10 decode: y', 'gpu', MORTON_GLSL, '(morton_compact10(hi >> 1) << 10) | morton_compact10(lo >> 1)', '(morton_compact10(hi >> 1) << 10) | morton_compact10(lo >> 2)'),
    ('G11 prefix: level', 'gpu', MORTON_GLSL, 'return id >> (3u * (level + 1u));', 'return id >> (3u * level);'),
    ('G12 node index: mask', 'gpu', MORTON_GLSL, 'uint(id >> (3u * level)) & 7u', 'uint(id >> (3u * level)) & 3u'),
    ('G13 common level: high word', 'gpu', MORTON_GLSL, 'hi != 0u ? 33u + uint(findMSB(hi))', 'hi != 0u ? 32u + uint(findMSB(hi))'),
    ('G14 common level: low word', 'gpu', MORTON_GLSL, '(lo != 0u ? 1u + uint(findMSB(lo)) : 0u)', '(lo != 0u ? uint(findMSB(lo)) : 0u)'),
    ('G15 common level: rounding', 'gpu', MORTON_GLSL, 'return (width + 2u) / 3u;', 'return (width + 1u) / 3u;'),
    ('G16 common level: xor', 'gpu', MORTON_GLSL, 'const uint64_t x = a ^ b;', 'const uint64_t x = a | b;'),
    # ---- runtime
    ('B1 context: switch ignored', 'gpu', CONTEXT, 'create_info.enable_buffer_device_address && f12', 'create_info.enable_buffer_device_address || f12'),
    ('B2 context: feature not enabled', 'gpu', CONTEXT, 'features12.setBufferDeviceAddress(info_.buffer_device_address);', 'features12.setBufferDeviceAddress(false);'),
    ('B3 context: VMA flag', 'gpu', CONTEXT, 'allocator_info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;', 'allocator_info.flags |= 0;'),
    ('B4 buffer: usage flag', 'gpu', BUFFER_CPP, 'usage |= vk::BufferUsageFlagBits::eShaderDeviceAddress;', ''),
    ('B5 buffer: address not queried', 'gpu', BUFFER_CPP, 'address_ = context.device().getBufferAddress(vk::BufferDeviceAddressInfo{ buffer_ });', 'address_ = 0;'),
    ('B6 buffer: missing address not reported', 'gpu', BUFFER_CPP, 'if (address_ == 0)\n\t\t\tthrow VulkanError("DeviceBuffer::device_address', 'if (false)\n\t\t\tthrow VulkanError("DeviceBuffer::device_address'),
    ('B7 buffer: move constructor keeps address', 'gpu', BUFFER_CPP, ',\n\t\t  address_(std::exchange(other.address_, 0))', ',\n\t\t  address_(other.address_)'),
    ('B8 buffer: move assignment keeps address', 'gpu', BUFFER_CPP, 'address_ = std::exchange(other.address_, 0);', 'address_ = other.address_;'),
    ('B9 buffer: address offset by one', 'gpu', BUFFER_CPP, 'return address_;', 'return address_ + 1;'),
]

only = os.environ.get('ONLY')
done = set()
if os.environ.get('DONE_LOG'):
    for line in open(os.environ['DONE_LOG'], encoding='utf-8'):
        if ': KILLED' in line:
            done.add(line.split(': KILLED')[0])


def read(path):
    return open(path, 'rb').read()


def build(kind):
    target = 'test_octree_reference' if kind == 'cpu' else 'test_vulkan'
    r = subprocess.run(['cmake', '--build', os.path.join(ROOT, 'build', 'verify-vk'), '--config', 'Debug', '--target', target],
                       capture_output=True, text=True, env=env)
    return r.returncode == 0


def run_tests(kind):
    cmd = [CPU_EXE] if kind == 'cpu' else [GPU_EXE, GPU_FILTER]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return re.findall(r'^\[  FAILED  \] (\S+)', r.stdout, re.M), r.returncode


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
        if o not in text:
            print(f'{name}: PATTERN NOT FOUND in {rel}', flush=True)
            survivors.append(name + ' (pattern)')
            continue
        open(path, 'wb').write(text.replace(o, n, 1).encode('utf-8'))
        if not build(kind):
            print(f'{name}: BUILD FAILED', flush=True)
            survivors.append(name + ' (build)')
        else:
            failed, code = run_tests(kind)
            if failed or code != 0:
                print(f'{name}: KILLED by {len(failed)} test(s): {", ".join(sorted(set(failed))[:3])}', flush=True)
            else:
                print(f'{name}: SURVIVED  <-- weak tests', flush=True)
                survivors.append(name)
        open(path, 'wb').write(original)
finally:
    for path, data in touched.items():
        open(path, 'wb').write(data)
    for kind in ('cpu', 'gpu'):
        ok = build(kind)
        failed, code = run_tests(kind)
        print(f'restored {kind}; build ok =', ok, '; failing tests at baseline:', failed, 'exit', code, flush=True)
print('SURVIVORS:', survivors, flush=True)
