"""Mutation check for slice S6b (64-bit sort, octree node formats, build/find/extract kernels, Octree host code).
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
HIST64 = SH + r'\radix_histogram64.comp'
SCATTER64 = SH + r'\radix_scatter64.comp'
CPP = V + r'\src\Octree.cpp'
PRIM = V + r'\src\Primitives.cpp'
FORMAT = V + r'\include\gpu_voxels\vulkan\OctreeFormat.h'
REF = r'tests\reference\octree_reference.h'
CPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\Debug\test_octree_reference.exe')
GPU_EXE = os.path.join(ROOT, r'build\verify-vk\tests\vulkan\Debug\test_vulkan.exe')
OCTREE_FILTER = '--gtest_filter=*Octree*:*CheckTree*:*FindStops*:-*OneMillion*'
SORT_FILTER = '--gtest_filter=*SortPairs64*'
env = dict(os.environ, VULKAN_SDK=os.environ.get('VULKAN_SDK', r'C:\VulkanSDK\1.4.363.0'))

mutations = [
    # ---- node formats in GLSL
    ('GN1 sign extension shift', 'gpu', NODES, '(int64_t(raw) << 16) >> 16', '(int64_t(raw) << 15) >> 15'),
    ('GN2 prob occupancy unsigned', 'gpu', NODES, 'n.occupancy = bitfieldExtract(int(w0), 16, 8);', 'n.occupancy = int((w0 >> 16u) & 255u);'),
    ('GN3 prob child low word dropped', 'gpu', NODES, 'oct_sign_extend_48(uint64_t(b.w[3u * slot + 1u]) | (uint64_t(b.w[3u * slot + 2u]) << 32))', 'oct_sign_extend_48(uint64_t(b.w[3u * slot + 2u]) << 32)'),
    ('GN4 prob store occupancy shift', 'gpu', NODES, '((uint(n.occupancy) & 255u) << 16u);\n    b.w[3u * slot + 1u]', '((uint(n.occupancy) & 255u) << 15u);\n    b.w[3u * slot + 1u]'),
    ('GN5 prob store high child mask', 'gpu', NODES, 'uint((n.child >> 32) & 0xFFFFul)', 'uint(n.child >> 32)'),
    ('GN6 det store high word', 'gpu', NODES, 'b.w[2u * slot + 1u] = uint(n.child >> 16);', 'b.w[2u * slot + 1u] = uint(n.child >> 15);'),
    ('GN7 det store low part', 'gpu', NODES, '(uint(n.child) << 16u);', '(uint(n.child) << 17u);'),
    ('GN8 det load high word', 'gpu', NODES, '(uint64_t(b.w[2u * slot + 1u]) << 16)', '(uint64_t(b.w[2u * slot + 1u]) << 15)'),
    ('GN9 det status mask', 'gpu', NODES, '    n.status = w0 & 255u;\n    n.flags = (w0 >> 8u) & 255u;\n    n.occupancy = 0;', '    n.status = w0 & 0xFFFFu;\n    n.flags = (w0 >> 8u) & 255u;\n    n.occupancy = 0;'),
    ('GN10 prob leaf shift', 'gpu', NODES, '((slot & 1u) * 16u)', '((slot & 1u) * 8u)'),
    ('GN11 det leaf shift', 'gpu', NODES, '((slot & 3u) * 8u)', '((slot & 3u) * 4u)'),
    ('GN12 prob leaf occupancy unsigned', 'gpu', NODES, 'n.occupancy = bitfieldExtract(int(v), 8, 8);', 'n.occupancy = int(v >> 8u);'),
    ('GN13 prob leaf store', 'gpu', NODES, 'b.w[w] = lo | (hi << 16u);', 'b.w[w] = lo | (hi << 8u);'),
    ('GN14 det leaf store', 'gpu', NODES, '(n[4u * w + 2u].status << 16u)', '(n[4u * w + 2u].status << 15u)'),
    ('GN15 filler level 1', 'gpu', NODES, '(level == 1u ? OCT_LAST_LEVEL : 0u)', '(level == 2u ? OCT_LAST_LEVEL : 0u)'),
    ('GN16 filler prob occupancy', 'gpu', NODES, 'n.occupancy = OCT_PROB_UNKNOWN;', 'n.occupancy = 0;'),
    ('GN17 occupied det leaf', 'gpu', NODES, 'n.status = OCT_OCCUPIED | OCT_STATIC;', 'n.status = OCT_OCCUPIED;'),
    ('GN18 occupied prob leaf', 'gpu', NODES, 'n.occupancy = OCT_PROB_MAX;', 'n.occupancy = OCT_PROB_MAX - 1;'),
    ('GN19 prob min initial', 'gpu', NODES, 'int pmin = kids[0].occupancy;', 'int pmin = 126;'),
    ('GN20 prob max', 'gpu', NODES, 'pmax = max(pmax, kids[i].occupancy);', 'pmax = min(pmax, kids[i].occupancy);'),
    ('GN21 prob all non part', 'gpu', NODES, '&& (kids[i].status & OCT_PART) == 0u;', '&& true;'),
    ('GN22 prob part rule', 'gpu', NODES, 'if (!(pmax == pmin && all_non_part))', 'if (!(pmax == pmin))'),
    ('GN23 prob last level', 'gpu', NODES, 'if (level == 1u)\n        p.status |= OCT_LAST_LEVEL;\n#else', 'if (level == 2u)\n        p.status |= OCT_LAST_LEVEL;\n#else'),
    ('GN24 det and initial', 'gpu', NODES, 'uint child_and = 255u;', 'uint child_and = 0u;'),
    ('GN25 det single bit unknown', 'gpu', NODES, 'child_and == OCT_FREE || child_and == OCT_UNKNOWN || child_and == OCT_OCCUPIED', 'child_and == OCT_FREE || child_and == OCT_OCCUPIED'),
    ('GN26 det static dynamic mask', 'gpu', NODES, 'p.status |= flags_or & (OCT_STATIC | OCT_DYNAMIC);', 'p.status |= flags_or & OCT_DYNAMIC;'),
    ('GN27 det last level', 'gpu', NODES, 'if (level == 1u)\n        p.status |= OCT_LAST_LEVEL;\n#endif', 'if (level == 2u)\n        p.status |= OCT_LAST_LEVEL;\n#endif'),
    ('GN28 det inner block bytes', 'gpu', NODES, 'const uint OCT_INNER_BLOCK_BYTES = 64u;', 'const uint OCT_INNER_BLOCK_BYTES = 72u;'),
    ('GN29 det leaf block bytes', 'gpu', NODES, 'const uint OCT_LEAF_BLOCK_BYTES = 8u;', 'const uint OCT_LEAF_BLOCK_BYTES = 16u;'),
    ('GN30 prob inner block bytes', 'gpu', NODES, 'const uint OCT_INNER_BLOCK_BYTES = 96u;', 'const uint OCT_INNER_BLOCK_BYTES = 104u;'),
    ('GN31 det or mask', 'gpu', NODES, 'child_or |= kids[i].status & OCT_OCCUPANCY_MASK;', 'child_or |= kids[i].status;'),
    ('GN32 block bytes by level', 'gpu', NODES, 'return level == 0u ? OCT_LEAF_BLOCK_BYTES : OCT_INNER_BLOCK_BYTES;', 'return level == 1u ? OCT_LEAF_BLOCK_BYTES : OCT_INNER_BLOCK_BYTES;'),
    # ---- build kernel
    ('GB1 parent shift', 'gpu', BUILD, 'const uint shift = 3u * (pc.level + 1u);', 'const uint shift = 3u * pc.level;'),
    ('GB2 slot mask', 'gpu', BUILD, 'const uint slot = uint(id >> (3u * pc.level)) & 7u;', 'const uint slot = uint(id >> (3u * pc.level)) & 3u;'),
    ('GB3 child block by element', 'gpu', BUILD, 'uint64_t(j) * uint64_t(oct_block_bytes(pc.level - 1u))', 'uint64_t(i) * uint64_t(oct_block_bytes(pc.level - 1u))'),
    ('GB4 child block size', 'gpu', BUILD, 'uint64_t(oct_block_bytes(pc.level - 1u))', 'uint64_t(oct_block_bytes(pc.level))'),
    ('GB6 leaf children at level 1', 'gpu', BUILD, 'if (pc.level == 1u)\n            {\n                OctLeafBlock lb', 'if (pc.level == 2u)\n            {\n                OctLeafBlock lb'),
    ('GB7 filler level', 'gpu', BUILD, 'nodes[s] = oct_filler(pc.level);', 'nodes[s] = oct_filler(0u);'),
    ('GB8 block address', 'gpu', BUILD, 'pc.chunk + uint64_t(block) * uint64_t(oct_block_bytes(pc.level))', 'pc.chunk + uint64_t(block) * uint64_t(oct_block_bytes(pc.level + 1u))'),
    ('GB9 next id high word', 'gpu', BUILD, 'next_ids[2u * block + 1u] = ids[2u * i + 1u];', 'next_ids[2u * block + 1u] = 0u;'),
    ('GB10 heads test', 'gpu', BUILD, 'heads[i] == 0u)\n        return;\n    const uint block', 'heads[i] == 1u)\n        return;\n    const uint block'),
    ('GB11 leaf for level 0', 'gpu', BUILD, 'if (pc.level == 0u)\n            nodes[slot] = oct_occupied_leaf();', 'if (pc.level == 1u)\n            nodes[slot] = oct_occupied_leaf();'),
    ('GB12 parent level argument', 'gpu', BUILD, 'nodes[slot] = oct_parent(kids, pc.level, child_ptr);', 'nodes[slot] = oct_parent(kids, pc.level + 1u, child_ptr);'),
    ('GB13 child pointer stored', 'gpu', BUILD, 'nodes[slot] = oct_parent(kids, pc.level, child_ptr);', 'nodes[slot] = oct_parent(kids, pc.level, child_ptr + uint64_t(oct_block_bytes(0u)));'),
    ('GB14 leaf store', 'gpu', BUILD, 'oct_store_leaf_block(OctLeafBlock(address), nodes);', 'oct_store_leaf_block(OctLeafBlock(address + 8ul), nodes);'),
    # ---- find kernel
    ('GF1 outside bound', 'gpu', FIND, '(id >> 42) != 0ul', '(id >> 43) != 0ul'),
    ('GF2 uniform nodes end the walk', 'gpu', FIND, 'while (level > 0u && (node.status & OCT_PART) != 0u && node.child != 0ul)', 'while (level > 0u && node.child != 0ul)'),
    ('GF3 child pointer test', 'gpu', FIND, '(node.status & OCT_PART) != 0u && node.child != 0ul)', '(node.status & OCT_PART) != 0u)'),
    ('GF4 slot shift', 'gpu', FIND, 'const uint slot = uint(id >> (3u * level)) & 7u;', 'const uint slot = uint(id >> (3u * (level + 1u))) & 7u;'),
    ('GF5 leaf level', 'gpu', FIND, 'if (level == 0u)\n            node = oct_load_leaf', 'if (level == 1u)\n            node = oct_load_leaf'),
    ('GF6 occupancy result', 'gpu', FIND, 'uint(node.occupancy) & 255u, 0u);', '0u, 0u);'),
    ('GF7 root level', 'gpu', FIND, 'uint level = OCT_ROOT_LEVEL;', 'uint level = 13u;'),
    ('GF8 loop bound', 'gpu', FIND, 'while (level > 0u &&', 'while (level > 1u &&'),
    ('GF9 result order', 'gpu', FIND, 'result[i] = uvec4(level, node.status,', 'result[i] = uvec4(node.status, level,'),
    # ---- extract kernel
    ('GE1 min level ignored for output', 'gpu', EXTRACT, '(!part || at_min_level)', '(!part)'),
    ('GE2 part nodes are output', 'gpu', EXTRACT, '(!part || at_min_level)', '(part || at_min_level)'),
    ('GE3 min level ignored for descent', 'gpu', EXTRACT, 'if (part && !at_min_level && node.child != 0ul)', 'if (part && node.child != 0ul)'),
    ('GE4 last level inverted', 'gpu', EXTRACT, '        if (!last_level)\n        {', '        if (last_level)\n        {'),
    ('GE5 node id', 'gpu', EXTRACT, 'const uint64_t id = block_id * 8ul + uint64_t(lane);', 'const uint64_t id = block_id * 8ul;'),
    ('GE6 leaf id', 'gpu', EXTRACT, 'emit(id * 8ul + uint64_t(c), 0u, leaf.status, leaf.occupancy);', 'emit(id * 8ul, 0u, leaf.status, leaf.occupancy);'),
    ('GE7 leaf selection', 'gpu', EXTRACT, 'if (selection[leaf.status] != 0u)', 'if (true)'),
    ('GE8 next item id high word', 'gpu', EXTRACT, 'next_items[4u * index + 3u] = uint(id >> 32);', 'next_items[4u * index + 3u] = 0u;'),
    ('GE9 next item address', 'gpu', EXTRACT, 'next_items[4u * index] = uint(node.child & 0xFFFFFFFFul);', 'next_items[4u * index] = uint((node.child >> 1) & 0xFFFFFFFFul);'),
    ('GE10 output packing', 'gpu', EXTRACT, 'level | (status << 8u) | ((uint(occupancy) & 255u) << 16u)', 'level | (status << 9u) | ((uint(occupancy) & 255u) << 16u)'),
    ('GE11 item index', 'gpu', EXTRACT, 'const uint item = t >> 3u;', 'const uint item = t >> 2u;'),
    ('GE13 thread bound', 'gpu', EXTRACT, 'if (t >= pc.item_count * 8u)', 'if (t >= pc.item_count * 7u)'),
    ('GE14 min level test', 'gpu', EXTRACT, 'const bool at_min_level = pc.level == pc.min_level;', 'const bool at_min_level = pc.level + 1u == pc.min_level;'),
    ('GE15 node selection', 'gpu', EXTRACT, 'if (selection[node.status] != 0u &&', 'if (true &&'),
    ('GE16 occupancy output', 'gpu', EXTRACT, 'out_nodes[4u * index + 2u] = level | (status << 8u) | ((uint(occupancy) & 255u) << 16u);', 'out_nodes[4u * index + 2u] = level | (status << 8u);'),
    ('GE17 emit counter', 'gpu', EXTRACT, 'const uint index = atomicAdd(counters[1], 1u);', 'const uint index = atomicAdd(counters[1], 2u);'),
    ('GE18 emit id high word', 'gpu', EXTRACT, 'out_nodes[4u * index + 1u] = uint(id >> 32);', 'out_nodes[4u * index + 1u] = 0u;'),
    # ---- small kernels
    ('GH1 first head', 'gpu', HEADS, '(i == 0u ||', '(i == 1u ||'),
    ('GH2 shift applied', 'gpu', HEADS, '(id_at(i) >> pc.shift) != (id_at(i - 1u) >> pc.shift)', '(id_at(i) >> pc.shift) != (id_at(i - 1u) >> (pc.shift + 3u))'),
    ('GH3 compact low word', 'gpu', COMPACT, 'out_ids[2u * offsets[i]] = ids[2u * i];', 'out_ids[2u * offsets[i]] = ids[2u * i + 1u];'),
    ('GH4 compact high word', 'gpu', COMPACT, 'out_ids[2u * offsets[i] + 1u] = ids[2u * i + 1u];', 'out_ids[2u * offsets[i] + 1u] = 0u;'),
    ('GM1 coordinate bound', 'gpu', MORTON_IDS, 'all(lessThan(c, uvec3(16384u)))', 'all(lessThan(c, uvec3(16385u)))'),
    ('GM2 sentinel', 'gpu', MORTON_IDS, 'uint64_t id = 1ul << 42;', 'uint64_t id = 1ul << 43;'),
    ('GM3 dropped count', 'gpu', MORTON_IDS, 'atomicAdd(dropped[0], 1u);', 'atomicAdd(dropped[0], 0u);'),
    ('GM4 coordinate y', 'gpu', MORTON_IDS, 'coords[3u * i + 1u]', 'coords[3u * i + 2u]'),
    ('GG1 2D index', 'gpu', GID, 'gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x', 'gl_GlobalInvocationID.y * gl_NumWorkGroups.x'),
    # ---- host code
    ('C1 grid rows', 'gpu', CPP, '(groups + x - 1) / x)', '(groups / x))'),
    ('C3 group count rounding', 'gpu', CPP, '(items + kLocalSize - 1) / kLocalSize, 1)', '(items) / kLocalSize, 1)'),
    ('C4 empty root status', 'gpu', CPP, 'root.status = kUnknown;', ''),
    ('C5 block count', 'gpu', CPP, '+ read_word(level_heads, count - 1);', '+ 0;'),
    ('C6 distinct count', 'gpu', CPP, '+ read_word(heads, n - 1);', '+ 0;'),
    ('C7 level shift', 'gpu', CPP, 'HeadsPush{ count, 3 * (level + 1) }', 'HeadsPush{ count, 3 * level }'),
    ('C8 previous address', 'gpu', CPP, '				previous_address = address;\n', ''),
    ('C9 coordinate sort bits', 'gpu', CPP, 'n, 43, keep_alive); // sentinel', 'n, 42, keep_alive); // sentinel'),
    ('C10 id sort bits', 'gpu', CPP, 'sort_and_build(ids, n, n, 42);', 'sort_and_build(ids, n, n, 41);'),
    ('C11 dropped assignment', 'gpu', CPP, '			dropped = dropped_now;\n		}\n\n		void build_ids', '		}\n\n		void build_ids'),
    ('C12 min level check', 'gpu', CPP, 'if (options.min_level >= kRootLevel)', 'if (options.min_level > kRootLevel)'),
    ('C13 selection size check', 'gpu', CPP, 'options.status_selection.size() != 256)', 'options.status_selection.size() != 255)'),
    ('C14 selection copied', 'gpu', CPP, 'selection[i] = options.status_selection[i] ? 1u : 0u;', 'selection[i] = 1u;'),
    ('C15 last level traversed', 'gpu', CPP, 'level >= int(std::max(options.min_level, 1u))', 'level > int(std::max(options.min_level, 1u))'),
    ('C16 capacity check', 'gpu', CPP, 'if (produced > next_capacity)', 'if (produced >= next_capacity)'),
    ('C17 has_children in dump', 'gpu', CPP, 'level > 0 && node.child != 0 }', 'level > 0 && (node.status & kPart) != 0 }'),
    ('C18 dump order', 'gpu', CPP, 'std::sort(nodes.begin(), nodes.end(), [](const OctreeNode& a, const OctreeNode& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });', ''),
    ('C19 byte count', 'gpu', CPP, 'stats.bytes += impl_->chunks[level]->buffer.size();', 'stats.bytes += 0;'),
    ('C20 voxel count', 'gpu', CPP, '			voxels = distinct;\n', ''),
    ('C21 sentinel bound', 'gpu', CPP, 'if (id < kSentinelId)', 'if (id <= kSentinelId)'),
    ('C22 leaf flag check', 'gpu', CPP, 'if (node.status & (kPart | kLastLevel))', 'if (node.status & kPart)'),
    ('C23 LAST_LEVEL check', 'gpu', CPP, '((node.status & kLastLevel) != 0) != (level == 1)', '((node.status & kLastLevel) != 0) != (level == 2)'),
    ('C24 PART without child', 'gpu', CPP, '(node.status & kPart) && node.child == 0', '(node.status & kPart) && node.child == 1'),
    ('C25 chunk range check', 'gpu', CPP, 'node.child < base || node.child >= base + chunk->buffer.size()', 'node.child < base'),
    ('C26 alignment check', 'gpu', CPP, 'if (offset % block_size != 0)', 'if (offset % block_size > 4)'),
    ('C27 double reference', 'gpu', CPP, 'if (visited[level - 1][block])', 'if (false)'),
    ('C28 rule check', 'gpu', CPP, 'if (expected.status != node.status || expected.occupancy != node.occupancy)', 'if (expected.status != node.status)'),
    ('C29 orphan check', 'gpu', CPP, 'if (seen != chunks[level]->blocks)', 'if (seen > chunks[level]->blocks)'),
    ('C30 find pad', 'gpu', CPP, 'static_cast<int8_t>(raw[4 * i + 2])', 'static_cast<int8_t>(raw[4 * i + 3])'),
    # ---- host format
    ('F1 prob occupancy shift', 'fmt', FORMAT, 'n.occupancy = int8_t(uint8_t(w0 >> 16));', 'n.occupancy = int8_t(uint8_t(w0 >> 15));'),
    ('F2 det child high', 'fmt', FORMAT, 'uint64_t(w0 >> 16) | (uint64_t(word(block, 2 * slot + 1)) << 16)', 'uint64_t(w0 >> 16) | (uint64_t(word(block, 2 * slot + 1)) << 15)'),
    ('F5 parent min', 'fmt', FORMAT, 'pmin = pmin < k.occupancy ? pmin : k.occupancy;', 'pmin = pmin > k.occupancy ? pmin : k.occupancy;'),
    ('F6 parent single bit', 'fmt', FORMAT, 'const bool single_bit = a == kFree || a == kUnknown || a == kOccupied;', 'const bool single_bit = a == kFree || a == kOccupied;'),
    ('F7 filler level', 'fmt', FORMAT, '(level == 1 ? kLastLevel : 0)', '(level == 2 ? kLastLevel : 0)'),
    ('F8 sign extension', 'fmt', FORMAT, '(int64_t(raw << 16)) >> 16', '(int64_t(raw << 16)) >> 15'),
    ('F9 addressable wrap', 'fmt', FORMAT, 'last >= address &&', ''),
    ('F9b addressable empty', 'fmt', FORMAT, 'return last >= address &&', 'return size != 0 &&'),
    ('F10 leaf mask', 'fmt', FORMAT, 'uint32_t w = word(block, slot >> 1) & ~(0xFFFFu << shift);', 'uint32_t w = word(block, slot >> 1) & ~0u;'),
    ('F11 block bytes', 'fmt', FORMAT, 'return kind == OctreeKind::Probabilistic ? 96 : 64;', 'return kind == OctreeKind::Probabilistic ? 96 : 72;'),
    ('F12 prob store high word', 'fmt', FORMAT, 'set_word(block, 3 * slot + 2, uint32_t((n.child >> 32) & 0xFFFF));', 'set_word(block, 3 * slot + 2, uint32_t(n.child >> 32));'),
    ('F13 parent last level', 'fmt', FORMAT, '		if (level == 1)\n			p.status |= kLastLevel;\n		return p;', '		if (level == 2)\n			p.status |= kLastLevel;\n		return p;'),
    ('F14 parent static', 'fmt', FORMAT, 'p.status |= sd & (kStatic | kDynamic);', 'p.status |= sd & kDynamic;'),
    # ---- reference additions (lookup / extract)
    ('RR1 lookup stops at uniform nodes', 'gpu', REF, 'while (node->has_children && (node->status & status::PART))', 'while (node->has_children)'),
    ('RR2 extract root check', 'gpu', REF, 'if (!(root().status & status::PART))\n\t\t\t\treturn out;', ''),
    ('RR3 extract part rule', 'gpu', REF, 'if (sel(n.status) && (!part || at_min))', 'if (sel(n.status) && (!part))'),
    ('RR4 extract descent', 'gpu', REF, 'if (part && !at_min && n.has_children)', 'if (part && n.has_children)'),
    ('RR5 extract last level', 'gpu', REF, 'if (!last)\n\t\t\t\t\t\t\t\tvisit(level - 1, id * 8 + c);', 'if (last)\n\t\t\t\t\t\t\t\tvisit(level - 1, id * 8 + c);'),
    # ---- 64-bit sort
    ('S1 histogram word switch', 'sort', HIST64, 'return pc.shift < 32u ?', 'return pc.shift < 40u ?'),
    ('S2 histogram high shift', 'sort', HIST64, '>> (pc.shift - 32u)) & pc.mask;', '>> (pc.shift - 31u)) & pc.mask;'),
    ('S3 histogram high word', 'sort', HIST64, '(keys[2u * i + 1u] >> (pc.shift - 32u))', '(keys[2u * i] >> (pc.shift - 32u))'),
    ('S4 scatter word switch', 'sort', SCATTER64, 'digit = pc.shift < 32u ?', 'digit = pc.shift < 40u ?'),
    ('S5 scatter high word key', 'sort', SCATTER64, 'keys_out[2u * position + 1u] = keys_in[2u * i + 1u];', 'keys_out[2u * position + 1u] = keys_in[2u * i];'),
    ('S6 scatter value', 'sort', SCATTER64, 'vals_out[position] = vals_in[i];', 'vals_out[position] = 0u;'),
    ('S7 scatter low key', 'sort', SCATTER64, 'keys_out[2u * position] = keys_in[2u * i];', 'keys_out[2u * position] = keys_in[2u * i + 1u];'),
    ('S8 key words', 'sort', PRIM, 'const uint64_t key_words = wide ? 2ull * n : n;', 'const uint64_t key_words = n;'),
    ('S9 copy back size', 'sort', PRIM, 'batch.copy(*src_keys, keys, key_bytes);', 'batch.copy(*src_keys, keys, bytes);'),
    ('S10 wide pipeline', 'sort', PRIM, 'batch.dispatch(wide ? kernels_->histogram64 : kernels_->histogram,', 'batch.dispatch(kernels_->histogram,'),
    ('S11 key bits limit', 'sort', PRIM, 'key_bits > 64))', 'key_bits > 65))'),
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
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
