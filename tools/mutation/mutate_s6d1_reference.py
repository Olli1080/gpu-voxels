# Hand-written mutation runner (slice S6d): one mutation at a time, build, run the tests, restore. Usage: python mutate_s6d1_reference.py [name prefix]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk or build/verify-cpu.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
path = root + 'tests/reference/octree_dynamic_reference.h'
orig = open(path, encoding='utf-8', newline='').read()
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

MUT = [
    ('M1 walk does not flag the visited nodes',
     "				nodes_.at({ level, id }).needs_update = true;\n				--level;",
     "				--level;"),
    ('M2 end node of the walk not flagged',
     "			nodes_.at({ level, id }).needs_update = true;\n			if (level == 1 && part(1, id))",
     "			if (level == 1 && part(1, id))"),
    ('M3 new leaves without the invalid bits',
     "child.status = status::OCCUPANCY_MASK | status::DYNAMIC;",
     "child.status = status::DYNAMIC;"),
    ('M4 LAST_LEVEL on the wrong level',
     "(l == 2 ? status::LAST_LEVEL : 0)",
     "(l == 3 ? status::LAST_LEVEL : 0)"),
    ('M5 garbage not counted',
     "			++garbage_[level - 1];\n",
     ""),
    ('M6 top-down copy takes the PART bit too',
     "(parent.status & ~(status::PART | status::LAST_LEVEL)));\n				if (!(child.status & status::PART))",
     "(parent.status & ~status::LAST_LEVEL));\n				if (!(child.status & status::PART))"),
    ('M7 top-down only for children WITH occupancy bits',
     "if ((child.status & status::OCCUPANCY_MASK) == 0)\n					child.status",
     "if ((child.status & status::OCCUPANCY_MASK) != 0)\n					child.status"),
    ('M8 flag of a child without PART not cleared',
     "				if (!(child.status & status::PART))\n					child.needs_update = false;",
     ""),
    ('M9 unflagged level 1 nodes not recomputed',
     "if ((kids[s]->status & status::PART) && !kids[s]->needs_update)",
     "if (false)"),
    ('M10 invalid leaves detected with a wrong test',
     "if ((leaf.status & status::OCCUPANCY_MASK) == status::OCCUPANCY_MASK)",
     "if ((leaf.status & status::OCCUPANCY_MASK) == 0)"),
    ('M11 LAST_LEVEL added above level 1',
     "parent.status = new_status(or_status, and_status, 99, or_status);",
     "parent.status = new_status(or_status, and_status, 1, or_status);"),
    ('M12 parent keeps NEEDS_UPDATE',
     "			parent.needs_update = false;\n		}\n\n		/// A level 1 node",
     "		}\n\n		/// A level 1 node"),
    ('M13 owner does not get PART',
     "				owner.status |= status::PART;",
     ""),
    ('M14 stale subtree not dropped recursively',
     "				if (level > 1)\n					drop_children(level - 1, child);",
     ""),
    ('M15 existing leaf: datum ORed instead of replacing the occupancy bits',
     "leaf.status = uint8_t((leaf.status & ~(status::OCCUPANCY_MASK | status::PART)) | set);\n				return;",
     "leaf.status = uint8_t(leaf.status | set);\n				return;"),
    ('R1 UNKNOWN leaf counted as -128 when updated', "leaf.occupancy == kUnknownOccupancy ? 0 : leaf.occupancy", "leaf.occupancy"),
    ('R2 clamp at -128', "std::max(old_value + datum_value(datum), -127)", "std::max(old_value + datum_value(datum), -128)"),
    ('R3 existing leaf set instead of updated', "leaf.occupancy = int8_t(std::min(std::max(old_value + datum_value(datum), -127), 127));", "leaf.occupancy = int8_t(datum_value(datum));"),
    ('R4 free datum -128', "return datum == Datum::Occupied ? 127 : -127;", "return datum == Datum::Occupied ? 127 : -128;"),
    ('R5 expansion does not copy the occupancy', "\t\t\t\tchild.occupancy = parent.occupancy;\n", ""),
    ('R6 parent takes the minimum', "highest = std::max(highest, k->occupancy);", "highest = std::min(highest, k->occupancy);"),
    ('R7 same ignores PART children', "same = same && k->occupancy == highest && !(k->status & status::PART);", "same = same && k->occupancy == highest;"),
    ('R8 parent always PART', "| (same ? 0 : status::PART) | status::UNKNOWN);", "| status::PART | status::UNKNOWN);"),
    ('R9 parent status without UNKNOWN', "| (same ? 0 : status::PART) | status::UNKNOWN);", "| (same ? 0 : status::PART));"),
    ('R10 new leaf not set', "leaf.occupancy = int8_t(datum_value(datum));\n\t\t\t\tleaf.status", "leaf.occupancy = 0;\n\t\t\t\tleaf.status"),
]

only = sys.argv[1] if len(sys.argv) > 1 else ''
results = []
try:
    for name, old, new in MUT:
        if only and not name.startswith(only):
            continue
        text = orig.replace('\r\n', '\n')
        if text.count(old) != 1:
            print(name, ': PATTERN COUNT', text.count(old), flush=True)
            continue
        nl = '\r\n' if '\r\n' in orig else '\n'
        open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new).replace('\n', nl))
        b = subprocess.run(['cmake', '--build', root + 'build/verify-cpu', '--config', 'Debug', '--target', 'test_octree_reference', '--parallel'],
                           env=env, capture_output=True, text=True)
        if b.returncode != 0:
            print(name, ': BUILD FAILED', flush=True)
            continue
        try:
            r = subprocess.run([root + 'build/verify-cpu/tests/Debug/test_octree_reference.exe', '--gtest_filter=DynamicOctreeReference*'],
                               env=env, capture_output=True, text=True, timeout=120)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
finally:
    open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
