"""Mutation check of the size limit checks (K34): every boundary moved by one in both directions. test_vulkan only."""
import os
import subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SRC = os.path.join(ROOT, r"packages\gpu_voxels\gpu_voxels\vulkan\src")
EXE = os.path.join(ROOT, r"build\verify-vk\tests\vulkan\Debug\test_vulkan.exe")
FILTER = "--gtest_filter=*AcceptsTheLimit*:*BoundByTheScan*:*ExactlyTheLimitOnly*"

PRIM, OCT, LIST = "Primitives.cpp", "Octree.cpp", "VoxelList.cpp"
MUTANTS = [
    ("L1 sort limit >=", PRIM, "if (n > kMaxSortElements)", "if (n >= kMaxSortElements)"),
    ("L2 sort limit +1", PRIM, "if (n > kMaxSortElements)", "if (n > kMaxSortElements + 1)"),
    ("L3 scan limit >=", PRIM, "if (n > kMaxElements)", "if (n >= kMaxElements)"),
    ("L4 scan limit +1", PRIM, "if (n > kMaxElements)", "if (n > kMaxElements + 1)"),
    ("L5 build limit >=", OCT, "if (input.size() > kMaxVoxels)", "if (input.size() >= kMaxVoxels)"),
    ("L6 build limit +1", OCT, "if (input.size() > kMaxVoxels)", "if (input.size() > kMaxVoxels + 1)"),
    ("L7 build ids limit >=", OCT, "if (valid.size() > kMaxVoxels)", "if (valid.size() >= kMaxVoxels)"),
    ("L8 build ids limit +1", OCT, "if (valid.size() > kMaxVoxels)", "if (valid.size() > kMaxVoxels + 1)"),
    ("L9 merge limit >=", LIST, "if (total > kMaxEntries)", "if (total >= kMaxEntries)"),
    ("L10 merge limit +1", LIST, "if (total > kMaxEntries)", "if (total > kMaxEntries + 1)"),
    ("L13 map limit >=", LIST, "if (map.voxel_count() > Primitives::kMaxElements)", "if (map.voxel_count() >= Primitives::kMaxElements)"),
    ("L14 map limit +1", LIST, "if (map.voxel_count() > Primitives::kMaxElements)", "if (map.voxel_count() > Primitives::kMaxElements + 1)"),
]


def run(cmd, timeout=None):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


originals = {f: open(os.path.join(SRC, f), "rb").read() for f in (PRIM, OCT, LIST)}


def restore():
    for f, data in originals.items():
        open(os.path.join(SRC, f), "wb").write(data)


survivors = []
try:
    for name, f, a, b in MUTANTS:
        if os.environ.get('ONLY') and name.split(' ')[0] not in os.environ['ONLY'].split(','):
            continue
        data = originals[f]
        assert data.count(a.encode()) == 1, (name, data.count(a.encode()))
        open(os.path.join(SRC, f), "wb").write(data.replace(a.encode(), b.encode()))
        build = run(["cmake", "--build", os.path.join(ROOT, r"build\verify-vk"), "--config", "Debug", "--parallel", "--target", "test_vulkan"])
        if build.returncode != 0:
            print(name + ": build failed (killed by the compiler)", flush=True)
            restore()
            continue
        test = run([EXE, FILTER], timeout=400)
        if test is None:
            print(name + ": KILLED by TIMEOUT", flush=True)
        elif test.returncode != 0:
            failed = [l for l in test.stdout.splitlines() if l.startswith("[  FAILED  ]") and "." in l]
            print(name + ": KILLED by " + (failed[0][13:] if failed else "crash"), flush=True)
        else:
            print(name + ": SURVIVED  <-- weak tests", flush=True)
            survivors.append(name)
        restore()
finally:
    restore()
build = run(["cmake", "--build", os.path.join(ROOT, r"build\verify-vk"), "--config", "Debug", "--parallel", "--target", "test_vulkan"])
test = run([EXE, FILTER], timeout=400)
print("restored; build ok =", build.returncode == 0, "; limit tests pass =", test is not None and test.returncode == 0)
print("SURVIVORS:", survivors)
