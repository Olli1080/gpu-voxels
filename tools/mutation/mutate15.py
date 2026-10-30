"""Mutation check of the block_pairs counter of the reference (tests/reference/octree_reference.h); CPU build only."""
import os
import subprocess

ROOT = os.environ.get('GVL_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
HEADER = os.path.join(ROOT, r"tests\reference\octree_reference.h")
EXE = os.path.join(ROOT, r"build\verify-cpu\tests\Debug\test_octree_reference.exe")
original = open(HEADER, "rb").read()

MUTANTS = [
    ("R1 first item", b"++r.block_pairs[kRootLevel - 1];", b""),
    ("R2 level index", b"++r.block_pairs[level - 1];", b"++r.block_pairs[level];"),
    ("R3 counted without refinement", b"if (refine)\r\n\t\t\t\t\t\t++r.block_pairs", b"if (true)\r\n\t\t\t\t\t\t++r.block_pairs"),
    ("R4 last level", b"const bool refine = !is_last && ", b"const bool refine = "),
    ("R5 min level", b"(a_part || b_part) && !end_here;\r\n\t\t\t\t\tif (refine)", b"(a_part || b_part);\r\n\t\t\t\t\tif (refine)"),
    ("R6 counts twice", b"++r.block_pairs[level - 1];", b"r.block_pairs[level - 1] += 2;"),
]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, shell=False)


survivors = []
try:
    for name, a, b in MUTANTS:
        assert original.count(a) == 1, (name, original.count(a))
        open(HEADER, "wb").write(original.replace(a, b))
        build = run(["cmake", "--build", os.path.join(ROOT, r"build\verify-cpu"), "--config", "Debug", "--parallel", "--target", "test_octree_reference"])
        if build.returncode != 0:
            print(name + ": build failed (counts as killed by the compiler)")
            continue
        test = run([EXE, "--gtest_filter=*BlockPairs*"])
        verdict = "KILLED" if test.returncode != 0 else "SURVIVED  <-- weak tests"
        print(name + ": " + verdict, flush=True)
        if test.returncode == 0:
            survivors.append(name)
finally:
    open(HEADER, "wb").write(original)
build = run(["cmake", "--build", os.path.join(ROOT, r"build\verify-cpu"), "--config", "Debug", "--parallel", "--target", "test_octree_reference"])
test = run([EXE])
print("restored; build ok =", build.returncode == 0, "; all reference tests pass =", test.returncode == 0)
print("SURVIVORS:", survivors)
