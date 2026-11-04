"""Checks the CUDA sources and a finished CUDA build for the K41 defect: a __global__ template whose host function is a placeholder.

Background (Agent/Analysis/2026-11-03_cuda_parity.md, section "K41"): for every __global__ template K, nvcc generates a function template
__wrapper__device_stub_K<...>(params). Its primary template (built from the *definition* of K) is the placeholder { cudaLaunchKernel(0, 0, 0, 0, 0, 0); },
the real launch code arrives as an explicit specialization (built from the *declaration* of K). If the declaration and the definition differ in the
top-level const of a by-value parameter, the host function of K binds to the placeholder: every launch fails with "invalid device function".

  python tools/cuda_stub_check.py sources            # declaration vs definition of every __global__ function under packages/ (no build needed)
  python tools/cuda_stub_check.py objects [builddir] # scans the *.cu.obj files of a CUDA build (default build/cuda-rel) for placeholder host functions

Exit code 1 if a problem was found. Plain Python 3, no third-party packages. The object scan understands /bigobj COFF files (MSVC).
"""
import glob
import os
import re
import struct
import sys

ROOT = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))


# ---------------------------------------------------------------- sources

def _split_params(s):
    out, depth, cur = [], 0, ''
    for ch in s:
        if ch in '<(':
            depth += 1
        elif ch in '>)':
            depth -= 1
        if ch == ',' and depth == 0:
            out.append(cur)
            cur = ''
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [' '.join(p.split()) for p in out]


def _strip_comments(t):
    t = re.sub(r'//[^\n]*', '', t)
    return re.sub(r'/\*.*?\*/', '', t, flags=re.S)


def _param_type(p):
    """The parameter without its name and without a top-level const of a by-value parameter ('const uint32_t n' -> 'uint32_t')."""
    m = re.match(r'^(.*?[\s\*&>])\s*\w+$', p)
    if m:
        p = m.group(1)
    if '*' not in p and '&' not in p:
        p = re.sub(r'^const\s+', '', p)
    return p.strip()


def check_sources():
    files = [f for ext in ('h', 'hpp', 'cuhpp', 'cu', 'cuh') for f in glob.glob(os.path.join(ROOT, 'packages', '**', '*.' + ext), recursive=True)]
    decls = {}
    pattern = r'__global__\s+(?:__launch_bounds__\([^)]*\)\s+)?(?:void\s+)?(\w+)\s*\(([^)]*(?:\([^)]*\)[^)]*)*)\)\s*([;{])'
    for f in files:
        text = _strip_comments(open(f, encoding='utf-8', errors='replace').read())
        for m in re.finditer(pattern, text, flags=re.S):
            # only templates get the wrapper templates of nvcc; a plain __global__ function is not affected
            if re.search(r'template\s*<[^;{}]*>\s*$', text[max(0, m.start() - 200):m.start()]):
                decls.setdefault(m.group(1), []).append((f, _split_params(m.group(2)), m.group(3)))
    problems = 0
    for name, items in sorted(decls.items()):
        for count in sorted({len(p) for _, p, _ in items}):
            group = [(f, p, e) for f, p, e in items if len(p) == count]
            # group the overloads by their parameter types (without names and without top-level const of by-value parameters)
            by_types = {}
            for f, p, e in group:
                by_types.setdefault(tuple(_param_type(x) for x in p), []).append((f, p, e))
            for types, members in by_types.items():
                consts = {tuple(bool(re.match(r'^const\s+', x)) and '*' not in x and '&' not in x for x in p) for _, p, _ in members}
                if len(consts) > 1 and any(e == ';' for _, _, e in members) and any(e == '{' for _, _, e in members):
                    problems += 1
                    print('%s (%d parameters): declaration and definition differ in the top-level const of a by-value parameter' % (name, count))
                    for f, p, e in members:
                        print('    %s %s: %s' % (os.path.relpath(f, ROOT), 'decl' if e == ';' else 'def ', ', '.join(p)[:200]))
    print('sources: %d __global__ names checked, %d with a const mismatch' % (len(decls), problems))
    return problems


# ---------------------------------------------------------------- objects

def _parse_bigobj(path):
    d = open(path, 'rb').read()
    sig1, sig2 = struct.unpack_from('<HH', d, 0)
    if not (sig1 == 0 and sig2 == 0xFFFF):
        raise ValueError('%s is not a /bigobj COFF file' % path)
    (nsec,) = struct.unpack_from('<I', d, 44)
    symptr, nsym = struct.unpack_from('<II', d, 48)
    secs = []
    for i in range(nsec):
        o = 56 + 40 * i
        size, rawptr, relptr = struct.unpack_from('<III', d, o + 16)
        (nreloc,) = struct.unpack_from('<H', d, o + 32)
        secs.append((size, rawptr, nreloc))
    strtab = symptr + 20 * nsym
    syms, i = [], 0
    while i < nsym:
        o = symptr + 20 * i
        nm = d[o:o + 8]
        if nm[:4] == b'\0\0\0\0':
            (off,) = struct.unpack_from('<I', nm, 4)
            name = d[strtab + off:d.index(b'\0', strtab + off)].decode('latin-1')
        else:
            name = nm.rstrip(b'\0').decode('latin-1')
        value, secnum, typ, sclass, naux = struct.unpack_from('<IiHBB', d, o + 8)
        syms.append((name, secnum, typ, sclass))
        i += 1 + naux
    return d, secs, syms


def placeholders_of(path):
    """(real, placeholders): the external host functions of the __global__ templates (??$kernel...) in one object file.
    A placeholder is recognised by its code: xor ecx,ecx and xor r9d,r9d (func = NULL, args = NULL) in a function below 256 bytes whose only
    relocation is the call of cudaLaunchKernel (a real launch stub loads its own address instead; a host launcher has more relocations)."""
    d, secs, syms = _parse_bigobj(path)
    real, bad = 0, []
    for name, secnum, typ, sclass in syms:
        if secnum > 0 and typ == 0x20 and sclass == 2 and name.startswith('??$kernel'):
            size, raw, nreloc = secs[secnum - 1]
            body = d[raw:raw + size]
            if b'\x33\xc9' in body and b'\x45\x33\xc9' in body and size < 0x100 and nreloc == 1:
                bad.append(name)
            else:
                real += 1
    return real, bad


def check_objects(build):
    objs = sorted(glob.glob(os.path.join(build, '**', '*.cu.obj'), recursive=True))
    if not objs:
        print('no *.cu.obj files under', build)
        return 1
    total = 0
    for o in objs:
        real, bad = placeholders_of(o)
        total += len(bad)
        if bad:
            print('%s: %d real, %d placeholder' % (os.path.relpath(o, build), real, len(bad)))
            for n in bad:
                print('    ', n[:160])
    print('objects: %d files, %d placeholder host functions' % (len(objs), total))
    return total


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'sources'
    if mode == 'sources':
        sys.exit(1 if check_sources() else 0)
    if mode == 'objects':
        sys.exit(1 if check_objects(sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'build', 'cuda-rel')) else 0)
    sys.exit('usage: cuda_stub_check.py sources | objects [builddir]')
