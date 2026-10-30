// Morton (Z-order) codes of the octree; mirrors include/gpu_voxels/vulkan/Morton.h bit for bit.
// The including shader must enable GL_ARB_gpu_shader_int64 (uint64_t) before the #include.
// x is bit 0, y bit 1, z bit 2 of every triple; 20 bits per axis (higher bits are ignored).

uint morton_spread10(uint x)
{
    x &= 1023u;
    x = (x | (x << 16)) & 0x030000FFu;
    x = (x | (x << 8)) & 0x0300F00Fu;
    x = (x | (x << 4)) & 0x030C30C3u;
    x = (x | (x << 2)) & 0x09249249u;
    return x;
}

uint morton_compact10(uint x)
{
    x &= 0x09249249u;
    x = (x ^ (x >> 2)) & 0x030C30C3u;
    x = (x ^ (x >> 4)) & 0x0300F00Fu;
    x = (x ^ (x >> 8)) & 0xFF0000FFu;
    x = (x ^ (x >> 16)) & 0x000003FFu;
    return x;
}

uint morton_code30(uvec3 c) { return morton_spread10(c.x) | (morton_spread10(c.y) << 1) | (morton_spread10(c.z) << 2); }

uint64_t morton_code60(uvec3 c)
{
    return (uint64_t(morton_code30(c >> 10u)) << 30) | uint64_t(morton_code30(c));
}

uvec3 morton_decode60(uint64_t code)
{
    const uint lo = uint(code);        // morton_compact10 only looks at bits 0..29: higher bits are ignored
    const uint hi = uint(code >> 30);
    return uvec3((morton_compact10(hi) << 10) | morton_compact10(lo),
                 (morton_compact10(hi >> 1) << 10) | morton_compact10(lo >> 1),
                 (morton_compact10(hi >> 2) << 10) | morton_compact10(lo >> 2));
}

// Node id of the ancestor of voxel `id` at level + 1 (the ancestor at `level` has id `id >> 3*level`).
uint64_t morton_prefix(uint64_t id, uint level) { return id >> (3u * (level + 1u)); }

// Index (0..7) of the ancestor of voxel `id` at `level` among its siblings.
uint morton_node_index(uint64_t id, uint level) { return uint(id >> (3u * level)) & 7u; }

// Number of levels one has to climb until the paths of two voxels meet (0 when equal).
uint morton_common_level(uint64_t a, uint64_t b)
{
    const uint64_t x = a ^ b;
    const uint hi = uint(x >> 32);
    const uint lo = uint(x);
    const uint width = hi != 0u ? 33u + uint(findMSB(hi)) : (lo != 0u ? 1u + uint(findMSB(lo)) : 0u);
    return (width + 2u) / 3u;
}
