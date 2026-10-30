// Helpers shared by the voxel list kernels (coordinate / id helpers need no buffers).
#include "bit_voxel_common.glsl"

// kind 0: BitVoxelList (8 words per voxel, bit = meaning), kind 1: CountingVoxelList (1 word: sign extended int8 count)
const uint LIST_KIND_BIT = 0u;
const uint LIST_KIND_COUNTING = 1u;

uint words_of_kind(uint kind) { return kind == LIST_KIND_BIT ? WORDS_PER_VOXEL : 1u; }

// Voxel inside the map? Negative, too large and NaN coordinates are outside.
bool voxel_of_point(vec3 p, float side, uvec3 dim, out uvec3 c)
{
    const vec3 f = floor(p / side);
    const bool inside = f.x >= 0.0 && f.x < float(dim.x) && f.y >= 0.0 && f.y < float(dim.y) && f.z >= 0.0 && f.z < float(dim.z);
    c = inside ? uvec3(f) : uvec3(0u);
    return inside;
}

uint voxel_id(uvec3 c, uvec3 dim) { return (c.z * dim.y + c.y) * dim.x + c.x; }

uvec3 coords_of_id(uint id, uvec3 dim)
{
    const uint x = id % dim.x;
    const uint t = id / dim.x;
    return uvec3(x, t % dim.y, t / dim.y);
}
