// Insert, step 4: the datum of every voxel. Deterministic: a new leaf (kind 1) becomes DYNAMIC | datum (the invalid bits are replaced, CUDA
// setNode), an existing leaf (kind 0) keeps every bit except the occupancy bits and PART (CUDA updateNode with the hard reset data).
// Probabilistic: a new leaf is SET to the occupancy of the datum (127 occupied, -127 free), an existing leaf is UPDATED (old occupancy, UNKNOWN
// counts as 0, plus the datum, clamped to -127 .. 127); the status then says UNKNOWN (CUDA _updateStatus), every other bit stays.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Ids { uint ids[]; };
layout(std430, set = 0, binding = 1) readonly buffer Info { uvec4 info[]; };
layout(std430, set = 0, binding = 2) readonly buffer Heads0 { uint heads0[]; };
layout(std430, set = 0, binding = 3) readonly buffer Offsets0 { uint excl0[]; };
layout(push_constant) uniform Push
{
    uint n;
    uint datum;     // OCT_OCCUPIED or OCT_FREE
    uint64_t base0; // first block of the new leaf chunk (0 if there is none)
} pc;

void main()
{
    const uint i = oct_gid();
    if (i >= pc.n)
        return;
    const uint64_t id = packUint2x32(uvec2(ids[2u * i], ids[2u * i + 1u]));
    const uint slot = uint(id) & 7u;
    const uvec4 f = info[i];
    const bool is_new = f.w == 1u;
    uint64_t block;
    if (is_new)
        block = pc.base0 + uint64_t(excl0[i] + heads0[i] - 1u) * uint64_t(OCT_LEAF_BLOCK_BYTES);
    else
        block = oct_load_node(packUint2x32(uvec2(f.y, f.z))).child;
    OctLeafBlock leaves = OctLeafBlock(block);
#if OCT_PROB
    const int delta = pc.datum == OCT_OCCUPIED ? 127 : -127;
    OctNode leaf = oct_load_leaf(leaves, slot);
    if (is_new)
        leaf.occupancy = delta;
    else
        leaf.occupancy = clamp((leaf.occupancy == OCT_PROB_UNKNOWN ? 0 : leaf.occupancy) + delta, -127, 127);
    leaf.status = (leaf.status & ~OCT_OCCUPANCY_MASK) | OCT_UNKNOWN;
    const uint shift = (slot & 1u) * 16u;
    const uint value = (leaf.status & 255u) | ((uint(leaf.occupancy) & 255u) << 8u);
    atomicAnd(leaves.w[slot >> 1u], ~(0xFFFFu << shift));
    atomicOr(leaves.w[slot >> 1u], value << shift);
#else
    const uint shift = (slot & 3u) * 8u;
    if (is_new)
    {
        atomicAnd(leaves.w[slot >> 2u], ~(255u << shift));
        atomicOr(leaves.w[slot >> 2u], (OCT_DYNAMIC | pc.datum) << shift);
    }
    else
    {
        atomicAnd(leaves.w[slot >> 2u], ~((OCT_OCCUPANCY_MASK | OCT_PART) << shift));
        atomicOr(leaves.w[slot >> 2u], pc.datum << shift);
    }
#endif
}
