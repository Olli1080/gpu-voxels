// Collision of probe voxels with the octree (see octree_collide_det.comp / octree_collide_prob.comp, which define OCT_PROB).
// Port of kernel_intersect_VoxelMap / _VoxelList / _MortonVoxelList (octree/kernels/kernel_Octree.h): every occupied probe
// voxel (+ offset) is looked up from the root; the walk descends into PART nodes while the node is above min_level and
// the node where it stops is tested. Differences from CUDA: a probe outside the 2^14 grid after the offset is ignored (CUDA
// wraps the negative coordinate around), the unknown counter is only used when requested (CUDA reads an uninitialised flag).
//
// Input modes: 0 BitVoxelMap (payload = 8 words per voxel), 1 ProbVoxelMap (payload = 1 word, sign extended int8),
//              2 voxel list (payload = 8 words per entry, positions = 3 words coordinates),
//              3 Morton ids (positions = 2 words per id, payload = optional 8 words per id).
// options: 1 set the COLLISION bit of the node, 2 OR the payload flags of colliding probes, 4 count probes that hit unknown
//          space, 8 mode 3 only: payload holds flags (otherwise every id is an occupied probe).
// results: [0] colliding probes, [1] probes hitting unknown space, [2..9] OR of the flags.
#include "octree_nodes.glsl"
#include "morton.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Payload { uint payload[]; };
layout(std430, set = 0, binding = 1) readonly buffer Positions { uint positions[]; };
layout(std430, set = 0, binding = 2) buffer Results { uint results[]; };
layout(push_constant) uniform Push
{
    uint64_t root; // block of 8 nodes of level 14
    uint n;
    uint mode;
    uint min_level;
    uint options;
    int offset_x;
    int offset_y;
    int offset_z;
    uint dim_x;
    uint dim_y;
    int map_threshold;  // mode 1: a voxel is occupied when its value is >= this
    int tree_threshold; // probabilistic tree: a node is occupied when its occupancy is >= this (and not unknown)
} pc;

#if OCT_PROB
const uint OCT_INNER_WORDS = 3u;
#else
const uint OCT_INNER_WORDS = 2u;
#endif

bool any_flag(uint first)
{
    uint bits = 0u;
    for (uint w = 0u; w < 8u; ++w)
        bits |= payload[first + w];
    return bits != 0u;
}

void main()
{
    const uint i = oct_gid();
    if (i >= pc.n)
        return;

    // ---- is the probe occupied, where is it
    bool occupied;
    bool has_flags = false;
    switch (pc.mode)
    {
    case 0u: has_flags = true; occupied = any_flag(8u * i); break;
    case 1u: occupied = int(payload[i]) >= pc.map_threshold; break;
    case 2u: has_flags = true; occupied = any_flag(8u * i); break;
    default:
        has_flags = (pc.options & 8u) != 0u;
        occupied = !has_flags || any_flag(8u * i);
        break;
    }
    if (!occupied)
        return;

    uint64_t id;
    if (pc.mode == 3u)
    {
        id = packUint2x32(uvec2(positions[2u * i], positions[2u * i + 1u]));
        if ((id >> 42) != 0ul)
            return;
    }
    else
    {
        ivec3 c;
        if (pc.mode == 2u)
            c = ivec3(int(positions[3u * i]), int(positions[3u * i + 1u]), int(positions[3u * i + 2u]));
        else
        {
            const uint t = i / pc.dim_x;
            c = ivec3(int(i % pc.dim_x), int(t % pc.dim_y), int(t / pc.dim_y));
        }
        c += ivec3(pc.offset_x, pc.offset_y, pc.offset_z);
        if (any(lessThan(c, ivec3(0))) || any(greaterThanEqual(c, ivec3(16384))))
            return;
        id = morton_code60(uvec3(c));
    }

    // ---- look up
    uint level = OCT_ROOT_LEVEL;
    uint64_t block = pc.root;
    uint slot = 0u;
    OctNode node = oct_load_inner(OctInnerBlock(block), 0u);
    while (level > pc.min_level && (node.status & OCT_PART) != 0u && node.child != 0ul)
    {
        --level;
        slot = uint(id >> (3u * level)) & 7u;
        block = node.child;
        if (level == 0u)
            node = oct_load_leaf(OctLeafBlock(block), slot);
        else
            node = oct_load_inner(OctInnerBlock(block), slot);
    }

#if OCT_PROB
    const bool collision = node.occupancy != OCT_PROB_UNKNOWN && node.occupancy >= pc.tree_threshold;
    const bool unknown = node.occupancy == OCT_PROB_UNKNOWN;
#else
    const bool collision = (node.status & OCT_OCCUPIED) != 0u;
    const bool unknown = (node.status & OCT_UNKNOWN) != 0u;
#endif

    if ((pc.options & 4u) != 0u && unknown)
        atomicAdd(results[1], 1u);
    if (!collision)
        return;
    atomicAdd(results[0], 1u);

    if ((pc.options & 1u) != 0u)
    {
        if (level == 0u)
        {
#if OCT_PROB
            atomicOr(OctLeafBlock(block).w[slot >> 1u], OCT_COLLISION << ((slot & 1u) * 16u));
#else
            atomicOr(OctLeafBlock(block).w[slot >> 2u], OCT_COLLISION << ((slot & 3u) * 8u));
#endif
        }
        else
            atomicOr(OctInnerBlock(block).w[OCT_INNER_WORDS * slot], OCT_COLLISION);
    }
    if ((pc.options & 2u) != 0u && has_flags)
        for (uint w = 0u; w < 8u; ++w)
            atomicOr(results[2u + w], payload[8u * i + w]);
}
