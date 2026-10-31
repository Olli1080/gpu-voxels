// Insert, step 3 (levels 13 down to 0): the head voxel of every new block on `level` initialises the block like the CUDA kernel
// kernel_insert_initNeededNodes (inner nodes: DYNAMIC (| LAST_LEVEL on level 1) and NEEDS_UPDATE, probabilistic occupancy 0; leaves:
// invalid status | DYNAMIC, occupancy 0) and links it into its parent: the node of the block one level up (also new, its block index comes
// from the scan of that level) or, when the walk ended on level + 1, the node the walk ended on. The parent gets PART and the child
// address. There is exactly one head voxel per block and one block per parent node, so every node has one writer.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Ids { uint ids[]; };
layout(std430, set = 0, binding = 1) readonly buffer Info { uvec4 info[]; };
layout(std430, set = 0, binding = 2) readonly buffer Heads { uint heads[]; };
layout(std430, set = 0, binding = 3) readonly buffer Offsets { uint excl[]; };
layout(std430, set = 0, binding = 4) readonly buffer HeadsUp { uint heads_up[]; };
layout(std430, set = 0, binding = 5) readonly buffer OffsetsUp { uint excl_up[]; };
layout(push_constant) uniform Push
{
    uint n;
    uint level;
    uint64_t base;    // first block of the new chunk of this level
    uint64_t base_up; // first block of the new chunk of level + 1
} pc;

void main()
{
    const uint i = oct_gid();
    if (i >= pc.n || heads[i] == 0u)
        return;
    const uint64_t id = packUint2x32(uvec2(ids[2u * i], ids[2u * i + 1u]));
    const uint level = pc.level;
    const uint64_t block = pc.base + uint64_t(excl[i]) * uint64_t(oct_block_bytes(level));

    if (level == 0u)
    {
        OctLeafBlock leaves = OctLeafBlock(block);
#if OCT_PROB
        for (uint w = 0u; w < 4u; ++w)
            leaves.w[w] = 0x00870087u; // invalid (7) | DYNAMIC (128), occupancy 0, for each of the 8 leaves
#else
        leaves.w[0] = 0x87878787u;
        leaves.w[1] = 0x87878787u;
#endif
    }
    else
    {
        OctNode fresh;
        fresh.status = OCT_DYNAMIC | (level == 1u ? OCT_LAST_LEVEL : 0u);
        fresh.flags = OCT_NEEDS_UPDATE;
        fresh.occupancy = 0;
        fresh.child = 0ul;
        OctInnerBlock inner = OctInnerBlock(block);
        for (uint s = 0u; s < 8u; ++s)
            oct_store_inner(inner, s, fresh);
    }

    const uvec4 f = info[i];
    uint64_t parent;
    if (f.x == level + 1u)
        parent = packUint2x32(uvec2(f.y, f.z));
    else
    {
        const uint index_up = excl_up[i] + heads_up[i] - 1u;
        parent = pc.base_up + uint64_t(index_up) * uint64_t(OCT_INNER_BLOCK_BYTES) + uint64_t(uint(id >> (3u * (level + 1u))) & 7u) * OCT_NODE_BYTES;
    }
    OctNode p = oct_load_node(parent);
    p.status |= OCT_PART;
    p.child = block;
    oct_store_node(parent, p);
}
