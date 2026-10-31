// Rebuild (slice S6d-4), step 2 for the live blocks of one level 0 .. 14: block j (old address in `blocks`) is copied to the new chunk of
// this level at base + j * block size. Leaves are copied as they are (one invocation per block). An inner node keeps status, flags and
// occupancy; with PART its child pointer becomes the new address of the child block (`child_base` + link * child block size, link from step 1),
// without PART it becomes 0: the stale subtree is not copied and its blocks are released with the old chunks. For the root level the new
// chunk is the old one (in place, one invocation per node, so nothing is read after it was written).
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Blocks { uint blocks[]; };
layout(std430, set = 0, binding = 1) readonly buffer Link { uint link[]; };
layout(push_constant) uniform Push
{
    uint count;              // live blocks of this level
    uint level;
    uint64_t base;           // first block of the new chunk of this level
    uint64_t child_base;     // first block of the new chunk of the level below (unused on level 0 and without PART nodes)
    uint child_block_bytes;  // block size of the level below
} pc;

void main()
{
    const uint g = oct_gid();
    if (pc.level == 0u)
    {
        if (g >= pc.count)
            return;
        OctLeafBlock from = OctLeafBlock(packUint2x32(uvec2(blocks[2u * g], blocks[2u * g + 1u])));
        OctLeafBlock to = OctLeafBlock(pc.base + uint64_t(g) * uint64_t(OCT_LEAF_BLOCK_BYTES));
        for (uint w = 0u; w < OCT_LEAF_BLOCK_BYTES / 4u; ++w)
            to.w[w] = from.w[w];
        return;
    }
    if (g >= pc.count * 8u)
        return;
    const uint j = g >> 3u;
    const uint s = g & 7u;
    OctInnerBlock from = OctInnerBlock(packUint2x32(uvec2(blocks[2u * j], blocks[2u * j + 1u])));
    OctInnerBlock to = OctInnerBlock(pc.base + uint64_t(j) * uint64_t(OCT_INNER_BLOCK_BYTES));
    OctNode node = oct_load_inner(from, s);
    if ((node.status & OCT_PART) != 0u)
        node.child = pc.child_base + uint64_t(link[g]) * uint64_t(pc.child_block_bytes);
    else
        node.child = 0ul;
    oct_store_inner(to, s, node);
}
