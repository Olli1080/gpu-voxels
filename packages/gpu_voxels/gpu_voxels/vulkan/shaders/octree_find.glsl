// Batched lookup (see octree_find_det.comp / octree_find_prob.comp, which define OCT_PROB).
// For every Morton id walks from the root and stops at the first node that cannot be refined: a leaf, a node without
// children, or a uniform node (no PART flag: its status holds for the whole box, CUDA ignores the children it still keeps).
// result[i] = (level, status, occupancy as unsigned byte, 0); level 255 = the id lies outside the 15 level tree.
#include "octree_nodes.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Ids { uint ids[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Result { uvec4 result[]; };
layout(push_constant) uniform Push
{
    uint n;
    uint64_t root; // block of 8 nodes of level 14; the root is node 0
} pc;

void main()
{
    const uint i = oct_gid();
    if (i >= pc.n)
        return;
    const uint64_t id = packUint2x32(uvec2(ids[2u * i], ids[2u * i + 1u]));
    if ((id >> 42) != 0ul)
    {
        result[i] = uvec4(255u, 0u, 0u, 0u);
        return;
    }
    uint level = OCT_ROOT_LEVEL;
    OctNode node = oct_load_inner(OctInnerBlock(pc.root), 0u);
    while (level > 0u && (node.status & OCT_PART) != 0u && node.child != 0ul)
    {
        --level;
        const uint slot = uint(id >> (3u * level)) & 7u;
        if (level == 0u)
            node = oct_load_leaf(OctLeafBlock(node.child), slot);
        else
            node = oct_load_inner(OctInnerBlock(node.child), slot);
    }
    result[i] = uvec4(level, node.status, uint(node.occupancy) & 255u, 0u);
}
