// Propagate for the nodes of level 1: items = addresses of level 1 nodes with PART. Invalid leaves (all occupancy bits set) take over from
// the node (oct_expand), then the node is recomputed from its 8 leaves (oct_recompute, level 1) and loses NEEDS_UPDATE. One invocation per
// node: it owns the whole leaf block.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Items { uint items[]; };
layout(push_constant) uniform Push { uint count; } pc;

void main()
{
    const uint g = oct_gid();
    if (g >= pc.count)
        return;
    const uint64_t address = packUint2x32(uvec2(items[2u * g], items[2u * g + 1u]));
    OctNode node = oct_load_node(address);
    OctLeafBlock leaves = OctLeafBlock(node.child);
    OctNode kids[8];
    for (uint k = 0u; k < 8u; ++k)
    {
        kids[k] = oct_load_leaf(leaves, k);
        if ((kids[k].status & OCT_OCCUPANCY_MASK) == OCT_OCCUPANCY_MASK)
            oct_expand(kids[k], node);
    }
    oct_store_leaf_block(leaves, kids);
    oct_recompute(node, kids, 1u);
    node.flags &= ~OCT_NEEDS_UPDATE;
    oct_store_node(address, node);
}
