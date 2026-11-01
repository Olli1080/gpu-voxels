// Propagate for the nodes of level 1: items = records (address low, address high, inherit, 0) of level 1 nodes with PART. Invalid leaves (all
// occupancy bits set) take over from the node (oct_expand); below a node with UPDATE_SUBTREE (the record passes it down or the node carries the
// flag) the probabilistic leaves are UPDATED with the occupancy of the node (clamped; an invalid leaf therefore gets it twice, as in CUDA);
// then the node is recomputed from its 8 leaves (oct_recompute, level 1) and loses NEEDS_UPDATE and UPDATE_SUBTREE. One invocation per node:
// it owns the whole leaf block.
// With from_counts 1 the item count is counts[1] (written by the top-down kernel of level 2, threads beyond it return), otherwise pc.count.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Items { uint items[]; };
layout(std430, set = 0, binding = 1) readonly buffer Counts { uint counts[]; };
layout(push_constant) uniform Push { uint count; uint from_counts; uint item_capacity; } pc;

void main()
{
    const uint g = oct_gid();
    if (g >= (pc.from_counts != 0u ? min(counts[1], pc.item_capacity) : pc.count))
        return;
    const uint64_t address = packUint2x32(uvec2(items[4u * g], items[4u * g + 1u]));
    OctNode node = oct_load_node(address);
    const bool subtree = items[4u * g + 2u] != 0u || (node.flags & OCT_UPDATE_SUBTREE) != 0u;
    OctLeafBlock leaves = OctLeafBlock(node.child);
    OctNode kids[8];
    for (uint k = 0u; k < 8u; ++k)
    {
        kids[k] = oct_load_leaf(leaves, k);
        if ((kids[k].status & OCT_OCCUPANCY_MASK) == OCT_OCCUPANCY_MASK)
            oct_expand(kids[k], node);
#if OCT_PROB
        if (subtree)
            kids[k].occupancy = clamp(kids[k].occupancy + node.occupancy, -127, 127);
#endif
    }
    oct_store_leaf_block(leaves, kids);
    oct_recompute(node, kids, 1u);
    node.flags &= ~(OCT_NEEDS_UPDATE | OCT_UPDATE_SUBTREE);
    oct_store_node(address, node);
}
