// Propagate, bottom-up step for the nodes of one level 2 .. 14: items = records (address low, address high, inherit, 0) of the nodes with PART
// that were flagged or below an UPDATE_SUBTREE node, of that level. Their children are final (lower levels run first): the node is recomputed
// from them (oct_recompute, level 99: no LAST_LEVEL above level 1) and loses NEEDS_UPDATE and UPDATE_SUBTREE.
// With from_counts 1 the item count is counts[level] (written by the top-down kernels, threads beyond it return), otherwise pc.count.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Items { uint items[]; };
layout(std430, set = 0, binding = 1) readonly buffer Counts { uint counts[]; };
layout(push_constant) uniform Push { uint count; uint level; uint from_counts; uint item_capacity; } pc;

void main()
{
    const uint g = oct_gid();
    if (g >= (pc.from_counts != 0u ? min(counts[pc.level], pc.item_capacity) : pc.count))
        return;
    const uint64_t address = packUint2x32(uvec2(items[4u * g], items[4u * g + 1u]));
    OctNode node = oct_load_node(address);
    OctNode kids[8];
    for (uint k = 0u; k < 8u; ++k)
        kids[k] = oct_load_node(node.child + uint64_t(k) * OCT_NODE_BYTES);
    oct_recompute(node, kids, 99u);
    node.flags &= ~(OCT_NEEDS_UPDATE | OCT_UPDATE_SUBTREE);
    oct_store_node(address, node);
}
