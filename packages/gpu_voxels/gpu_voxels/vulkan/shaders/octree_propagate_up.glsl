// Propagate, bottom-up step for the nodes of one level 2 .. 14: items = addresses of the nodes with PART and NEEDS_UPDATE of that level.
// Their children are final (lower levels run first): the node is recomputed from them (oct_recompute, level 99: no LAST_LEVEL above level 1)
// and loses NEEDS_UPDATE.
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
    OctNode kids[8];
    for (uint k = 0u; k < 8u; ++k)
        kids[k] = oct_load_node(node.child + uint64_t(k) * OCT_NODE_BYTES);
    oct_recompute(node, kids, 99u);
    node.flags &= ~OCT_NEEDS_UPDATE;
    oct_store_node(address, node);
}
