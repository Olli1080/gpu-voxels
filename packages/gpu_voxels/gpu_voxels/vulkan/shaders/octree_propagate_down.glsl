// Propagate, top-down step for the nodes of one level (slice S6d). items = records (address low, address high, inherit, 0) of nodes with PART
// that are flagged or sit below a node with UPDATE_SUBTREE, on `level`; one invocation per (item, child). `subtree` = the record passes it down
// or the node carries the flag. A child without occupancy bits takes over from its parent (oct_expand: deterministic every bit except PART
// and LAST_LEVEL, probabilistic the occupancy), otherwise below a `subtree` node a probabilistic child is changed (oct_subtree_update); a child
// without PART loses NEEDS_UPDATE and UPDATE_SUBTREE; a child with PART that is flagged or below a `subtree` node (on level 2: every child
// with PART, level 1 nodes are always recomputed) goes into the work list of the next level, passing `subtree` on.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Items { uint items[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Next { uint next[]; };
layout(std430, set = 0, binding = 2) buffer Counter { uint counter[]; };
layout(push_constant) uniform Push { uint count; uint level; } pc;

void main()
{
    const uint g = oct_gid();
    if (g >= pc.count * 8u)
        return;
    const uint item = g >> 3u;
    const uint c = g & 7u;
    const OctNode parent = oct_load_node(packUint2x32(uvec2(items[4u * item], items[4u * item + 1u])));
    const bool subtree = items[4u * item + 2u] != 0u || (parent.flags & OCT_UPDATE_SUBTREE) != 0u;
    const uint64_t address = parent.child + uint64_t(c) * OCT_NODE_BYTES;
    OctNode child = oct_load_node(address);
    if ((child.status & OCT_OCCUPANCY_MASK) == 0u)
        oct_expand(child, parent);
    else if (subtree)
        oct_subtree_update(child, parent);
    if ((child.status & OCT_PART) == 0u)
        child.flags &= ~(OCT_NEEDS_UPDATE | OCT_UPDATE_SUBTREE);
    oct_store_node(address, child);
    if ((child.status & OCT_PART) != 0u && ((child.flags & OCT_NEEDS_UPDATE) != 0u || subtree || pc.level == 2u))
    {
        const uint index = atomicAdd(counter[0], 1u);
        next[4u * index] = uint(address & 0xFFFFFFFFul);
        next[4u * index + 1u] = uint(address >> 32);
        next[4u * index + 2u] = subtree ? 1u : 0u;
        next[4u * index + 3u] = 0u;
    }
}
