// Propagate, top-down step for the nodes of one level (slice S6d). items = addresses of nodes with PART and NEEDS_UPDATE on `level`; one
// invocation per (item, child). A child without occupancy bits takes over from its parent (oct_expand: deterministic every bit except PART
// and LAST_LEVEL, probabilistic the occupancy), a child without PART loses NEEDS_UPDATE, a child with PART that is flagged (on level 2:
// every child with PART, level 1 nodes are always recomputed) goes into the work list of the next level.
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
    const OctNode parent = oct_load_node(packUint2x32(uvec2(items[2u * item], items[2u * item + 1u])));
    const uint64_t address = parent.child + uint64_t(c) * OCT_NODE_BYTES;
    OctNode child = oct_load_node(address);
    if ((child.status & OCT_OCCUPANCY_MASK) == 0u)
        oct_expand(child, parent);
    if ((child.status & OCT_PART) == 0u)
        child.flags &= ~OCT_NEEDS_UPDATE;
    oct_store_node(address, child);
    if ((child.status & OCT_PART) != 0u && ((child.flags & OCT_NEEDS_UPDATE) != 0u || pc.level == 2u))
    {
        const uint index = atomicAdd(counter[0], 1u);
        next[2u * index] = uint(address & 0xFFFFFFFFul);
        next[2u * index + 1u] = uint(address >> 32);
    }
}
