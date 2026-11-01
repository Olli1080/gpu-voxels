// Garbage count (slice S6d-6), one step for the blocks of one level 1 .. 14: `blocks` = addresses (low, high word) of the reachable inner blocks of
// this level, their number is counter[level] (clamped to `item_capacity`); one invocation per (block, slot). Every node with a child pointer
// (PART or not: the pointer of a uniform node still keeps its block reachable, as in the host walk of check_tree) appends the address of its child
// block to `next` and counts it in counter[level - 1]. Allocated blocks that no pointer reaches are the garbage. `next_capacity` = entries `next`
// can take (0 for level 1: leaf blocks are only counted); the counter keeps counting past it, so a block referenced twice (a corrupt tree) shows
// as more blocks than were allocated instead of an out-of-bounds write.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Blocks { uint blocks[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Next { uint next[]; };
layout(std430, set = 0, binding = 2) buffer Counter { uint counter[]; };
layout(push_constant) uniform Push { uint level; uint item_capacity; uint next_capacity; } pc;

void main()
{
    const uint count = min(counter[pc.level], pc.item_capacity);
    const uint g = oct_gid();
    if (g >= count * 8u)
        return;
    const uint j = g >> 3u;
    const uint s = g & 7u;
    OctInnerBlock block = OctInnerBlock(packUint2x32(uvec2(blocks[2u * j], blocks[2u * j + 1u])));
    const OctNode node = oct_load_inner(block, s);
    if (node.child != 0ul)
    {
        const uint index = atomicAdd(counter[pc.level - 1u], 1u);
        if (index < pc.next_capacity)
        {
            next[2u * index] = uint(node.child & 0xFFFFFFFFul);
            next[2u * index + 1u] = uint(node.child >> 32);
        }
    }
}
