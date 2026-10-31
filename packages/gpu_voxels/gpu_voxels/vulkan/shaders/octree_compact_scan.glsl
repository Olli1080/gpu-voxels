// Rebuild (slice S6d-4), step 1 for the live blocks of one level 1 .. 14: blocks = old addresses (low, high word) of the live inner blocks of
// this level; one invocation per (block, slot). A node with PART owns a live child block: its address is appended to `next` (the live
// blocks of the level below, in the order the atomic counter hands out) and the index is remembered in `link` (the new position of that child
// block, so step 2 can write the new pointer); every other node gets 0xFFFFFFFF (a node without PART is a leaf of the tree: whatever its
// child pointer says is stale and dies with the rebuild). `capacity` = entries `next` can take; the counter keeps counting past it so the host
// sees a corrupt tree (a block referenced twice) instead of an out-of-bounds write.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Blocks { uint blocks[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Next { uint next[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Link { uint link[]; };
layout(std430, set = 0, binding = 3) buffer Counter { uint counter[]; };
layout(push_constant) uniform Push { uint count; uint capacity; } pc;

void main()
{
    const uint g = oct_gid();
    if (g >= pc.count * 8u)
        return;
    const uint j = g >> 3u;
    const uint s = g & 7u;
    OctInnerBlock block = OctInnerBlock(packUint2x32(uvec2(blocks[2u * j], blocks[2u * j + 1u])));
    const OctNode node = oct_load_inner(block, s);
    uint index = 0xFFFFFFFFu;
    if ((node.status & OCT_PART) != 0u)
    {
        index = atomicAdd(counter[0], 1u);
        if (index < pc.capacity)
        {
            next[2u * index] = uint(node.child & 0xFFFFFFFFul);
            next[2u * index + 1u] = uint(node.child >> 32);
        }
    }
    link[g] = index;
}
