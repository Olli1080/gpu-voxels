// Insert, step 1 (slice S6d): for every voxel id (a level 0 id, sorted, distinct) the walk of the CUDA kernel
// kernel_insert_countNeededNodes: from the root while the node has PART and the level is above max(target, 1). Every node on the way and the node the
// walk ends on get NEEDS_UPDATE (SET_UPDATE_FLAG). info[i] = (level the walk ended on, address of that node low, high, kind) with kind 0 =
// the walk ended on a level 1 node with PART (the leaf exists and is updated) and kind 1 = the node has no PART (new blocks are needed
// below it) and, for a target level above 0, kind 2 = the walk ended on a node of the target level (that node is set). stats[14] counts kind 1 walks that end on a node which still owns a child block (that block becomes garbage); stats[0 .. 13] belong to the count kernel.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Ids { uint ids[]; };
layout(std430, set = 0, binding = 1) writeonly buffer Info { uvec4 info[]; };
layout(std430, set = 0, binding = 2) buffer Stats { uint stats[]; };
layout(push_constant) uniform Push
{
    uint n;
    uint target;   // target level of the insert, 0 .. 13
    uint64_t root; // block of 8 nodes of level 14; the root is node 0
} pc;

void main()
{
    const uint i = oct_gid();
    if (i >= pc.n)
        return;
    const uint64_t id = packUint2x32(uvec2(ids[2u * i], ids[2u * i + 1u]));
    uint level = OCT_ROOT_LEVEL;
    uint64_t block = pc.root;
    uint slot = 0u;
    OctNode node;
    while (true)
    {
        OctInnerBlock b = OctInnerBlock(block);
        atomicOr(b.w[OCT_NODE_WORDS * slot], OCT_NEEDS_UPDATE << 8u);
        node = oct_load_inner(b, slot);
        if (level > max(pc.target, 1u) && (node.status & OCT_PART) != 0u && node.child != 0ul)
        {
            --level;
            slot = uint(id >> (3u * level)) & 7u;
            block = node.child;
        }
        else
            break;
    }
    const bool leaf_exists = pc.target == 0u && level == 1u && (node.status & OCT_PART) != 0u && node.child != 0ul;
    const bool at_target = pc.target > 0u && level == pc.target;
    if (!leaf_exists && !at_target && node.child != 0ul)
        atomicAdd(stats[14], 1u);
    const uint64_t address = block + uint64_t(slot) * OCT_NODE_BYTES;
    info[i] = uvec4(level, uint(address & 0xFFFFFFFFul), uint(address >> 32), leaf_exists ? 0u : (at_target ? 2u : 1u));
}
