// One level of the tree-vs-tree collision (see octree_collide_trees_det.comp / _prob.comp, which define OCT_PROB).
// Port of the work function of load_balancer/kernel_config/Intersect.cuh as a level-synchronous wavefront: a work item is a
// pair of blocks of 8 sibling nodes of `level` (one per tree) that cover the same space; a side whose node is uniform (no PART
// flag) is represented by that node itself (a_active = 0: "the children are the node again"). One invocation per (item, lane):
//  * the pair collides when both nodes are occupied (deterministic: OCCUPIED bit, probabilistic: known and >= threshold);
//  * a colliding pair counts as the whole box (8^level voxels) when both nodes are uniform or `level <= min_level`
//    (counts[1 + level] is incremented, the host weights the levels), and marks the real node of A (else of B);
//  * a colliding pair with a PART node above min_level is refined (next work item) except at level 1, where the leaves are
//    compared one by one (min_level 0): a leaf pair counts 1 and marks the leaf of A (or of B when only B has leaves).
// counters[0] = number of items written to next_items, counters[1 + level] = number of colliding pairs counted at that level.
#include "octree_nodes.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Items { uint items[]; };       // 6 words: a address lo/hi, b address lo/hi, flags (bit 0 a_active, bit 1 b_active), 0
layout(std430, set = 0, binding = 1) writeonly buffer NextItems { uint next_items[]; };
layout(std430, set = 0, binding = 2) buffer Counters { uint counters[]; };
layout(push_constant) uniform Push
{
    uint item_count;
    uint level;
    uint min_level;
    uint mark;
    int tree_threshold;
    uint next_capacity;
} pc;

#if OCT_PROB
const uint OCT_NODE_BYTES = 12u;
#else
const uint OCT_NODE_BYTES = 8u;
#endif

bool node_occupied(OctNode n)
{
#if OCT_PROB
    return n.occupancy != OCT_PROB_UNKNOWN && n.occupancy >= pc.tree_threshold;
#else
    return (n.status & OCT_OCCUPIED) != 0u;
#endif
}

void mark_leaf(uint64_t leaf_block, uint slot)
{
#if OCT_PROB
    atomicOr(OctLeafBlock(leaf_block).w[slot >> 1u], OCT_COLLISION << ((slot & 1u) * 16u));
#else
    atomicOr(OctLeafBlock(leaf_block).w[slot >> 2u], OCT_COLLISION << ((slot & 3u) * 8u));
#endif
}

void main()
{
    const uint t = oct_gid();
    if (t >= pc.item_count * 8u)
        return;
    const uint item = t >> 3u;
    const uint lane = t & 7u;
    const uint64_t a_base = packUint2x32(uvec2(items[6u * item], items[6u * item + 1u]));
    const uint64_t b_base = packUint2x32(uvec2(items[6u * item + 2u], items[6u * item + 3u]));
    const uint flags = items[6u * item + 4u];
    const bool a_active = (flags & 1u) != 0u;
    const bool b_active = (flags & 2u) != 0u;
    const uint64_t a_address = a_active ? a_base + uint64_t(lane * OCT_NODE_BYTES) : a_base;
    const uint64_t b_address = b_active ? b_base + uint64_t(lane * OCT_NODE_BYTES) : b_base;

    const OctNode a = oct_load_inner(OctInnerBlock(a_address), 0u);
    const OctNode b = oct_load_inner(OctInnerBlock(b_address), 0u);
    const bool collide = node_occupied(a) && node_occupied(b);
    if (!collide)
        return;

    const bool end_here = pc.min_level >= pc.level;
    // a node with the PART flag but without a child block (corrupt tree) is handled as a uniform node: no null dereference.
    // A uniform node never has the flag, so PART also implies that the side is a real block, not a stand-in.
    const bool a_part = (a.status & OCT_PART) != 0u && a.child != 0ul;
    const bool b_part = (b.status & OCT_PART) != 0u && b.child != 0ul;
    const bool is_last_level = ((a.status | b.status) & OCT_LAST_LEVEL) != 0u;

    if ((!a_part && !b_part) || end_here)
    {
        atomicAdd(counters[1u + pc.level], 1u);
        if (pc.mark != 0u)
            atomicOr(OctInnerBlock(a_active ? a_address : b_address).w[0], OCT_COLLISION);
    }

    if (!is_last_level && (a_part || b_part) && !end_here)
    {
        const uint64_t a_next = a_part ? a.child : a_address;
        const uint64_t b_next = b_part ? b.child : b_address;
        const uint index = atomicAdd(counters[0], 1u);
        if (index < pc.next_capacity)
        {
            next_items[6u * index] = uint(a_next & 0xFFFFFFFFul);
            next_items[6u * index + 1u] = uint(a_next >> 32);
            next_items[6u * index + 2u] = uint(b_next & 0xFFFFFFFFul);
            next_items[6u * index + 3u] = uint(b_next >> 32);
            next_items[6u * index + 4u] = (a_part ? 1u : 0u) | (b_part ? 2u : 0u);
            next_items[6u * index + 5u] = 0u;
        }
    }

    if (pc.level == 1u && pc.min_level == 0u && (a_part || b_part))
    {
        for (uint j = 0u; j < 8u; ++j)
        {
            if (a_part && b_part)
            {
                if (node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)) && node_occupied(oct_load_leaf(OctLeafBlock(b.child), j)))
                {
                    atomicAdd(counters[1], 1u);
                    if (pc.mark != 0u)
                        mark_leaf(a.child, j);
                }
            }
            else if (a_part)
            {
                if (node_occupied(oct_load_leaf(OctLeafBlock(a.child), j)))
                {
                    atomicAdd(counters[1], 1u);
                    if (pc.mark != 0u)
                        mark_leaf(a.child, j);
                }
            }
            else if (node_occupied(oct_load_leaf(OctLeafBlock(b.child), j)))
            {
                atomicAdd(counters[1], 1u);
                if (pc.mark != 0u)
                    mark_leaf(b.child, j);
            }
        }
    }
}
