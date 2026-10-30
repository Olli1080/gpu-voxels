// Tree-vs-tree collision on the load balancing framework (see octree_lb_trees_det.comp / _prob.comp, which define OCT_PROB).
// Port of load_balancer/kernel_config/Intersect.cuh doLoadBalancedWork; the per-pair rules are those of octree_collide_trees.glsl
// (the level-synchronous variant) and the two must give identical results. A work item is the pair of blocks of 8 sibling nodes
// of one level (word 4: bit 0 a_active, bit 1 b_active, bits 8..15 level); one thread per (item, lane) handles one node pair.
// counters[1 + level] = colliding pairs counted at that level (the host weights them with 8^level).
#include "octree_nodes.glsl"

layout(std430, set = 0, binding = 3) buffer Counters { uint counters[]; };
layout(push_constant) uniform Push
{
    uint stack_size;
    uint stack_threshold;
    uint idle_threshold;
    uint min_level;
    uint mark;
    int tree_threshold;
} pc;
#include "octree_lb.glsl"

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
    const uint tid = gl_LocalInvocationID.x;
    if (tid == 0u)
        lb_num = stack_counts[gl_WorkGroupID.x];
    barrier();

    while (true)
    {
        if (lb_should_stop())
            break;
        const uint items = lb_pop();
        const uint item = tid / LB_LANES;
        const uint lane = tid % LB_LANES;

        bool insert = false;
        uint64_t a_next = 0ul;
        uint64_t b_next = 0ul;
        uint next_flags = 0u;
        if (item < items)
        {
            const uint first = item * LB_WORDS;
            const uint64_t a_base = packUint2x32(uvec2(lb_cache[first], lb_cache[first + 1u]));
            const uint64_t b_base = packUint2x32(uvec2(lb_cache[first + 2u], lb_cache[first + 3u]));
            const uint flags = lb_cache[first + 4u];
            const uint level = (flags >> 8u) & 255u;
            const bool a_active = (flags & 1u) != 0u;
            const bool b_active = (flags & 2u) != 0u;
            const uint64_t a_address = a_active ? a_base + uint64_t(lane * OCT_NODE_BYTES) : a_base;
            const uint64_t b_address = b_active ? b_base + uint64_t(lane * OCT_NODE_BYTES) : b_base;

            const OctNode a = oct_load_inner(OctInnerBlock(a_address), 0u);
            const OctNode b = oct_load_inner(OctInnerBlock(b_address), 0u);
            if (node_occupied(a) && node_occupied(b))
            {
                const bool end_here = pc.min_level >= level;
                // a node with the PART flag but without a child block (corrupt tree) is handled as a uniform node
                const bool a_part = (a.status & OCT_PART) != 0u && a.child != 0ul;
                const bool b_part = (b.status & OCT_PART) != 0u && b.child != 0ul;
                const bool is_last_level = ((a.status | b.status) & OCT_LAST_LEVEL) != 0u;

                if ((!a_part && !b_part) || end_here)
                {
                    atomicAdd(counters[1u + level], 1u);
                    if (pc.mark != 0u)
                        atomicOr(OctInnerBlock(a_active ? a_address : b_address).w[0], OCT_COLLISION);
                }
                if (!is_last_level && (a_part || b_part) && !end_here)
                {
                    insert = true;
                    a_next = a_part ? a.child : a_address;
                    b_next = b_part ? b.child : b_address;
                    next_flags = (a_part ? 1u : 0u) | (b_part ? 2u : 0u) | ((level - 1u) << 8u);
                }
                if (level == 1u && pc.min_level == 0u && (a_part || b_part))
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
        }

        uint total;
        const uint exclusive = lb_prefix(insert, total);
        if (insert)
        {
            const uint at = lb_base() + (lb_num + exclusive) * LB_WORDS;
            stacks[at] = uint(a_next & 0xFFFFFFFFul);
            stacks[at + 1u] = uint(a_next >> 32);
            stacks[at + 2u] = uint(b_next & 0xFFFFFFFFul);
            stacks[at + 3u] = uint(b_next >> 32);
            stacks[at + 4u] = next_flags;
            stacks[at + 5u] = 0u;
        }
        lb_commit(total);
    }
    if (tid == 0u)
        stack_counts[gl_WorkGroupID.x] = lb_num;
}
