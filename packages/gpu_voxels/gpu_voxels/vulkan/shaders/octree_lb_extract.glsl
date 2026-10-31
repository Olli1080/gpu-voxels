// Extraction on the load balancing framework (see octree_lb_extract_det.comp / _prob.comp, which define OCT_PROB).
// Port of load_balancer/Extract.cuh + kernel_config/Extract.cuh with the rules of octree_extract.glsl (the level-synchronous
// variant); the two must give identical results. A work item is a block of 8 sibling nodes of one level >= 1
// (words 0/1 address, 2/3 id of the block, word 4: level in bits 8..15). counters[1] = number of emitted nodes (counted always,
// written only when the index is below out_capacity: the counting pass has capacity 0).
#include "octree_nodes.glsl"

layout(std430, set = 0, binding = 3) buffer Out { uint out_nodes[]; };    // 4 words: id low/high, level | status << 8 | occupancy << 16, 0
layout(std430, set = 0, binding = 4) buffer Counters { uint counters[]; };
layout(std430, set = 0, binding = 5) readonly buffer Selection { uint selection[]; };
layout(push_constant) uniform Push
{
    uint stack_size;
    uint stack_threshold;
    uint idle_threshold;
    uint min_level;
    uint out_capacity;
} pc;
#include "octree_lb.glsl"

void emit(uint64_t id, uint level, uint status, int occupancy)
{
    const uint index = atomicAdd(counters[1], 1u);
    if (index < pc.out_capacity)
    {
        out_nodes[4u * index] = uint(id & 0xFFFFFFFFul);
        out_nodes[4u * index + 1u] = uint(id >> 32);
        out_nodes[4u * index + 2u] = level | (status << 8u) | ((uint(occupancy) & 255u) << 16u);
        out_nodes[4u * index + 3u] = 0u;
    }
}

void main()
{
    const uint tid = gl_LocalInvocationID.x;
    if (tid == 0u)
    {
        lb_num = stack_counts[gl_WorkGroupID.x];
        lb_pops = 0u;
    }
    barrier();

    while (true)
    {
        if (lb_should_stop())
            break;
        const uint items = lb_pop();
        const uint item = tid / LB_LANES;
        const uint lane = tid % LB_LANES;

        bool insert = false;
        uint64_t next_address = 0ul;
        uint64_t next_id = 0ul;
        uint next_flags = 0u;
        if (item < items)
        {
            const uint first = item * LB_WORDS;
            const uint64_t address = packUint2x32(uvec2(lb_cache[first], lb_cache[first + 1u]));
            const uint64_t block_id = packUint2x32(uvec2(lb_cache[first + 2u], lb_cache[first + 3u]));
            const uint level = (lb_cache[first + 4u] >> 8u) & 255u;

            const OctNode node = oct_load_inner(OctInnerBlock(address), lane);
            const uint64_t id = block_id * 8ul + uint64_t(lane);
            const bool part = (node.status & OCT_PART) != 0u;
            const bool last_level = (node.status & OCT_LAST_LEVEL) != 0u;
            const bool at_min_level = level == pc.min_level;

            if (selection[node.status] != 0u && (!part || at_min_level))
                emit(id, level, node.status, node.occupancy);

            if (part && !at_min_level && node.child != 0ul)
            {
                if (!last_level)
                {
                    insert = true;
                    next_address = node.child;
                    next_id = id;
                    next_flags = (level - 1u) << 8u;
                }
                else
                {
                    OctLeafBlock leaves = OctLeafBlock(node.child);
                    for (uint c = 0u; c < 8u; ++c)
                    {
                        const OctNode leaf = oct_load_leaf(leaves, c);
                        if (selection[leaf.status] != 0u)
                            emit(id * 8ul + uint64_t(c), 0u, leaf.status, leaf.occupancy);
                    }
                }
            }
        }

        uint total;
        const uint exclusive = lb_prefix(insert, total);
        if (insert)
        {
            const uint at = lb_base() + (lb_num + exclusive) * LB_WORDS;
            stacks[at] = uint(next_address & 0xFFFFFFFFul);
            stacks[at + 1u] = uint(next_address >> 32);
            stacks[at + 2u] = uint(next_id & 0xFFFFFFFFul);
            stacks[at + 3u] = uint(next_id >> 32);
            stacks[at + 4u] = next_flags;
            stacks[at + 5u] = 0u;
        }
        lb_commit(total);
    }
    if (tid == 0u)
        stack_counts[gl_WorkGroupID.x] = lb_num;
}
