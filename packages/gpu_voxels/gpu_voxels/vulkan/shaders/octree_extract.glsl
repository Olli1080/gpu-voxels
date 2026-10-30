// One level of the extraction (see octree_extract_det.comp / octree_extract_prob.comp, which define OCT_PROB).
// Work items are blocks of 8 sibling nodes of `level` (>= 1); one invocation per (item, node). Rules of the CUDA kernel:
//  * a selected node is emitted when it is uniform (no PART) or when it sits on min_level;
//  * a PART node below min_level is refined (next work item = its child block), a level 1 node (LAST_LEVEL) emits its
//    selected leaf children instead;
//  * `selection` is the 256 entry table indexed by the full status byte (non zero = extract).
// counters[0] = number of items written to next_items, counters[1] = number of emitted nodes (counted always, written only when
// the index is below capacity: the counting pass has capacity 0).
#include "octree_nodes.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Items { uint items[]; };       // 4 words: address low/high, id low/high
layout(std430, set = 0, binding = 1) writeonly buffer NextItems { uint next_items[]; };
layout(std430, set = 0, binding = 2) writeonly buffer Out { uint out_nodes[]; };    // 4 words: id low/high, level | status << 8 | occupancy << 16, 0
layout(std430, set = 0, binding = 3) buffer Counters { uint counters[]; };
layout(std430, set = 0, binding = 4) readonly buffer Selection { uint selection[]; };
layout(push_constant) uniform Push
{
    uint item_count;
    uint level;
    uint min_level;
    uint out_capacity;
    uint next_capacity;
} pc;

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
    const uint t = oct_gid();
    if (t >= pc.item_count * 8u)
        return;
    const uint item = t >> 3u;
    const uint lane = t & 7u;
    const uint64_t address = packUint2x32(uvec2(items[4u * item], items[4u * item + 1u]));
    const uint64_t block_id = packUint2x32(uvec2(items[4u * item + 2u], items[4u * item + 3u]));

    const OctNode node = oct_load_inner(OctInnerBlock(address), lane);
    const uint64_t id = block_id * 8ul + uint64_t(lane);
    const bool part = (node.status & OCT_PART) != 0u;
    const bool last_level = (node.status & OCT_LAST_LEVEL) != 0u;
    const bool at_min_level = pc.level == pc.min_level;

    if (selection[node.status] != 0u && (!part || at_min_level))
        emit(id, pc.level, node.status, node.occupancy);

    if (part && !at_min_level && node.child != 0ul)
    {
        if (!last_level)
        {
            const uint index = atomicAdd(counters[0], 1u);
            if (index < pc.next_capacity)
            {
                next_items[4u * index] = uint(node.child & 0xFFFFFFFFul);
                next_items[4u * index + 1u] = uint(node.child >> 32);
                next_items[4u * index + 2u] = uint(id & 0xFFFFFFFFul);
                next_items[4u * index + 3u] = uint(id >> 32);
            }
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
