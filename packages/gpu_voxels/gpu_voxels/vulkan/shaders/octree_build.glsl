// One level of the bottom-up build (see octree_build_det.comp / octree_build_prob.comp, which define OCT_PROB).
// The elements are the sorted distinct ids of the level (`ids`, written by the previous level for level > 0); heads/offsets
// mark the groups that share a parent. The thread of the first element of a group builds the whole block of 8 sibling nodes
// of that parent: occupied children get their final status (computed from their own child block, which the previous level
// already wrote), every other slot is a filler. This yields the tree that CUDA's build + propagate leaves behind.
#include "octree_nodes.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Ids { uint ids[]; };
layout(std430, set = 0, binding = 1) readonly buffer Heads { uint heads[]; };
layout(std430, set = 0, binding = 2) readonly buffer Offsets { uint offsets[]; };
layout(std430, set = 0, binding = 3) writeonly buffer NextIds { uint next_ids[]; };
layout(push_constant) uniform Push
{
    uint n;
    uint level;
    uint64_t chunk;      // first block of this level
    uint64_t prev_chunk; // first block of the level below (level > 0)
} pc;

uint64_t id_at(uint i) { return packUint2x32(uvec2(ids[2u * i], ids[2u * i + 1u])); }

void main()
{
    const uint i = oct_gid();
    if (i >= pc.n || heads[i] == 0u)
        return;
    const uint block = offsets[i];
    const uint shift = 3u * (pc.level + 1u);
    const uint64_t parent = id_at(i) >> shift;

    OctNode nodes[8];
    for (uint s = 0u; s < 8u; ++s)
        nodes[s] = oct_filler(pc.level);

    for (uint j = i; j < pc.n && (id_at(j) >> shift) == parent; ++j)
    {
        const uint64_t id = id_at(j);
        const uint slot = uint(id >> (3u * pc.level)) & 7u;
        if (pc.level == 0u)
            nodes[slot] = oct_occupied_leaf();
        else
        {
            // block j of the level below holds the children of this node
            const uint64_t child_ptr = pc.prev_chunk + uint64_t(j) * uint64_t(oct_block_bytes(pc.level - 1u));
            OctNode kids[8];
            if (pc.level == 1u)
            {
                OctLeafBlock lb = OctLeafBlock(child_ptr);
                for (uint c = 0u; c < 8u; ++c)
                    kids[c] = oct_load_leaf(lb, c);
            }
            else
            {
                OctInnerBlock ib = OctInnerBlock(child_ptr);
                for (uint c = 0u; c < 8u; ++c)
                    kids[c] = oct_load_inner(ib, c);
            }
            nodes[slot] = oct_parent(kids, pc.level, child_ptr);
        }
    }

    const uint64_t address = pc.chunk + uint64_t(block) * uint64_t(oct_block_bytes(pc.level));
    if (pc.level == 0u)
        oct_store_leaf_block(OctLeafBlock(address), nodes);
    else
    {
        OctInnerBlock out_block = OctInnerBlock(address);
        for (uint s = 0u; s < 8u; ++s)
            oct_store_inner(out_block, s, nodes[s]);
    }
    next_ids[2u * block] = ids[2u * i];
    next_ids[2u * block + 1u] = ids[2u * i + 1u];
}
