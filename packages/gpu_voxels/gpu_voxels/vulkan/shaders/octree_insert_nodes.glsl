// Insert, step 4 for a target level above 0: the datum of every voxel on the node of the target level (CUDA setNode / updateNode of an inner
// node). The node is either NEW (kind 1: in the block of that level created by the link kernel) or EXISTING (kind 2: the node the walk ended
// on). Deterministic: status = (status without occupancy bits and PART) | datum, so an existing node loses PART (hard and soft mode alike).
// Probabilistic: the occupancy is SET to 127 / -127 (not added), the status says UNKNOWN; the hard mode clears PART (the subtree is cut, its
// block stays reachable), the soft mode keeps PART and sets UPDATE_SUBTREE on an existing node that has it.
// The wrapper defines OCT_PROB and includes this file.
#include "octree_nodes.glsl"
#include "octree_dyn.glsl"

layout(local_size_x = 64) in;
#include "octree_gid.glsl"
layout(std430, set = 0, binding = 0) readonly buffer Ids { uint ids[]; };
layout(std430, set = 0, binding = 1) readonly buffer Info { uvec4 info[]; };
layout(std430, set = 0, binding = 2) readonly buffer HeadsT { uint heads_t[]; };
layout(std430, set = 0, binding = 3) readonly buffer OffsetsT { uint excl_t[]; };
layout(push_constant) uniform Push
{
    uint n;
    uint datum;     // OCT_OCCUPIED or OCT_FREE
    uint soft;      // 1 = soft mode
    uint level;     // target level, 1 .. 13
    uint64_t base;  // first block of the new chunk of the target level (0 if there is none)
} pc;

void main()
{
    const uint i = oct_gid();
    if (i >= pc.n)
        return;
    const uint64_t id = packUint2x32(uvec2(ids[2u * i], ids[2u * i + 1u]));
    const uvec4 f = info[i];
    uint64_t address;
    if (f.w == 1u)
        address = pc.base + uint64_t(excl_t[i] + heads_t[i] - 1u) * uint64_t(OCT_INNER_BLOCK_BYTES) + uint64_t(uint(id >> (3u * pc.level)) & 7u) * OCT_NODE_BYTES;
    else if (f.w == 2u)
        address = packUint2x32(uvec2(f.y, f.z));
    else
        return;
    OctNode node = oct_load_node(address);
#if OCT_PROB
    if (f.w == 2u && pc.soft != 0u && (node.status & OCT_PART) != 0u)
        node.flags |= OCT_UPDATE_SUBTREE;
    if (pc.soft == 0u)
        node.status &= ~OCT_PART;
    node.occupancy = pc.datum == OCT_OCCUPIED ? 127 : -127;
    node.status = (node.status & ~OCT_OCCUPANCY_MASK) | OCT_UNKNOWN;
#else
    node.status = (node.status & ~(OCT_OCCUPANCY_MASK | OCT_PART)) | pc.datum;
#endif
    oct_store_node(address, node);
}
