// Node formats of the octree (native Vulkan layout, mirrored by include/gpu_voxels/vulkan/OctreeFormat.h).
// Requires GL_ARB_gpu_shader_int64 and GL_EXT_buffer_reference; define OCT_PROB as 0 (deterministic) or 1 (probabilistic)
// before including this file.
//
// Child addresses are stored with their low 48 bits and sign extended from bit 47 when loaded (canonical x86-64 style
// addresses: Intel hands out addresses with the upper 16 bits set; the host checks that every chunk is canonical).
//
// A tree level is stored as ONE chunk (buffer) of blocks of 8 sibling nodes; an inner node points to the block of its
// children with a 48-bit device address (0 = no children), exactly like the CUDA tree.
//   deterministic inner node, 8 bytes:   word0 = status | flags << 8 | (child & 0xFFFF) << 16, word1 = child >> 16
//   probabilistic inner node, 12 bytes:  word0 = status | flags << 8 | (occupancy & 0xFF) << 16, word1/word2 = child (low, high 16 bit)
//   deterministic leaf, 1 byte:          status                       (block of 8 leaves = 8 bytes)
//   probabilistic leaf, 2 bytes:         status | (occupancy & 0xFF) << 8   (block = 16 bytes)

const uint OCT_FREE = 1u;
const uint OCT_UNKNOWN = 2u;
const uint OCT_OCCUPIED = 4u;
const uint OCT_PART = 8u;
const uint OCT_LAST_LEVEL = 16u;
const uint OCT_COLLISION = 32u;
const uint OCT_STATIC = 64u;
const uint OCT_DYNAMIC = 128u;
const uint OCT_OCCUPANCY_MASK = 7u;
const uint OCT_ROOT_LEVEL = 14u;
const int OCT_PROB_UNKNOWN = -128;
const int OCT_PROB_MAX = 127;

struct OctNode
{
    uint status;
    uint flags;
    int occupancy;
    uint64_t child;
};

#if OCT_PROB
const uint OCT_INNER_BLOCK_BYTES = 96u;
const uint OCT_LEAF_BLOCK_BYTES = 16u;
layout(buffer_reference, std430, buffer_reference_align = 4) buffer OctInnerBlock { uint w[24]; };
layout(buffer_reference, std430, buffer_reference_align = 4) buffer OctLeafBlock { uint w[4]; };
#else
const uint OCT_INNER_BLOCK_BYTES = 64u;
const uint OCT_LEAF_BLOCK_BYTES = 8u;
layout(buffer_reference, std430, buffer_reference_align = 8) buffer OctInnerBlock { uint w[16]; };
layout(buffer_reference, std430, buffer_reference_align = 8) buffer OctLeafBlock { uint w[2]; };
#endif

uint64_t oct_sign_extend_48(uint64_t raw) { return uint64_t((int64_t(raw) << 16) >> 16); }

// Size in bytes of a block of nodes of the given tree level (level 0 = leaves).
uint oct_block_bytes(uint level) { return level == 0u ? OCT_LEAF_BLOCK_BYTES : OCT_INNER_BLOCK_BYTES; }

OctNode oct_load_inner(OctInnerBlock b, uint slot)
{
    OctNode n;
#if OCT_PROB
    const uint w0 = b.w[3u * slot];
    n.status = w0 & 255u;
    n.flags = (w0 >> 8u) & 255u;
    n.occupancy = bitfieldExtract(int(w0), 16, 8);
    n.child = oct_sign_extend_48(uint64_t(b.w[3u * slot + 1u]) | (uint64_t(b.w[3u * slot + 2u]) << 32));
#else
    const uint w0 = b.w[2u * slot];
    n.status = w0 & 255u;
    n.flags = (w0 >> 8u) & 255u;
    n.occupancy = 0;
    n.child = oct_sign_extend_48(uint64_t(w0 >> 16u) | (uint64_t(b.w[2u * slot + 1u]) << 16));
#endif
    return n;
}

void oct_store_inner(OctInnerBlock b, uint slot, OctNode n)
{
#if OCT_PROB
    b.w[3u * slot] = n.status | (n.flags << 8u) | ((uint(n.occupancy) & 255u) << 16u);
    b.w[3u * slot + 1u] = uint(n.child);
    b.w[3u * slot + 2u] = uint((n.child >> 32) & 0xFFFFul);
#else
    b.w[2u * slot] = n.status | (n.flags << 8u) | (uint(n.child) << 16u);
    b.w[2u * slot + 1u] = uint(n.child >> 16);
#endif
}

OctNode oct_load_leaf(OctLeafBlock b, uint slot)
{
    OctNode n;
    n.flags = 0u;
    n.child = 0ul;
#if OCT_PROB
    const uint v = (b.w[slot >> 1u] >> ((slot & 1u) * 16u)) & 0xFFFFu;
    n.status = v & 255u;
    n.occupancy = bitfieldExtract(int(v), 8, 8);
#else
    n.status = (b.w[slot >> 2u] >> ((slot & 3u) * 8u)) & 255u;
    n.occupancy = 0;
#endif
    return n;
}

// Writes the whole block of 8 leaves (a thread owns a whole block, so no atomics are needed).
void oct_store_leaf_block(OctLeafBlock b, in OctNode n[8])
{
#if OCT_PROB
    for (uint w = 0u; w < 4u; ++w)
    {
        const uint lo = n[2u * w].status | ((uint(n[2u * w].occupancy) & 255u) << 8u);
        const uint hi = n[2u * w + 1u].status | ((uint(n[2u * w + 1u].occupancy) & 255u) << 8u);
        b.w[w] = lo | (hi << 16u);
    }
#else
    for (uint w = 0u; w < 2u; ++w)
        b.w[w] = n[4u * w].status | (n[4u * w + 1u].status << 8u) | (n[4u * w + 2u].status << 16u) | (n[4u * w + 3u].status << 24u);
#endif
}

// Node without children and without occupied voxels, as left by build + propagate: UNKNOWN | STATIC (| LAST_LEVEL at level 1).
OctNode oct_filler(uint level)
{
    OctNode n;
    n.status = OCT_UNKNOWN | OCT_STATIC | (level == 1u ? OCT_LAST_LEVEL : 0u);
    n.flags = 0u;
#if OCT_PROB
    n.occupancy = OCT_PROB_UNKNOWN;
#else
    n.occupancy = 0;
#endif
    n.child = 0ul;
    return n;
}

OctNode oct_occupied_leaf()
{
    OctNode n;
    n.flags = 0u;
    n.child = 0ul;
#if OCT_PROB
    n.status = OCT_STATIC;
    n.occupancy = OCT_PROB_MAX;
#else
    n.status = OCT_OCCUPIED | OCT_STATIC;
    n.occupancy = 0;
#endif
    return n;
}

// Status of an inner node (of `level`) with children from the 8 children (the CUDA getNewStatus rule / max occupancy).
OctNode oct_parent(in OctNode kids[8], uint level, uint64_t child_ptr)
{
    OctNode p;
    p.flags = 0u;
    p.child = child_ptr;
#if OCT_PROB
    int pmax = kids[0].occupancy;
    int pmin = kids[0].occupancy;
    bool all_non_part = true;
    for (uint i = 0u; i < 8u; ++i)
    {
        pmax = max(pmax, kids[i].occupancy);
        pmin = min(pmin, kids[i].occupancy);
        all_non_part = all_non_part && (kids[i].status & OCT_PART) == 0u;
    }
    p.occupancy = pmax;
    p.status = OCT_UNKNOWN | OCT_STATIC;
    if (!(pmax == pmin && all_non_part))
        p.status |= OCT_PART;
    if (level == 1u)
        p.status |= OCT_LAST_LEVEL;
#else
    uint child_or = 0u;
    uint child_and = 255u;
    uint flags_or = 0u;
    for (uint i = 0u; i < 8u; ++i)
    {
        child_or |= kids[i].status & OCT_OCCUPANCY_MASK;
        child_and &= kids[i].status & OCT_OCCUPANCY_MASK;
        flags_or |= kids[i].status;
    }
    const bool single_bit = child_and == OCT_FREE || child_and == OCT_UNKNOWN || child_and == OCT_OCCUPIED;
    p.occupancy = 0;
    p.status = (single_bit && child_or == child_and) ? child_and : (child_or | OCT_PART);
    p.status |= flags_or & (OCT_STATIC | OCT_DYNAMIC);
    if (level == 1u)
        p.status |= OCT_LAST_LEVEL;
#endif
    return p;
}
