// Helpers of the dynamic octree kernels (insert / propagate, slice S6d). Include after octree_nodes.glsl (OCT_PROB defined by the wrapper).
// A node is addressed by its device address: a block reference at that address with slot 0 reads or writes exactly that node.
const uint OCT_NEEDS_UPDATE = 128u;
const uint OCT_UPDATE_SUBTREE = 64u;     // flag: the subtree of this node is updated with its occupancy (probabilistic soft insert)
#if OCT_PROB
const uint OCT_NODE_WORDS = 3u;          // words of one inner node
const uint64_t OCT_NODE_BYTES = 12ul;
#else
const uint OCT_NODE_WORDS = 2u;
const uint64_t OCT_NODE_BYTES = 8ul;
#endif

OctNode oct_load_node(uint64_t address) { return oct_load_inner(OctInnerBlock(address), 0u); }
void oct_store_node(uint64_t address, OctNode n) { oct_store_inner(OctInnerBlock(address), 0u, n); }

// A fresh child (no occupancy bit) or an invalid leaf takes over from its parent: deterministic every bit except PART and LAST_LEVEL,
// probabilistic the occupancy (the status then only says UNKNOWN, every other bit stays).
void oct_expand(inout OctNode child, in OctNode parent)
{
#if OCT_PROB
    child.occupancy = parent.occupancy;
    child.status = (child.status & ~OCT_OCCUPANCY_MASK) | OCT_UNKNOWN;
#else
    child.status = (child.status & (OCT_PART | OCT_LAST_LEVEL)) | (parent.status & ~(OCT_PART | OCT_LAST_LEVEL) & 255u);
#endif
}

// Status and occupancy of `parent` from its 8 children. level 1 adds LAST_LEVEL in the deterministic rule, 99 means "above level 1".
void oct_recompute(inout OctNode parent, in OctNode kids[8], uint level)
{
#if OCT_PROB
    int highest = kids[0].occupancy;
    for (uint k = 1u; k < 8u; ++k)
        highest = max(highest, kids[k].occupancy);
    bool same = true;
    for (uint k = 0u; k < 8u; ++k)
        same = same && kids[k].occupancy == highest && (kids[k].status & OCT_PART) == 0u;
    parent.occupancy = highest;
    parent.status = (parent.status & ~(OCT_PART | OCT_OCCUPANCY_MASK)) | (same ? 0u : OCT_PART) | OCT_UNKNOWN;
#else
    const OctNode result = oct_parent(kids, level, 0ul);
    parent.status = result.status;
#endif
}

// CUDA topDownSubtreeUpdate of an inner node below a node with UPDATE_SUBTREE (probabilistic only; the deterministic one does nothing): a child
// with PART takes the occupancy of the parent, a child without PART is UPDATED with it (clamped, no UNKNOWN special case).
void oct_subtree_update(inout OctNode child, in OctNode parent)
{
#if OCT_PROB
    child.occupancy = (child.status & OCT_PART) != 0u ? parent.occupancy : clamp(child.occupancy + parent.occupancy, -127, 127);
#endif
}
