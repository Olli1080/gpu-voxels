// Independent CPU reference of the DYNAMIC octree operations of the deterministic tree (slice S6d-1): hard insert of single voxels
// (target level 0) and propagate, on the memory tree including the PART / NEEDS_UPDATE flags and stale child blocks.
//
// Written from the CUDA code (octree/NTree.hpp insertVoxel, kernels/kernel_Octree.h kernel_insert_*, kernels/kernel_common.h
// insertNode / setNode / updateNode / topDownUpdate / bottomUpUpdate, load_balancer/kernel_config/Propagate.cuh, octree/Nodes.h
// getNewStatus). Nothing could be run under CUDA: every rule below is INFERRED from reading (Agent/Analysis/2026-10-31_s6d_design.md).
// It shares no code with the Vulkan implementation: nodes live in a std::map keyed by (level, id), every operation is a plain
// recursion on single voxels (CUDA inserts a sorted batch in parallel and propagates with a load-balanced queue).
//
// Model of the memory tree: a node has a status byte, a NEEDS_UPDATE flag and a flag "has a child block"; the block of a node at
// level L consists of the 8 nodes (L-1, 8*id + 0..7) and exists iff the node has a child pointer. A node WITHOUT PART may still own
// a child block (a uniform node that keeps its children, or a node cut by a hard insert): the block is reachable and shown by
// dump(), but traversals ignore it. A block that loses its parent pointer is garbage (counted per level, never listed).
//
// Rules (deterministic tree, hard insert of an occupied / free voxel, target level 0):
//  * Insert, per voxel: walk from the root while the node has PART and level > 1. If the walk ends on a level 1 node with PART the
//    leaf exists and is updated: status = (status & ~(FREE|UNKNOWN|OCCUPIED|PART)) | datum. Otherwise the node (no PART) gets a NEW
//    child block on every level down to the leaves: inner nodes DYNAMIC (LAST_LEVEL on level 1) with NEEDS_UPDATE, leaves
//    INVALID (FREE|UNKNOWN|OCCUPIED) | DYNAMIC; the nodes on the voxel's path get PART, the leaf becomes DYNAMIC | datum (the
//    invalid bits are replaced); the stale block the node pointed to becomes garbage. Every node visited on the way down and the
//    node where the walk ended get NEEDS_UPDATE.
//  * Propagate (from the root): a node with PART and NEEDS_UPDATE processes its child block: a child without occupancy bits takes
//    all bits except PART / LAST_LEVEL from its parent ("expanding a compressed node"), a child without PART loses NEEDS_UPDATE,
//    a child with PART and NEEDS_UPDATE is propagated first; then the parent gets getNewStatus(OR, AND of the children) (no
//    LAST_LEVEL above level 1; the node of level 1 adds it) and loses NEEDS_UPDATE. A level 1 block is different: every child with
//    PART takes the leaves' invalid bits from itself first, then its status is recomputed from its leaves whether it is flagged or
//    not, and its flag is cleared.
#ifndef GVL_TEST_OCTREE_DYNAMIC_REFERENCE_H_INCLUDED
#define GVL_TEST_OCTREE_DYNAMIC_REFERENCE_H_INCLUDED

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "octree_reference.h"

namespace gvl_reference::octree
{
	class DynamicOctreeReference
	{
	public:
		struct Memory
		{
			uint8_t status = 0;
			bool needs_update = false;
			bool has_block = false;
		};

		enum class Datum { Occupied, Free };

		/// An empty tree: the root alone with status UNKNOWN (what Octree::clear() leaves).
		DynamicOctreeReference()
		{
			nodes_[{ kRootLevel, 0 }].status = status::UNKNOWN;
		}

		/// Starts from the nodes of a canonical tree (OctreeReference::dump()), all flags clear.
		explicit DynamicOctreeReference(const std::vector<Node>& dump)
		{
			for (const Node& n : dump)
			{
				Memory& m = nodes_[{ n.level, n.id }];
				m.status = n.status;
				m.has_block = n.has_children;
			}
		}

		/// CUDA insertVoxel (target level 0, hard reset data) for one voxel.
		void insert(uint64_t voxel_id, Datum datum)
		{
			const uint8_t set = datum == Datum::Occupied ? status::OCCUPIED : status::FREE;
			uint32_t level = kRootLevel;
			uint64_t id = 0;
			// the walk
			while (level > 1 && part(level, id))
			{
				nodes_.at({ level, id }).needs_update = true;
				--level;
				id = voxel_id >> (3 * level);
			}
			nodes_.at({ level, id }).needs_update = true;
			if (level == 1 && part(1, id))
			{
				Memory& leaf = nodes_.at({ 0, voxel_id });
				leaf.status = uint8_t((leaf.status & ~(status::OCCUPANCY_MASK | status::PART)) | set);
				return;
			}
			// new blocks below the node where the walk ended
			drop_children(level, id);
			for (uint32_t l = level; l >= 1; --l)
			{
				const uint64_t block = voxel_id >> (3 * l);
				for (uint32_t s = 0; s < 8; ++s)
				{
					Memory& child = nodes_[{ l - 1, block * 8 + s }];
					child = Memory{};
					if (l == 1)
						child.status = status::OCCUPANCY_MASK | status::DYNAMIC;
					else
					{
						child.status = uint8_t(status::DYNAMIC | (l == 2 ? status::LAST_LEVEL : 0));
						child.needs_update = true;
					}
				}
				Memory& owner = nodes_.at({ l, voxel_id >> (3 * l) });
				owner.has_block = true;
				owner.status |= status::PART;
			}
			Memory& leaf = nodes_.at({ 0, voxel_id });
			leaf.status = uint8_t((leaf.status & ~(status::OCCUPANCY_MASK | status::PART)) | set);
		}

		/// Propagate on the whole tree.
		void propagate()
		{
			Memory& root = nodes_.at({ kRootLevel, 0 });
			if (!(root.status & status::PART) || !root.needs_update)
			{
				root.needs_update = false;
				return;
			}
			propagate_node(kRootLevel, 0);
		}

		/// Reachable nodes ordered by (level, id), in the format of OctreeReference::dump().
		std::vector<Node> dump() const
		{
			std::vector<Node> out;
			collect(kRootLevel, 0, out);
			std::sort(out.begin(), out.end(), [](const Node& a, const Node& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
			return out;
		}

		/// Blocks per level that lost their parent pointer.
		uint64_t garbage_blocks(uint32_t level) const { return garbage_[level]; }
		/// True if some reachable node still carries NEEDS_UPDATE.
		bool has_pending_updates() const
		{
			for (const Node& n : dump())
				if (nodes_.at({ n.level, n.id }).needs_update)
					return true;
			return false;
		}

	private:
		bool part(uint32_t level, uint64_t id) const { return (nodes_.at({ level, id }).status & status::PART) != 0; }

		void drop_children(uint32_t level, uint64_t id)
		{
			Memory& owner = nodes_.at({ level, id });
			if (!owner.has_block)
				return;
			owner.has_block = false;
			++garbage_[level - 1];
			for (uint32_t s = 0; s < 8; ++s)
			{
				const uint64_t child = id * 8 + s;
				if (level > 1)
					drop_children(level - 1, child);
				nodes_.erase({ level - 1, child });
			}
		}

		void propagate_node(uint32_t level, uint64_t id)
		{
			Memory& parent = nodes_.at({ level, id });
			if (level == 1)
			{
				recompute_from_leaves(id);
				return;
			}
			Memory* kids[8];
			for (uint32_t s = 0; s < 8; ++s)
			{
				Memory& child = nodes_.at({ level - 1, id * 8 + s });
				kids[s] = &child;
				if ((child.status & status::OCCUPANCY_MASK) == 0)
					child.status = uint8_t((child.status & (status::PART | status::LAST_LEVEL)) | (parent.status & ~(status::PART | status::LAST_LEVEL)));
				if (!(child.status & status::PART))
					child.needs_update = false;
			}
			for (uint32_t s = 0; s < 8; ++s)
				if ((kids[s]->status & status::PART) && kids[s]->needs_update)
				{
					propagate_node(level - 1, id * 8 + s);
				}
			if (level - 1 == 1)
			{
				// the block of level 1 nodes: every node with PART is recomputed from its leaves, flagged or not (CUDA does it; for a
				// consistent tree it changes nothing, mutation M9 survives for that reason, later steps rely on it: soft insert, subtree updates)
				for (uint32_t s = 0; s < 8; ++s)
					if ((kids[s]->status & status::PART) && !kids[s]->needs_update)
						recompute_from_leaves(id * 8 + s);
			}
			uint8_t or_status = 0, and_status = 0xFF;
			for (uint32_t s = 0; s < 8; ++s)
			{
				or_status |= kids[s]->status;
				and_status &= kids[s]->status;
			}
			parent.status = new_status(or_status, and_status, 99, or_status);
			parent.needs_update = false;
		}

		/// A level 1 node with PART: invalid leaves take the node's bits, the node is recomputed from its leaves.
		void recompute_from_leaves(uint64_t id)
		{
			Memory& node = nodes_.at({ 1, id });
			if (!(node.status & status::PART))
			{
				node.needs_update = false;
				return;
			}
			uint8_t or_status = 0, and_status = 0xFF;
			for (uint32_t s = 0; s < 8; ++s)
			{
				Memory& leaf = nodes_.at({ 0, id * 8 + s });
				if ((leaf.status & status::OCCUPANCY_MASK) == status::OCCUPANCY_MASK)
					leaf.status = uint8_t((leaf.status & (status::PART | status::LAST_LEVEL)) | (node.status & ~(status::PART | status::LAST_LEVEL)));
				or_status |= leaf.status;
				and_status &= leaf.status;
			}
			node.status = new_status(or_status, and_status, 1, or_status);
			node.needs_update = false;
		}

		void collect(uint32_t level, uint64_t id, std::vector<Node>& out) const
		{
			const Memory& m = nodes_.at({ level, id });
			out.push_back(Node{ level, id, m.status, 0, level > 0 && m.has_block });
			if (level == 0 || !m.has_block)
				return;
			for (uint32_t s = 0; s < 8; ++s)
				collect(level - 1, id * 8 + s, out);
		}

		std::map<std::pair<uint32_t, uint64_t>, Memory> nodes_;
		uint64_t garbage_[kRootLevel + 1] = {};
	};
}

#endif
