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
// Probabilistic tree (S6d-3a, same insert / propagate structure, other node rules from kernel_common.h _updateStatus, topDownUpdate,
// _bottomUpUpdateProb, updateNode / setNode of LeafNodeProb): occupancy values UNKNOWN -128, MIN -127, MAX 127, INITIAL 0. The occupancy bits
// of a status are always UNKNOWN once a node is valid. New nodes start with occupancy 0 (leaves: status INVALID | DYNAMIC). A NEW leaf is
// SET to the datum (127 occupied, -127 free), an EXISTING leaf is UPDATED: occupancy = clamp(old (UNKNOWN counts as 0) + datum, -127, 127).
// Propagate: a child without occupancy bits (an invalid leaf: all three) copies the occupancy of its parent and gets the status
// UNKNOWN (every other bit stays); a parent takes the MAXIMUM occupancy of its children, gets PART unless all children have exactly that
// occupancy and none has PART, status = (status without PART and occupancy bits) | PART? | UNKNOWN (LAST_LEVEL, STATIC, DYNAMIC stay as
// they are: they are not recomputed from the children).
//
// Target levels and the soft mode (S6d-3b, kernel_insert_countNeededNodes / kernel_insert_setNodes with target_level > 0, updateNode of
// InnerNode / InnerNodeProb, topDownSubtreeUpdate): an insert at level t > 0 walks while the node has PART and the level is above t. If it
// ends on a node of level t, that node is SET: deterministic status = (status without occupancy bits and PART) | datum, probabilistic
// occupancy = 127 / -127 (set, not added) and status UNKNOWN; the PART bit is cleared in the HARD mode, which cuts the subtree (the child
// block stays reachable but stale), and kept in the SOFT mode (probabilistic only: the deterministic soft mode cuts as well), which also sets
// the flag UPDATE_SUBTREE on a node that has PART. If the walk ends above level t on a node without PART, new blocks are created down to
// level t and the node of level t is SET as a new node. Propagate: a node with PART passes `update_subtree` (its own flag or the one of
// the block it sits in) to its children; probabilistic children that are not fresh are then changed: a child with PART takes the parent's
// occupancy, a child without PART is UPDATED with it (clamp(child + parent, -127, 127), no UNKNOWN special case); level 1 nodes update their
// leaves the same way after the invalid leaves have taken over (an invalid leaf therefore gets the parent's occupancy twice). Children
// below an update_subtree node are visited whether flagged or not. clearNeedsUpdate clears NEEDS_UPDATE and UPDATE_SUBTREE together.
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
			int8_t occupancy = 0; ///< probabilistic tree only
			bool needs_update = false;
			bool update_subtree = false;
			bool has_block = false;
		};

		enum class Datum { Occupied, Free };
		enum class Mode { Hard, Soft };

		/// An empty tree: the root alone with status UNKNOWN (probabilistic: occupancy -128), what Octree::clear() leaves.
		explicit DynamicOctreeReference(Kind kind = Kind::Deterministic) : kind_(kind)
		{
			Memory& root = nodes_[{ kRootLevel, 0 }];
			root.status = status::UNKNOWN;
			root.occupancy = kind == Kind::Probabilistic ? kUnknownOccupancy : int8_t(0);
		}

		/// Starts from the nodes of a canonical tree (OctreeReference::dump()), all flags clear.
		explicit DynamicOctreeReference(const std::vector<Node>& dump, Kind kind = Kind::Deterministic) : kind_(kind)
		{
			for (const Node& n : dump)
			{
				Memory& m = nodes_[{ n.level, n.id }];
				m.status = n.status;
				m.occupancy = n.occupancy;
				m.has_block = n.has_children;
			}
		}

		/// CUDA insertVoxel for one voxel (a level 0 id; for level > 0 only the bits above 3 * level count): the node of `level` on its path is
		/// set occupied / free. Hard = hard reset data, Soft = soft reset data (differs from hard in the probabilistic tree only).
		/// `occupancy` (slice S6f-4, probabilistic tree): the value the insert uses instead of 127 / -127 (the sensor pipeline inserts free space and objects with
		/// the occupancy of their rays / points); kDatumValue = the constant of the datum.
		static constexpr int kDatumValue = 1000;

		void insert(uint64_t voxel_id, Datum datum, uint32_t level_wanted = 0, Mode mode = Mode::Hard, int occupancy = kDatumValue)
		{
			const uint32_t to_level = std::max(level_wanted, 1u);
			uint32_t level = kRootLevel;
			uint64_t id = 0;
			// the walk
			while (level > to_level && part(level, id))
			{
				nodes_.at({ level, id }).needs_update = true;
				--level;
				id = voxel_id >> (3 * level);
			}
			nodes_.at({ level, id }).needs_update = true;
			if (level_wanted == 0 && level == 1 && part(1, id))
			{
				set_target(nodes_.at({ 0, voxel_id }), 0, datum, mode, true, occupancy);
				return;
			}
			if (level_wanted > 0 && level == level_wanted)
			{
				set_target(nodes_.at({ level, id }), level, datum, mode, true, occupancy);
				return;
			}
			// new blocks below the node where the walk ended, down to the wanted level
			drop_children(level, id);
			for (uint32_t l = level; l > level_wanted; --l)
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
			set_target(nodes_.at({ level_wanted, voxel_id >> (3 * level_wanted) }), level_wanted, datum, mode, false, occupancy);
		}

		/// Propagate on the whole tree.
		void propagate()
		{
			Memory& root = nodes_.at({ kRootLevel, 0 });
			if (!(root.status & status::PART) || !root.needs_update)
			{
				root.needs_update = false;
				root.update_subtree = false;
				return;
			}
			propagate_node(kRootLevel, 0, false);
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
		/// True if a node reachable through PART nodes from the root still carries NEEDS_UPDATE (stale subtrees below a cut node keep
		/// their flags, as in CUDA: nothing visits them).
		bool has_pending_updates() const { return pending(kRootLevel, 0); }

	private:
		bool part(uint32_t level, uint64_t id) const { return (nodes_.at({ level, id }).status & status::PART) != 0; }

		bool pending(uint32_t level, uint64_t id) const
		{
			const Memory& m = nodes_.at({ level, id });
			if (m.needs_update)
				return true;
			if (level == 0 || !(m.status & status::PART) || !m.has_block)
				return false;
			for (uint32_t s = 0; s < 8; ++s)
				if (pending(level - 1, id * 8 + s))
					return true;
			return false;
		}

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

		/// The node (level >= 2, PART, flagged) processes its child block. `inherit`: an ancestor passes UPDATE_SUBTREE down.
		void propagate_node(uint32_t level, uint64_t id, bool inherit)
		{
			Memory& parent = nodes_.at({ level, id });
			const bool subtree = inherit || parent.update_subtree;
			Memory* kids[8];
			for (uint32_t s = 0; s < 8; ++s)
			{
				Memory& child = nodes_.at({ level - 1, id * 8 + s });
				kids[s] = &child;
				if ((child.status & status::OCCUPANCY_MASK) == 0)
					expand(child, parent);
				else if (subtree)
					subtree_update(child, parent);
				if (!(child.status & status::PART))
				{
					child.needs_update = false;
					child.update_subtree = false;
				}
			}
			for (uint32_t s = 0; s < 8; ++s)
			{
				// level 1 nodes are recomputed from their leaves whenever they have PART, flagged or not (CUDA does it; for a consistent
				// tree without a subtree update it changes nothing: mutation M9 survives for that reason)
				if (!(kids[s]->status & status::PART))
					continue;
				if (level == 2)
					recompute_from_leaves(id * 8 + s, subtree);
				else if (kids[s]->needs_update || subtree)
					propagate_node(level - 1, id * 8 + s, subtree);
			}
			recompute(parent, kids, 99);
			parent.needs_update = false;
			parent.update_subtree = false;
		}

		/// A level 1 node with PART: invalid leaves take over from the node, with UPDATE_SUBTREE the leaves are updated with its occupancy,
		/// then the node is recomputed from its leaves.
		void recompute_from_leaves(uint64_t id, bool inherit)
		{
			Memory& node = nodes_.at({ 1, id });
			if (!(node.status & status::PART))
			{
				node.needs_update = false;
				node.update_subtree = false;
				return;
			}
			const bool subtree = inherit || node.update_subtree;
			Memory* leaves[8];
			for (uint32_t s = 0; s < 8; ++s)
			{
				Memory& leaf = nodes_.at({ 0, id * 8 + s });
				leaves[s] = &leaf;
				if ((leaf.status & status::OCCUPANCY_MASK) == status::OCCUPANCY_MASK)
					expand(leaf, node);
				if (subtree && kind_ == Kind::Probabilistic)
					leaf.occupancy = clamp_occupancy(int(leaf.occupancy) + int(node.occupancy));
			}
			recompute(node, leaves, 1);
			node.needs_update = false;
			node.update_subtree = false;
		}

		static int clamp_occupancy(int value) { return std::min(std::max(value, -127), 127); }

		/// CUDA topDownSubtreeUpdate of an inner node (probabilistic; the deterministic one does nothing).
		void subtree_update(Memory& child, const Memory& parent) const
		{
			if (kind_ != Kind::Probabilistic)
				return;
			child.occupancy = (child.status & status::PART) ? parent.occupancy : int8_t(clamp_occupancy(int(child.occupancy) + int(parent.occupancy)));
		}

		/// The datum of an insert on the node of `level`: a leaf (level 0, deterministic: set; probabilistic: a new leaf is set, an existing one
		/// updated) or an inner node (set, PART cleared in the hard mode, UPDATE_SUBTREE in the soft probabilistic mode).
		void set_target(Memory& node, uint32_t level, Datum datum, Mode mode, bool existing, int occupancy = kDatumValue) const
		{
			const int value = occupancy == kDatumValue ? datum_value(datum) : occupancy;
			const uint8_t set = datum == Datum::Occupied ? status::OCCUPIED : status::FREE;
			if (kind_ == Kind::Deterministic)
			{
				node.status = uint8_t((node.status & ~(status::OCCUPANCY_MASK | status::PART)) | set);
				return;
			}
			if (level == 0)
			{
				const int old_value = node.occupancy == kUnknownOccupancy ? 0 : node.occupancy;
				node.occupancy = existing ? int8_t(clamp_occupancy(old_value + value)) : int8_t(value);
			}
			else
			{
				if (existing && mode == Mode::Soft && (node.status & status::PART))
					node.update_subtree = true;
				if (mode == Mode::Hard)
					node.status = uint8_t(node.status & ~status::PART);
				node.occupancy = int8_t(value);
			}
			node.status = uint8_t((node.status & ~status::OCCUPANCY_MASK) | status::UNKNOWN);
		}

		static int datum_value(Datum datum) { return datum == Datum::Occupied ? 127 : -127; }

		/// A fresh child takes over from its parent: deterministic every bit except PART / LAST_LEVEL, probabilistic the occupancy
		/// (and its status becomes UNKNOWN).
		void expand(Memory& child, const Memory& parent) const
		{
			if (kind_ == Kind::Probabilistic)
			{
				child.occupancy = parent.occupancy;
				child.status = uint8_t((child.status & ~status::OCCUPANCY_MASK) | status::UNKNOWN);
			}
			else
				child.status = uint8_t((child.status & (status::PART | status::LAST_LEVEL)) | (parent.status & ~(status::PART | status::LAST_LEVEL)));
		}

		/// The parent from its 8 children (`level` 1 adds LAST_LEVEL in the deterministic rule, 99 stands for "above level 1").
		void recompute(Memory& parent, Memory* const (&kids)[8], uint32_t level) const
		{
			if (kind_ == Kind::Probabilistic)
			{
				int8_t highest = kids[0]->occupancy;
				for (const Memory* k : kids)
					highest = std::max(highest, k->occupancy);
				bool same = true;
				for (const Memory* k : kids)
					same = same && k->occupancy == highest && !(k->status & status::PART);
				parent.occupancy = highest;
				parent.status = uint8_t((parent.status & ~(status::PART | status::OCCUPANCY_MASK)) | (same ? 0 : status::PART) | status::UNKNOWN);
				return;
			}
			uint8_t or_status = 0, and_status = 0xFF;
			for (const Memory* k : kids)
			{
				or_status |= k->status;
				and_status &= k->status;
			}
			parent.status = new_status(or_status, and_status, level, or_status);
		}

		void collect(uint32_t level, uint64_t id, std::vector<Node>& out) const
		{
			const Memory& m = nodes_.at({ level, id });
			out.push_back(Node{ level, id, m.status, m.occupancy, level > 0 && m.has_block });
			if (level == 0 || !m.has_block)
				return;
			for (uint32_t s = 0; s < 8; ++s)
				collect(level - 1, id * 8 + s, out);
		}

		Kind kind_ = Kind::Deterministic;
		std::map<std::pair<uint32_t, uint64_t>, Memory> nodes_;
		uint64_t garbage_[kRootLevel + 1] = {};
	};
}

#endif
