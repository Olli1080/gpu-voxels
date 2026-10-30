// Independent CPU reference of the octree node semantics (slice S6a).
//
// Written from the CUDA code (octree/Nodes.h getNewStatus, octree/NTree.hpp build, octree/kernels/kernel_common.h
// _bottomUpUpdateProb / _updateStatus / topDownUpdate, octree/Morton.h) and deliberately shares NO code with the Vulkan
// implementation: Morton codes are computed bit by bit in loops, the tree is built RECURSIVELY from a sorted voxel set
// (CUDA builds bottom-up with sort + per-level scans), nodes live in a std::map keyed by (level, id).
//
// What it describes: the observable tree after build() + propagate(), i.e. the canonical state that the Vulkan build
// must reproduce node by node. ALL of the following was derived by reading the CUDA code; nothing could be run (no CUDA
// toolkit on the development machine), so every rule below is *inferred* until golden data from a CUDA machine exist (Q5).
//
//  * Levels 0..14, voxels (coordinates < 2^14) are level 0, the root is level 14, a node of level L has the Morton id
//    `voxel_id >> 3L`; coordinates >= 2^14 are dropped (CUDA: assert only) and counted in dropped().
//  * Only nodes whose box contains an occupied voxel own a block of 8 children ("on-path" nodes); the other 7 siblings
//    of an on-path child are uniform nodes without children ("filler"). The memory tree keeps uniform nodes' children
//    in CUDA, but build() never creates any for fillers, so the dump contains exactly the on-path blocks.
//  * Deterministic tree: occupied leaf = OCCUPIED|STATIC, filler = UNKNOWN|STATIC (LAST_LEVEL added at level 1);
//    an on-path node takes getNewStatus(OR, AND of the children's occupancy bits, level): the common bit when all 8 children
//    carry the same single occupancy bit (no PART), else OR|PART; STATIC/DYNAMIC bits are OR-ed; level 1 adds LAST_LEVEL.
//  * Probabilistic tree: occupied leaf = occupancy 127, status STATIC; filler = occupancy -128, status STATIC|UNKNOWN
//    (LAST_LEVEL at level 1); an on-path node has occupancy = max of the children, status UNKNOWN|STATIC, PART unless all
//    children have the same occupancy and none is PART, LAST_LEVEL at level 1 (UNKNOWN is what _updateStatus always sets).
//  * An empty tree is the root alone: status UNKNOWN (probabilistic: occupancy -128).
#ifndef GVL_TEST_OCTREE_REFERENCE_H_INCLUDED
#define GVL_TEST_OCTREE_REFERENCE_H_INCLUDED

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <ostream>
#include <set>
#include <utility>
#include <vector>

namespace gvl_reference::octree
{
	constexpr uint32_t kRootLevel = 14;
	constexpr uint32_t kCoordinateLimit = 1u << 14;

	namespace status
	{
		constexpr uint8_t FREE = 1, UNKNOWN = 2, OCCUPIED = 4, PART = 8, LAST_LEVEL = 16, COLLISION = 32, STATIC = 64, DYNAMIC = 128;
		constexpr uint8_t OCCUPANCY_MASK = FREE | UNKNOWN | OCCUPIED;
	}

	enum class Kind { Deterministic, Probabilistic };

	constexpr int8_t kOccupied = 127;
	constexpr int8_t kUnknownOccupancy = -128;

	/// Morton code bit by bit: x is bit 0, y bit 1, z bit 2 of every triple.
	inline uint64_t morton(uint32_t x, uint32_t y, uint32_t z)
	{
		uint64_t code = 0;
		for (uint32_t b = 0; b < 20; ++b)
		{
			code |= uint64_t((x >> b) & 1u) << (3 * b);
			code |= uint64_t((y >> b) & 1u) << (3 * b + 1);
			code |= uint64_t((z >> b) & 1u) << (3 * b + 2);
		}
		return code;
	}

	struct Voxel { uint32_t x = 0, y = 0, z = 0; };

	inline Voxel unmorton(uint64_t code)
	{
		Voxel v;
		for (uint32_t b = 0; b < 20; ++b)
		{
			v.x |= uint32_t((code >> (3 * b)) & 1u) << b;
			v.y |= uint32_t((code >> (3 * b + 1)) & 1u) << b;
			v.z |= uint32_t((code >> (3 * b + 2)) & 1u) << b;
		}
		return v;
	}

	/// Levels one has to climb until the paths of two voxels meet.
	inline uint32_t common_level(uint64_t a, uint64_t b)
	{
		uint32_t level = 0;
		while (a != b)
		{
			a >>= 3;
			b >>= 3;
			++level;
		}
		return level;
	}

	/// CUDA getNewStatus: status of a parent from the OR and the AND of the occupancy bits of its children.
	inline uint8_t new_status(uint8_t child_or, uint8_t child_and, uint32_t level, uint8_t static_dynamic_or)
	{
		const uint8_t o = child_or & status::OCCUPANCY_MASK;
		const uint8_t a = child_and & status::OCCUPANCY_MASK;
		const bool single_bit = a == status::FREE || a == status::UNKNOWN || a == status::OCCUPIED;
		uint8_t result = (single_bit && o == a) ? a : uint8_t(o | status::PART);
		result |= static_dynamic_or & (status::STATIC | status::DYNAMIC);
		if (level == 1)
			result |= status::LAST_LEVEL;
		return result;
	}

	struct Node
	{
		uint32_t level = 0;
		uint64_t id = 0;       ///< Morton prefix: voxel id >> 3*level
		uint8_t status = 0;
		int8_t occupancy = 0;  ///< probabilistic tree only (0 in the deterministic tree)
		bool has_children = false;

		friend bool operator==(const Node& a, const Node& b)
		{
			return a.level == b.level && a.id == b.id && a.status == b.status && a.occupancy == b.occupancy && a.has_children == b.has_children;
		}
		friend std::ostream& operator<<(std::ostream& os, const Node& n)
		{
			return os << "{level " << n.level << ", id " << n.id << ", status " << int(n.status) << ", occupancy " << int(n.occupancy)
				<< (n.has_children ? ", children}" : "}");
		}
	};

	class OctreeReference
	{
	public:
		/// Builds the canonical tree of the given voxels (duplicates allowed).
		OctreeReference(Kind kind, const std::vector<Voxel>& voxels) : kind_(kind)
		{
			for (const Voxel& v : voxels)
			{
				if (v.x >= kCoordinateLimit || v.y >= kCoordinateLimit || v.z >= kCoordinateLimit)
					++dropped_;
				else
					ids_.push_back(morton(v.x, v.y, v.z));
			}
			std::sort(ids_.begin(), ids_.end());
			ids_.erase(std::unique(ids_.begin(), ids_.end()), ids_.end());

			if (ids_.empty())
			{
				Node root;
				root.level = kRootLevel;
				root.status = status::UNKNOWN;
				root.occupancy = kind_ == Kind::Probabilistic ? kUnknownOccupancy : int8_t(0);
				nodes_[{ kRootLevel, 0 }] = root;
			}
			else
				build(kRootLevel, 0, 0, ids_.size());
		}

		[[nodiscard]] Kind kind() const { return kind_; }
		[[nodiscard]] size_t dropped() const { return dropped_; }
		/// Distinct occupied voxels inside the 2^14 grid.
		[[nodiscard]] size_t voxel_count() const { return ids_.size(); }
		[[nodiscard]] const std::vector<uint64_t>& voxel_ids() const { return ids_; }

		/// All nodes of the memory tree, ordered by (level, id).
		[[nodiscard]] std::vector<Node> dump() const
		{
			std::vector<Node> result;
			result.reserve(nodes_.size());
			for (const auto& [key, node] : nodes_)
				result.push_back(node);
			return result;
		}

		[[nodiscard]] size_t node_count() const { return nodes_.size(); }
		[[nodiscard]] size_t node_count(uint32_t level) const
		{
			return size_t(std::count_if(nodes_.begin(), nodes_.end(), [level](const auto& kv) { return kv.first.first == level; }));
		}
		[[nodiscard]] const Node& root() const { return nodes_.at({ kRootLevel, 0 }); }

		/// The node without children that covers the voxel `id` (a level 0 leaf inside the on-path region, otherwise the
		/// uniform filler node at the highest level where the path leaves the on-path nodes).
		[[nodiscard]] const Node& find(uint64_t id) const
		{
			const Node* node = &root();
			while (node->has_children)
				node = &nodes_.at({ node->level - 1, id >> (3 * (node->level - 1)) });
			return *node;
		}

		/// The node where a lookup stops: like find(), but a uniform node (no PART flag) ends the walk, its children are ignored
		/// (CUDA treats such nodes as leaves). This is what the Vulkan find() returns.
		[[nodiscard]] const Node& lookup(uint64_t id) const
		{
			const Node* node = &root();
			while (node->has_children && (node->status & status::PART))
				node = &nodes_.at({ node->level - 1, id >> (3 * (node->level - 1)) });
			return *node;
		}

		/// Result of collide(): counts, OR of the 256 bit flags of the colliding probes and the nodes that received the COLLISION bit.
		struct CollisionResult
		{
			uint32_t collisions = 0;
			uint32_t with_unknown = 0;
			std::array<uint32_t, 8> flags{};
			std::set<std::pair<uint32_t, uint64_t>> flagged; ///< (level, id) of the nodes that would get status::COLLISION
		};

		/// The node where a collision lookup stops: the walk from the root descends into PART nodes while the node is above
		/// `min_level` (CUDA TRAVES_MACRO); at min_level the node itself is tested, PART or not.
		[[nodiscard]] const Node& stop_node(uint64_t id, uint32_t min_level) const
		{
			const Node* node = &root();
			while (node->has_children && (node->status & status::PART) && node->level > min_level)
				node = &nodes_.at({ node->level - 1, id >> (3 * (node->level - 1)) });
			return *node;
		}

		/// Collision of probe voxels (Morton ids of occupied voxels, already offset; ids >= 2^42 are outside and ignored) with the
		/// tree, CUDA kernel_intersect_*: a probe collides when the node where its lookup stops is occupied (deterministic: OCCUPIED
		/// bit, probabilistic: occupancy != -128 and >= tree_threshold); a probe hits unknown space when that node is UNKNOWN
		/// (deterministic: UNKNOWN bit, probabilistic: occupancy == -128), independent of the collision. `flags` (optional, one
		/// entry per probe) are OR-ed over the colliding probes.
		[[nodiscard]] CollisionResult collide(const std::vector<uint64_t>& ids, const std::vector<std::array<uint32_t, 8>>* flags,
			uint32_t min_level, int tree_threshold = 10) const
		{
			CollisionResult result;
			for (size_t i = 0; i < ids.size(); ++i)
			{
				if (ids[i] >= (uint64_t(1) << 42))
					continue;
				const Node& n = stop_node(ids[i], min_level);
				const bool det = kind_ == Kind::Deterministic;
				const bool occupied = det ? (n.status & status::OCCUPIED) != 0 : (n.occupancy != kUnknownOccupancy && n.occupancy >= tree_threshold);
				const bool unknown = det ? (n.status & status::UNKNOWN) != 0 : n.occupancy == kUnknownOccupancy;
				if (unknown)
					++result.with_unknown;
				if (!occupied)
					continue;
				++result.collisions;
				result.flagged.insert({ n.level, n.id });
				if (flags)
					for (size_t w = 0; w < 8; ++w)
						result.flags[w] |= (*flags)[i][w];
			}
			return result;
		}

		/// Result of collide_tree().
		struct TreeCollisionResult
		{
			uint64_t voxels = 0;
			/// Work items of a level-synchronous traversal: block_pairs[l] = pairs of blocks (8 sibling nodes each) of level l that
			/// are refined, i.e. the size of the frontier at level l (levels 13 .. 1; level 13 holds the first item, the children of the roots).
			std::array<uint64_t, 15> block_pairs{};
			std::set<std::pair<uint32_t, uint64_t>> flagged_this;
			std::set<std::pair<uint32_t, uint64_t>> flagged_other;
		};

		/// Tree vs tree after load_balancer/kernel_config/Intersect.cuh, written as a plain recursion. A pair of nodes of one level
		/// collides when both are occupied; it counts as its whole box (8^level voxels) when both are uniform (no PART) or level <=
		/// min_level; otherwise it is refined, a uniform side standing for its own children; at level 1 the leaves are compared
		/// (min_level 0). Marks: the real node of this tree, else of the other tree; at level 0 the leaf of this tree (else of the other).
		[[nodiscard]] TreeCollisionResult collide_tree(const OctreeReference& other, uint32_t min_level, int threshold = 10, bool mark = true) const
		{
			TreeCollisionResult r;
			struct Rec
			{
				const OctreeReference& a_tree;
				const OctreeReference& b_tree;
				uint32_t min_level;
				int threshold;
				bool mark;
				TreeCollisionResult& r;

				bool occupied(const OctreeReference& t, const Node& n) const
				{
					return t.kind_ == Kind::Deterministic ? (n.status & status::OCCUPIED) != 0 : (n.occupancy != kUnknownOccupancy && n.occupancy >= threshold);
				}
				const Node& child(const OctreeReference& t, const Node& n, uint64_t c) const { return t.nodes_.at({ n.level - 1, n.id * 8 + c }); }

				void visit(const Node& a, const Node& b, uint32_t level, bool a_active, bool b_active)
				{
					if (!(occupied(a_tree, a) && occupied(b_tree, b)))
						return;
					const bool end_here = min_level >= level;
					const bool a_part = a.status & status::PART, b_part = b.status & status::PART;
					const bool is_last = ((a.status | b.status) & status::LAST_LEVEL) && !end_here;
					if ((!a_part && !b_part) || end_here)
					{
						uint64_t box = 1;
						for (uint32_t i = 0; i < level; ++i)
							box *= 8;
						r.voxels += box;
						if (mark)
							(a_active ? r.flagged_this : r.flagged_other).insert(a_active ? std::make_pair(a.level, a.id) : std::make_pair(b.level, b.id));
					}
					const bool refine = !is_last && (a_part || b_part) && !end_here;
					if (refine)
						++r.block_pairs[level - 1];
					if (refine)
						for (uint64_t c = 0; c < 8; ++c)
						{
							// a uniform node (no PART flag) is its own stand-in, a PART node always owns a block
							visit(a_part ? child(a_tree, a, c) : a, b_part ? child(b_tree, b, c) : b, level - 1, a_part, b_part);
						}
					if (level == 1 && min_level == 0 && (a_part || b_part))
						for (uint64_t j = 0; j < 8; ++j)
						{
							if (a_part && b_part)
							{
								const Node& la = child(a_tree, a, j);
								const Node& lb = child(b_tree, b, j);
								if (occupied(a_tree, la) && occupied(b_tree, lb))
								{
									++r.voxels;
									if (mark)
										r.flagged_this.insert({ la.level, la.id });
								}
							}
							else if (a_part)
							{
								const Node& la = child(a_tree, a, j);
								if (occupied(b_tree, b) && occupied(a_tree, la))
								{
									++r.voxels;
									if (mark)
										r.flagged_this.insert({ la.level, la.id });
								}
							}
							else
							{
								const Node& lb = child(b_tree, b, j);
								if (occupied(a_tree, a) && occupied(b_tree, lb))
								{
									++r.voxels;
									if (mark)
										r.flagged_other.insert({ lb.level, lb.id });
								}
							}
						}
				}
			};
			Rec rec{ *this, other, min_level, threshold, mark, r };
			const Node& ra = root();
			const Node& rb = other.root();
			const bool a_part = (ra.status & status::PART) && ra.has_children, b_part = (rb.status & status::PART) && rb.has_children;
			++r.block_pairs[kRootLevel - 1]; // the first work item: the children of the two roots
			for (uint64_t c = 0; c < 8; ++c)
				rec.visit(a_part ? rec.child(*this, ra, c) : ra, b_part ? rec.child(other, rb, c) : rb, kRootLevel - 1, a_part, b_part);
			return r;
		}

		/// dump() with the COLLISION bit set on the given nodes.
		[[nodiscard]] std::vector<Node> dump_with_collisions(const std::set<std::pair<uint32_t, uint64_t>>& flagged) const
		{
			std::vector<Node> result = dump();
			for (Node& n : result)
				if (flagged.count({ n.level, n.id }))
					n.status |= status::COLLISION;
			return result;
		}

		/// Cubes of the extract kernel (octree/load_balancer/kernel_config/Extract.cuh): starting below the root, a selected node is
		/// emitted when it is uniform (no PART) or on min_level; a PART node above min_level is refined, a level 1 node emits its
		/// selected leaves. `selection` has 256 entries indexed by the status byte (empty = everything). Unordered semantics:
		/// the result is returned in traversal order.
		[[nodiscard]] std::vector<Node> extract(const std::vector<uint8_t>& selection, uint32_t min_level) const
		{
			std::vector<Node> out;
			if (!(root().status & status::PART))
				return out;
			auto selected = [&](uint8_t s) { return selection.empty() || selection[s] != 0; };
			struct Rec
			{
				const OctreeReference& tree;
				std::vector<Node>& out;
				decltype(selected)& sel;
				uint32_t min_level;
				void visit(uint32_t level, uint64_t id)
				{
					const Node& n = tree.nodes_.at({ level, id });
					const bool part = n.status & status::PART, last = n.status & status::LAST_LEVEL, at_min = level == min_level;
					if (sel(n.status) && (!part || at_min))
						out.push_back(n);
					if (part && !at_min && n.has_children)
					{
						for (uint64_t c = 0; c < 8; ++c)
						{
							if (!last)
								visit(level - 1, id * 8 + c);
							else if (const Node& leaf = tree.nodes_.at({ 0, id * 8 + c }); sel(leaf.status))
								out.push_back(leaf);
						}
					}
				}
			};
			Rec rec{ *this, out, selected, min_level };
			for (uint64_t c = 0; c < 8; ++c)
				rec.visit(kRootLevel - 1, c);
			return out;
		}

	private:
		Kind kind_;
		std::vector<uint64_t> ids_;
		size_t dropped_ = 0;
		std::map<std::pair<uint32_t, uint64_t>, Node> nodes_;

		Node filler(uint32_t level, uint64_t id) const
		{
			Node n;
			n.level = level;
			n.id = id;
			n.status = status::UNKNOWN | status::STATIC;
			if (kind_ == Kind::Probabilistic)
				n.occupancy = kUnknownOccupancy;
			if (level == 1)
				n.status |= status::LAST_LEVEL;
			return n;
		}

		/// Node `id` of `level` whose box contains the occupied voxels ids_[begin, end) (empty range = filler).
		Node build(uint32_t level, uint64_t id, size_t begin, size_t end)
		{
			Node n;
			if (begin == end)
				n = filler(level, id);
			else if (level == 0)
			{
				n.level = 0;
				n.id = id;
				n.status = status::STATIC | (kind_ == Kind::Deterministic ? status::OCCUPIED : uint8_t(0));
				n.occupancy = kind_ == Kind::Probabilistic ? kOccupied : int8_t(0);
			}
			else
			{
				n.level = level;
				n.id = id;
				n.has_children = true;
				Node children[8];
				size_t cursor = begin;
				for (uint32_t c = 0; c < 8; ++c)
				{
					const uint64_t child_id = id * 8 + c;
					size_t child_end = cursor;
					while (child_end < end && (ids_[child_end] >> (3 * (level - 1))) == child_id)
						++child_end;
					children[c] = build(level - 1, child_id, cursor, child_end);
					cursor = child_end;
				}
				n.status = parent_status(children, level, n.occupancy);
			}
			nodes_[{ level, id }] = n;
			return n;
		}

		uint8_t parent_status(const Node (&children)[8], uint32_t level, int8_t& occupancy) const
		{
			if (kind_ == Kind::Deterministic)
			{
				uint8_t o = 0, a = 0xFF, sd = 0;
				for (const Node& c : children)
				{
					o |= c.status & status::OCCUPANCY_MASK;
					a &= c.status & status::OCCUPANCY_MASK;
					sd |= c.status;
				}
				occupancy = 0;
				return new_status(o, a, level, sd);
			}
			int pmax = children[0].occupancy, pmin = children[0].occupancy;
			bool all_non_part = true;
			for (const Node& c : children)
			{
				pmax = std::max<int>(pmax, c.occupancy);
				pmin = std::min<int>(pmin, c.occupancy);
				all_non_part = all_non_part && !(c.status & status::PART);
			}
			occupancy = int8_t(pmax);
			uint8_t s = status::UNKNOWN | status::STATIC;
			if (!(pmax == pmin && all_non_part))
				s |= status::PART;
			if (level == 1)
				s |= status::LAST_LEVEL;
			return s;
		}
	};
}

#endif
