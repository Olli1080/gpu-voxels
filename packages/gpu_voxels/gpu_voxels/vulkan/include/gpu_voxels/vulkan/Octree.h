// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Octree on the Vulkan backend, pointer-faithful port of the CUDA NTree (slice S6b: build, find, extract, check_tree).
 *
 * 15 levels (0 = voxels, 14 = root), branching factor 8, Morton ids (voxel coordinates < 2^14 per axis). Every level is a list of
 * chunks (buffers) of blocks of 8 sibling nodes (build creates one chunk per level, inserts add chunks); an inner node holds the 48-bit device address of its child block, so the tree
 * needs the optional feature bufferDeviceAddress (DeviceInfo::buffer_device_address); the constructor throws without it.
 * Node formats: OctreeFormat.h. Vulkan gives no help for wrong node addresses (Agent/knowledge_base/vulkan_migration/
 * octree_porting_hazards.md): check_tree() validates every pointer on the host and is meant to be called after every mutation
 * in tests.
 *
 * build() computes the final node states directly (bottom-up while the level blocks are created); this is the tree CUDA's
 * build + propagate leaves behind. Insertion, propagation and rebuild of an existing tree are slice S6d (insert / propagate / rebuild below).
 */
#ifndef GPU_VOXELS_VULKAN_OCTREE_H_INCLUDED
#define GPU_VOXELS_VULKAN_OCTREE_H_INCLUDED

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/OctreeFormat.h>
#include <gpu_voxels/vulkan/Primitives.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	class BitVoxelMap;
	class BitVoxelList;
	class MetaPointCloud;
	class ProbVoxelMap;

	struct OctreeVoxel
	{
		uint32_t x = 0, y = 0, z = 0;
	};

	/// A node of the memory tree.
	struct OctreeNode
	{
		uint32_t level = 0;
		uint64_t id = 0;       ///< Morton prefix: voxel id >> 3 * level
		uint8_t status = 0;
		int8_t occupancy = 0;  ///< probabilistic tree only
		bool has_children = false;
		friend bool operator==(const OctreeNode&, const OctreeNode&) = default;
	};

	/// Result of find(): the node where the lookup stopped.
	struct OctreeLookup
	{
		static constexpr uint32_t kOutside = 255;
		uint32_t level = 0;  ///< kOutside when the id is >= 2^42
		uint8_t status = 0;
		int8_t occupancy = 0;
		friend bool operator==(const OctreeLookup&, const OctreeLookup&) = default;
	};

	/// A node emitted by extract().
	struct OctreeCube
	{
		uint32_t level = 0;
		uint64_t id = 0;       ///< Morton prefix: voxel id >> 3 * level
		uint8_t status = 0;
		int8_t occupancy = 0;  ///< probabilistic tree only
		friend bool operator==(const OctreeCube&, const OctreeCube&) = default;
	};

	struct OctreeExtractOptions
	{
		/// Nodes of this level are emitted even when they are refined further (0..13; the root is never emitted).
		uint32_t min_level = 0;
		/// 256 flags indexed by the full status byte; empty = extract every node.
		std::vector<uint8_t> status_selection;
		/// Use the load balancing framework (persistent work groups with their own stacks) instead of the level-synchronous wavefront.
		/// Same result, different work distribution.
		bool use_load_balancer = false;
	};

	/// Options of the collision queries (CUDA intersect_sparse / intersect_morton).
	struct OctreeCollisionOptions
	{
		/// The lookup of a probe stops at this tree level (coarser = faster and conservative, 0 = exact voxels).
		uint32_t min_level = 0;
		/// Added to the voxel coordinates of maps and lists (not to Morton ids). A probe outside the 2^14 grid afterwards is ignored.
		int32_t offset_x = 0, offset_y = 0, offset_z = 0;
		/// Set the COLLISION status bit of every tree node that collided.
		bool set_collision_flags = true;
		/// OR the 256 bit vectors of the colliding probes into OctreeCollisionResult::voxel_flags (bit maps, bit lists, ids with flags).
		bool compute_voxel_flags = false;
		/// Count the probes that hit unknown space in OctreeCollisionResult::collisions_with_unknown.
		bool count_unknown = false;
		/// Probabilistic tree: a node is occupied when its occupancy is known and >= this (CUDA THRESHOLD_OCCUPANCY).
		int32_t tree_occupancy_threshold = 10;
		/// ProbVoxelMap probes: a voxel is occupied when its value is >= this (CUDA isVoxelOccupied: 50).
		int32_t map_occupancy_threshold = 50;
	};

	struct OctreeCollisionResult
	{
		uint32_t collisions = 0;               ///< colliding probes
		uint32_t collisions_with_unknown = 0;  ///< probes that hit unknown space (only with count_unknown)
		BitVector256 voxel_flags{};            ///< OR over the colliding probes (only with compute_voxel_flags)
		friend bool operator==(const OctreeCollisionResult&, const OctreeCollisionResult&) = default;
	};

	/// Options of the tree-vs-tree collision (CUDA intersect_load_balance(NTree)).
	struct OctreeTreeCollisionOptions
	{
		/// Pairs of nodes on this level or above are not refined further; a colliding pair then counts as its whole box (conservative).
		uint32_t min_level = 0;
		/// Set the COLLISION bit of the colliding nodes: of this tree, or of `other` when this tree has no real node there (CUDA marks one node per pair).
		bool mark_collisions = true;
		/// Probabilistic trees: a node is occupied when its occupancy is known and >= this.
		int32_t occupancy_threshold = 10;
		/// Use the load balancing framework instead of the level-synchronous wavefront (same result).
		bool use_load_balancer = false;
	};

	/// Parameters of the load balancing framework (CUDA AbstractLoadBalancer: 2688 tasks, stacks of 700 items, idle threshold 2/3).
	struct OctreeBalancerConfig
	{
		uint32_t tasks = 128;       ///< work groups, each with its own stack
		uint32_t stack_size = 700;  ///< work items per stack, at least 64
		float idle_fraction = 2.0f / 3.0f; ///< the kernel ends when this fraction of the work groups is idle; then the stacks are rebalanced
	};

	/// What the last balanced traversal did.
	struct OctreeBalancerStats
	{
		uint32_t kernel_launches = 0; ///< work kernels
		uint32_t balance_steps = 0;   ///< redistributions of the stacks
	};

	/// The datum of a hard insert: the voxel becomes occupied or free.
	enum class OctreeDatum : uint8_t { Occupied, Free };

	/// How an insert on a node above level 0 treats the subtree below it (CUDA hard / soft reset data). Hard: the node is set and loses PART,
	/// its subtree is cut (the block stays allocated and reachable). Soft: the node keeps PART and, in a probabilistic tree, the subtree is updated
	/// with the occupancy of the node by the next propagate (UPDATE_SUBTREE); in a deterministic tree soft cuts as well.
	enum class OctreeInsertMode : uint8_t { Hard, Soft };

	struct OctreeStats
	{
		uint64_t bytes = 0;                  ///< all chunks
		uint64_t blocks[octree_format::kLevelCount] = {}; ///< allocated blocks per level (a block holds 8 nodes), garbage included
		uint64_t garbage_blocks[octree_format::kLevelCount] = {}; ///< allocated blocks that are no longer reachable (cut subtrees)
		uint32_t chunks = 0;                 ///< all chunks of all levels
		uint64_t nodes() const { uint64_t n = 0; for (auto b : blocks) n += 8 * b; return n; }
	};

	/// When needs_rebuild() asks for a rebuild (slice S6d-4). CUDA: the allocated node memory reaches `max_bytes` (NTree::needsRebuild, default 200 MB,
	/// 0 = never). The garbage fraction is an addition of this port: inserts leave cut subtrees allocated, so a tree can be mostly garbage long
	/// before it is big. Evaluating it reads the tree back (only while garbage can exist).
	struct OctreeRebuildPolicy
	{
		uint64_t max_bytes = uint64_t(200) * 1024 * 1024; ///< rebuild when all chunks together take at least this much (0 = no limit)
		double max_garbage_fraction = 0.0;                ///< rebuild when this fraction of the bytes is garbage blocks (0 = off, at most 1)
	};

	class Octree
	{
	public:
		static constexpr uint32_t kLevels = octree_format::kLevelCount;
		static constexpr uint32_t kCoordinateLimit = 1u << 14;
		/// Maximum number of input voxels of one build (limit of the sort primitive, Primitives::kMaxSortElements).
		static constexpr uint32_t kMaxVoxels = Primitives::kMaxSortElements;

		/// \param voxel_side_length edge of a voxel in metres (CUDA: resolution, whole millimetres); needed by the point cloud inserts only,
		///        0 = not set (they throw until set_voxel_side_length)
		/// \throws VulkanError if the device lacks bufferDeviceAddress or the side length is negative or NaN. Starts as an empty tree (the root only).
		Octree(Context& context, OctreeKind kind, float voxel_side_length = 0.0f);
		~Octree();
		Octree(Octree&&) noexcept;
		Octree& operator=(Octree&&) noexcept;
		Octree(const Octree&) = delete;
		Octree& operator=(const Octree&) = delete;

		[[nodiscard]] OctreeKind kind() const;
		[[nodiscard]] Context& context() const;
		[[nodiscard]] float voxel_side_length() const;
		/// \throws VulkanError for a negative or NaN side length (0 = not set). The tree itself does not depend on it.
		void set_voxel_side_length(float voxel_side_length);

		/// Replaces the tree by the tree of the given voxels (duplicates are merged). Voxels with a coordinate >= 2^14 are dropped
		/// and counted in dropped_count() (CUDA only asserts). Blocking.
		void build(std::span<const OctreeVoxel> voxels);
		/// Same for Morton ids (voxel coordinates interleaved, see Morton.h); ids >= 2^42 are dropped.
		void build_from_ids(std::span<const uint64_t> morton_ids);
		/// Replaces the tree by the node list of dump(): the complete blocks of every reachable node (all 8 nodes of a block, the root
		/// alone), flags are cleared. The blocks of a level are spread over chunks of at most `max_blocks_per_chunk` blocks (0 = one
		/// chunk per level): the way to get a tree with several chunks per level before inserts exist, and the restore path of a
		/// stored tree. voxel_count() is 0 afterwards. \throws VulkanError for an incomplete or inconsistent list (the old tree stays).
		void load_nodes(std::span<const OctreeNode> nodes, uint64_t max_blocks_per_chunk = 0);
		/// Back to the empty tree; frees all chunks.
		void clear();

		/// Dynamic update (slices S6d-2 / S6d-3a, port of CUDA insertVoxel with the hard reset data at level 0): in a deterministic tree
		/// every voxel is set occupied or free, in a probabilistic tree a NEW voxel is set to occupancy 127 / -127 and an EXISTING one is
		/// updated (old occupancy + 127 / -127, clamped to -127 .. 127, UNKNOWN counts as 0). New blocks are added as new chunks, nodes on
		/// the paths get NEEDS_UPDATE. Duplicates are merged,
		/// ids >= 2^42 are dropped and counted in dropped_count(). With `propagate_now` the update is propagated at once, otherwise
		/// propagate() has to follow before collisions or extraction see consistent statuses. A node without PART that owned a child block
		/// (a uniform node) loses that block: it stays allocated as garbage (memory_usage().garbage_blocks) until the tree is rebuilt.
		/// voxel_count() is not updated.
		void insert_ids(std::span<const uint64_t> morton_ids, OctreeDatum datum = OctreeDatum::Occupied, bool propagate_now = true);
		void insert(std::span<const OctreeVoxel> voxels, OctreeDatum datum = OctreeDatum::Occupied, bool propagate_now = true);
		/// Slice S6d-3b: the same for whole nodes of `level` (0 .. 13) given by their ids on that level (voxel id >> 3 * level, below 8^(14 - level)),
		/// e.g. free space as big cubes. In a deterministic tree the node is set (status = occupancy bits of the datum, PART cleared); in a
		/// probabilistic tree its occupancy is SET to 127 / -127 (not added). If the path ends above the level on a node without PART, blocks are
		/// created down to the level. `mode`: see OctreeInsertMode. Level 0 is `insert_ids` (and ignores the mode: a leaf has no subtree).
		void insert_nodes(uint32_t level, std::span<const uint64_t> node_ids, OctreeDatum datum = OctreeDatum::Occupied,
			OctreeInsertMode mode = OctreeInsertMode::Hard, bool propagate_now = true);
		/// Slice S6e, the façade of CUDA GvlNTree (insertPointCloud, insertCoordinateList, insertMetaPointCloud): the voxel of a point is
		/// floor(point / voxel_side_length) like in the other Vulkan maps (CUDA truncates the float to uint32, which is undefined for negative
		/// values and puts them into voxel 0 on the device), converted on the GPU. Points outside the 2^14 grid (negative, too far, NaN, inf) are
		/// not inserted but counted in the return value and in dropped_count(). `meaning`: eBVM_OCCUPIED inserts occupied voxels; eBVM_FREE
		/// inserts free voxels (an addition: CUDA logs an error for anything but eBVM_OCCUPIED and does nothing); every other meaning throws
		/// (the tree cannot store it). Same effect as insert_ids (hard insert, duplicates merged), `propagate_now` as there.
		/// \throws VulkanError if the voxel side length is not set, for another context, for a meaning that is not supported.
		/// \return number of points that lie outside the tree
		uint32_t insert_point_cloud(std::span<const Point3f> points, uint8_t meaning = eBVM_OCCUPIED, bool propagate_now = true);
		/// Voxel coordinates (CUDA insertCoordinateList); a coordinate >= 2^14 is dropped. \return number of coordinates outside the tree
		uint32_t insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning = eBVM_OCCUPIED, bool propagate_now = true);
		/// All sub clouds of `cloud` (their points stay on the GPU) with one meaning. \return number of points outside the tree
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning = eBVM_OCCUPIED, bool propagate_now = true);
		/// One meaning per sub cloud (CUDA inserts everything with the first meaning and warns): the sub clouds are inserted in their order, runs
		/// of sub clouds with the same meaning as one batch (duplicates merged), every run propagated before the next one is inserted, so the result
		/// is that of one insert_point_cloud call per run: a voxel in an occupied and a later free run ends up as the later insert makes it (in a
		/// probabilistic tree the occupancies add up). `propagate_now` concerns the end. \throws VulkanError unless there is one meaning per sub cloud
		/// or for a meaning that is not supported (checked before anything is inserted).
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings, bool propagate_now = true);

		/// Recomputes the statuses of all nodes with NEEDS_UPDATE from their children, top-down expansion of fresh nodes included
		/// (CUDA propagate), level by level on the GPU. Nothing to do on a tree without pending updates.
		void propagate();
		/// Nodes that still carry NEEDS_UPDATE and are reachable through PART nodes (0 after propagate(); stale subtrees below a cut node
		/// keep their flags, nothing visits them); read back and walked on the host.
		[[nodiscard]] uint64_t pending_update_count() const;

		/// Slice S6d-4 (CUDA rebuild: extract every cube, build a fresh tree): propagates pending updates, then copies the nodes reachable through
		/// PART nodes into new chunks, one per level and compacted, and releases the old chunks, so cut subtrees (garbage) and the chunks of
		/// many inserts disappear. Nodes keep status, occupancy and collision bits, flags are clear; a node without PART loses the child block it
		/// may still point to (dead data: lookups, extraction and collisions stop at it). voxel_count() and dropped_count() are kept. Done on the
		/// GPU level by level (two kernels per level, peak memory = old tree + live part); if it throws the tree is unchanged.
		/// \throws VulkanError for a corrupt tree (a block referenced twice) or an allocation failure.
		void rebuild();
		/// CUDA needsRebuild: true when the tree reached the limits of rebuild_policy(). Cheap unless max_garbage_fraction is set.
		[[nodiscard]] bool needs_rebuild() const;
		/// \throws VulkanError if max_garbage_fraction is outside 0 .. 1.
		void set_rebuild_policy(const OctreeRebuildPolicy& policy);
		[[nodiscard]] OctreeRebuildPolicy rebuild_policy() const;

		/// Distinct voxels of the last build.
		[[nodiscard]] uint32_t voxel_count() const;
		/// Input voxels of the last build that were outside the tree.
		[[nodiscard]] uint64_t dropped_count() const;
		[[nodiscard]] OctreeStats memory_usage() const;

		/// Looks up every id on the GPU: the walk from the root stops at a leaf, a node without children or a uniform node
		/// (no PART flag).
		[[nodiscard]] std::vector<OctreeLookup> find(std::span<const uint64_t> morton_ids) const;
		/// Cubes of the tree on the GPU (rules of the CUDA extract kernel); the order of the result is unspecified.
		[[nodiscard]] std::vector<OctreeCube> extract(const OctreeExtractOptions& options = {}) const;

		/// All nodes reachable from the root, ordered by (level, id); read back to the host and walked there.
		[[nodiscard]] std::vector<OctreeNode> dump() const;
		/// Collision with the occupied voxels of a bit map: every voxel with a non-zero bit vector (CUDA isVoxelOccupied, all 256
		/// bits) is a probe at its coordinates + offset. Marks colliding nodes unless options.set_collision_flags is false.
		[[nodiscard]] OctreeCollisionResult collide_with(const BitVoxelMap& map, const OctreeCollisionOptions& options = {});
		/// Same for a probabilistic map (probes: value >= options.map_occupancy_threshold); compute_voxel_flags is not available.
		[[nodiscard]] OctreeCollisionResult collide_with(const ProbVoxelMap& map, const OctreeCollisionOptions& options = {});
		/// Same for a bit voxel list (entries with a non-zero bit vector at their coordinates + offset).
		[[nodiscard]] OctreeCollisionResult collide_with(const BitVoxelList& list, const OctreeCollisionOptions& options = {});
		/// Same for Morton ids (the Morton voxel list of CUDA): `flags` is empty (every id is an occupied probe) or has one bit vector per
		/// id (probes with a zero vector are ignored); the offset is not applied. Ids >= 2^42 are ignored.
		[[nodiscard]] OctreeCollisionResult collide_with_ids(std::span<const uint64_t> morton_ids, std::span<const BitVector256> flags = {},
			const OctreeCollisionOptions& options = {});
		/// Collision of two trees of the same kind by simultaneous traversal: the number of voxels (of the tree resolution) in
		/// collision. Uniform nodes (no PART flag) stand for all their descendants. Both trees must be different objects.
		/// A root without children is handled as a uniform node (CUDA dereferences a null pointer there).
		[[nodiscard]] uint64_t collide_with(Octree& other, const OctreeTreeCollisionOptions& options = {});
		/// Removes the COLLISION bit from every node.
		void clear_collision_flags();

		/// Raw access to a chunk of a level (blocks of 8 nodes, formats in OctreeFormat.h) for tests and tools; writing to it can
		/// corrupt the tree (check_tree() finds that). \throws VulkanError if there is no such chunk. `chunk(level)` is chunk 0.
		[[nodiscard]] DeviceBuffer& chunk(uint32_t level);
		[[nodiscard]] DeviceBuffer& chunk(uint32_t level, uint32_t index);
		/// Number of chunks of a level (0 for a level that holds no block).
		[[nodiscard]] uint32_t chunk_count(uint32_t level) const;
		/// Configures the load balancing framework (used by traversals with use_load_balancer). \throws VulkanError for invalid values.
		void set_balancer_config(const OctreeBalancerConfig& config);
		[[nodiscard]] OctreeBalancerStats balancer_stats() const;
		/// Test hook: caps the x group count of every dispatch (default 65535) so that the 2D dispatch path is exercised with small trees.
		void set_max_groups_x(uint32_t groups);
		/// Test hook: insert batches with more than this many ids are sorted on the GPU, smaller ones on the host (default 20000).
		void set_gpu_sort_limit(uint32_t ids);
		/// Test hooks of propagate(): the bounded single-batch path is taken while at most this many ids were inserted since the last propagate (default
		/// 131072, 0 = always the exact path with a read back per level); `test_set_pending_bound` overwrites the bound (a bound that is too small makes
		/// propagate() throw instead of writing out of bounds).
		void set_bounded_propagate_limit(uint64_t ids);
		/// Test hook of the exact propagate path: a work list is sized exactly after a count pass when 8 entries per node of the level above would take
		/// more than this many bytes (default 4 MiB; 0 = always counted first, UINT64_MAX = never: lists of 8 entries per node).
		void set_exact_list_threshold(uint64_t bytes);
		void test_set_pending_bound(uint64_t ids);
		/// Test hook: memory_usage() counts the garbage blocks with the host walk only (default: on the GPU, with the host walk as the fallback for a corrupt tree).
		void test_set_host_garbage_count(bool host);
		/// Test hook: how often memory_usage() had to take the host walk because the GPU count found more blocks than were allocated (a corrupt tree).
		[[nodiscard]] uint64_t test_garbage_host_fallbacks() const;

		/// Validates the structure on the host (pointers inside registered chunks, alignment, every block referenced once,
		/// status rules). \throws VulkanError listing the first problems.
		void check_tree() const;

	private:
		struct Impl;
		std::unique_ptr<Impl> impl_;
	};
}

#endif
