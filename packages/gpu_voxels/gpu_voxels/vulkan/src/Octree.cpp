// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/Octree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_map>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/Morton.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/Primitives.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	using namespace octree_format;

	namespace
	{
		constexpr uint32_t kLocalSize = 64;
		constexpr uint64_t kSentinelId = uint64_t(1) << 42; ///< sorts behind every valid id

		struct CountPush { uint32_t n; };
		struct PointsPush { uint32_t n, first; float voxel_side_length; };
		struct HeadsPush { uint32_t n, shift; };
		struct SensorVoxelsPush { uint32_t n; float voxel_side_length; uint32_t pad[2]; float rows[12]; };
		static_assert(sizeof(SensorVoxelsPush) == 64);
		struct RunCountsPush { uint32_t n; int32_t initial_probability, update_probability; };
		struct RayPush { uint32_t first, count, box_min[3], box_size[3], origin[3]; };
		struct BuildPush { uint32_t n, level; uint64_t chunk, prev_chunk; };
		struct FindPush { uint32_t n, pad; uint64_t root; };
		struct ExtractPush { uint32_t item_count, level, min_level, out_capacity, next_capacity; };
		struct CollidePush
		{
			uint64_t root;
			uint32_t n, mode, min_level, options;
			int32_t offset_x, offset_y, offset_z;
			uint32_t dim_x, dim_y;
			int32_t map_threshold, tree_threshold;
			uint32_t pad;
		};
		static_assert(sizeof(CollidePush) == 56);
		struct ClearPush { uint32_t count, stride, mask; };
		struct BalancePush { uint32_t stack_size, tasks; };
		constexpr uint32_t kBalanceLevels = 15;
		constexpr uint32_t kBalanceThreads = 64;
		constexpr uint32_t kBalanceItems = 8;
		constexpr uint32_t kBalanceWords = 6;
		struct TreePush { uint32_t item_count, level, min_level, mark; int32_t threshold; uint32_t next_capacity; };
		struct InsertWalkPush { uint32_t n, target; uint64_t root; };
		struct InsertHeadsPush { uint32_t n, level, target; };
		struct InsertCountPush { uint32_t n, target; };
		struct InsertNodePush { uint32_t n, datum, soft, level; uint64_t base; };
		struct InsertLinkPush { uint32_t n, level; uint64_t base, base_up; };
		struct InsertLeafPush { uint32_t n, datum; uint64_t base0; };
		struct PropagatePush { uint32_t count, level, from_counts, next_capacity, item_capacity, count_only = 0; };
		struct PropagateLeavesPush { uint32_t count, from_counts, item_capacity; };
		struct PropagateUpPush { uint32_t count, level, from_counts, item_capacity; };
		struct CompactScanPush { uint32_t count, capacity; };
		struct GarbageScanPush { uint32_t level, item_capacity, next_capacity; };
		struct CompactCopyPush { uint32_t count, level; uint64_t base, child_base; uint32_t child_block_bytes, pad; };
		constexpr uint8_t kNeedsUpdate = 128; ///< node flag: the node or something below it changed (CUDA nf_NEEDS_UPDATE)

		/// Group counts for a 1D problem of `items` invocations (2D grid when the x limit is exceeded, see octree_gid.glsl).
		struct Grid { uint32_t x, y; };
		Grid grid_for(const Context& context, uint64_t items, uint32_t max_groups_x)
		{
			const uint64_t groups = std::max<uint64_t>((items + kLocalSize - 1) / kLocalSize, 1);
			const uint64_t limit = std::max<uint64_t>(1, std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], max_groups_x));
			const uint64_t x = std::min(groups, limit);
			return { static_cast<uint32_t>(x), static_cast<uint32_t>((groups + x - 1) / x) };
		}

		uint32_t read_word(const DeviceBuffer& buffer, uint64_t index)
		{
			uint32_t w = 0;
			buffer.download(&w, 4, index * 4);
			return w;
		}
	}

	struct Octree::Impl
	{
		struct Chunk
		{
			uint64_t blocks = 0;
			DeviceBuffer buffer;
		};

		/// All chunks of one level. Build creates one chunk per level; an insert adds chunks (slice S6d), so a level is a list.
		struct Level
		{
			std::vector<Chunk> chunks;
			bool empty() const { return chunks.empty(); }
			uint64_t blocks() const
			{
				uint64_t n = 0;
				for (const auto& c : chunks)
					n += c.blocks;
				return n;
			}
			uint64_t bytes() const
			{
				uint64_t n = 0;
				for (const auto& c : chunks)
					n += c.buffer.size();
				return n;
			}
		};

		struct Kernels
		{
			Kernels(Context& c, OctreeKind kind)
				: morton_ids(c, GVL_SHADER_SPAN(octree_morton_ids_comp), 3, sizeof(CountPush)),
				  points_to_ids(c, GVL_SHADER_SPAN(octree_points_to_ids_comp), 3, sizeof(PointsPush)),
				  sensor_voxels(c, GVL_SHADER_SPAN(octree_sensor_voxels_comp), 3, sizeof(SensorVoxelsPush)),
				  run_starts(c, GVL_SHADER_SPAN(octree_run_starts_comp), 3, sizeof(CountPush)),
				  run_counts(c, GVL_SHADER_SPAN(octree_run_counts_comp), 5, sizeof(RunCountsPush)),
				  ray_bounds(c, GVL_SHADER_SPAN(octree_ray_bounds_comp), 2, sizeof(CountPush)),
				  ray_cast(c, GVL_SHADER_SPAN(octree_ray_cast_comp), 2, sizeof(RayPush)),
				  heads(c, GVL_SHADER_SPAN(octree_heads_comp), 2, sizeof(HeadsPush)),
				  compact(c, GVL_SHADER_SPAN(octree_compact_ids_comp), 4, sizeof(CountPush)),
				  build(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_build_prob_comp) : GVL_SHADER_SPAN(octree_build_det_comp), 4, sizeof(BuildPush)),
				  find(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_find_prob_comp) : GVL_SHADER_SPAN(octree_find_det_comp), 2, sizeof(FindPush)),
				  extract(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_extract_prob_comp) : GVL_SHADER_SPAN(octree_extract_det_comp), 5, sizeof(ExtractPush)),
				  collide(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_collide_prob_comp) : GVL_SHADER_SPAN(octree_collide_det_comp), 3, sizeof(CollidePush)),
				  clear_collision(c, GVL_SHADER_SPAN(octree_clear_collision_comp), 1, sizeof(ClearPush)),
				  collide_trees(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_collide_trees_prob_comp) : GVL_SHADER_SPAN(octree_collide_trees_det_comp), 3, sizeof(TreePush)),
				  lb_count(c, GVL_SHADER_SPAN(octree_lb_count_comp), 4, sizeof(BalancePush)),
				  lb_move(c, GVL_SHADER_SPAN(octree_lb_move_comp), 6, sizeof(BalancePush)),
				  lb_trees(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_lb_trees_prob_comp) : GVL_SHADER_SPAN(octree_lb_trees_det_comp), 4, 6 * sizeof(uint32_t)),
				  lb_extract(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_lb_extract_prob_comp) : GVL_SHADER_SPAN(octree_lb_extract_det_comp), 6, 5 * sizeof(uint32_t)),
				  insert_walk(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_insert_walk_prob_comp) : GVL_SHADER_SPAN(octree_insert_walk_det_comp), 3, sizeof(InsertWalkPush)),
				  insert_heads(c, GVL_SHADER_SPAN(octree_insert_heads_comp), 3, sizeof(InsertHeadsPush)),
				  insert_count(c, GVL_SHADER_SPAN(octree_insert_count_comp), 3, sizeof(InsertCountPush)),
				  insert_link(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_insert_link_prob_comp) : GVL_SHADER_SPAN(octree_insert_link_det_comp), 6, sizeof(InsertLinkPush)),
				  insert_leaves(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_insert_leaves_prob_comp) : GVL_SHADER_SPAN(octree_insert_leaves_det_comp), 4, sizeof(InsertLeafPush)),
				  prop_down(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_propagate_down_prob_comp) : GVL_SHADER_SPAN(octree_propagate_down_det_comp), 3, sizeof(PropagatePush)),
				  prop_leaves(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_propagate_leaves_prob_comp) : GVL_SHADER_SPAN(octree_propagate_leaves_det_comp), 2, sizeof(PropagateLeavesPush)),
				  insert_nodes(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_insert_nodes_prob_comp) : GVL_SHADER_SPAN(octree_insert_nodes_det_comp), 4, sizeof(InsertNodePush)),
				  prop_up(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_propagate_up_prob_comp) : GVL_SHADER_SPAN(octree_propagate_up_det_comp), 2, sizeof(PropagateUpPush)),
				  garbage_scan(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_garbage_scan_prob_comp) : GVL_SHADER_SPAN(octree_garbage_scan_det_comp), 3, sizeof(GarbageScanPush)),
				  compact_scan(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_compact_scan_prob_comp) : GVL_SHADER_SPAN(octree_compact_scan_det_comp), 4, sizeof(CompactScanPush)),
				  compact_copy(c, kind == OctreeKind::Probabilistic ? GVL_SHADER_SPAN(octree_compact_copy_prob_comp) : GVL_SHADER_SPAN(octree_compact_copy_det_comp), 2, sizeof(CompactCopyPush))
			{
			}
			ComputePipeline morton_ids, points_to_ids, sensor_voxels, run_starts, run_counts, ray_bounds, ray_cast, heads, compact, build, find, extract, collide, clear_collision, collide_trees, lb_count, lb_move, lb_trees, lb_extract;
			ComputePipeline insert_walk, insert_heads, insert_count, insert_link, insert_leaves, prop_down, prop_leaves, insert_nodes, prop_up, garbage_scan, compact_scan, compact_copy;
		};

		Impl(Context& c, OctreeKind k) : context(c), kind(k), primitives(c), kernels(c, k), levels(kLevelCount)
		{
			make_empty();
		}

		template<typename Push>
		void dispatch(CommandBatch& batch, const ComputePipeline& pipeline, std::initializer_list<const DeviceBuffer*> buffers,
			uint64_t items, const Push& push) const
		{
			const Grid g = grid_for(context, items, max_groups_x);
			batch.dispatch(pipeline, buffers, g.x, g.y, 1, &push, static_cast<uint32_t>(sizeof(Push)));
		}

		Context& context;
		OctreeKind kind;
		uint32_t max_groups_x = 65535;
		float voxel_side_length = 0.0f; ///< metres; 0 = not set (point cloud inserts throw)
		mutable Primitives primitives; // recording scans does not change the tree
		Kernels kernels;
		std::vector<Level> levels; ///< index = level
		bool host_garbage_count = false; ///< test hook: count the garbage with the host walk only
		mutable uint64_t garbage_fallbacks = 0; ///< test hook: how often the GPU count gave up and the host walk was used
		bool may_have_garbage = false; ///< an insert replaced a child pointer: blocks may be allocated but unreachable (counted by a walk)
		uint32_t voxels = 0;
		uint64_t dropped = 0;
		OctreeRebuildPolicy policy;
		OctreeBalancerConfig balancer;
		mutable OctreeBalancerStats balancer_stats; // statistics of the last traversal, written by const traversals

		// ------------------------------------------------------------------------------------------------ building

		void reset_chunks()
		{
			for (auto& l : levels)
				l.chunks.clear();
			may_have_garbage = false;
			voxels = 0;
			dropped = 0;
			pending_bound = 0;
			pending_bounded = true;
		}

		void make_empty()
		{
			reset_chunks();
			std::vector<uint8_t> block(inner_block_bytes(kind), 0);
			HostNode root = filler(kind, kRootLevel);
			root.status = kUnknown; // an empty tree: nothing was built, nothing is static
			encode_inner(kind, block.data(), 0, root);
			for (uint32_t s = 1; s < 8; ++s)
				encode_inner(kind, block.data(), s, filler(kind, kRootLevel));
			add_chunk(kRootLevel, 1, DeviceBuffer(context, block.size()));
			levels[kRootLevel].chunks.front().buffer.upload(block.data(), block.size());
		}

		static void check_chunk(const DeviceBuffer& buffer)
		{
			const uint64_t address = buffer.device_address();
			if (address % 16 != 0 || !addressable(address, buffer.size()))
				throw VulkanError("Octree: a chunk lies outside the 48 bit / 16 byte aligned address range the nodes can hold (address " + std::to_string(address) + ", size " + std::to_string(buffer.size()) + ")");
		}

		void add_chunk(uint32_t level, uint64_t blocks, DeviceBuffer buffer)
		{
			check_chunk(buffer);
			levels[level].chunks.push_back(Chunk{ blocks, std::move(buffer) });
		}

		/// Sorts the ids of `ids` (n entries, 43 bit keys) and builds the tree from the valid prefix of `n_valid` entries.
		void sort_and_build(DeviceBuffer& ids, uint32_t n, uint32_t n_valid, uint32_t key_bits)
		{
			if (n > 1)
			{
				DeviceBuffer values(context, uint64_t(n) * 4);
				KeepAlive keep_alive;
				CommandBatch batch(context);
				primitives.record_sort_pairs64(batch, ids, values, n, key_bits, keep_alive);
				batch.submit_and_wait();
			}
			if (n_valid == 0)
			{
				make_empty();
				return;
			}
			build_sorted(ids, n_valid);
		}

		/// `ids`: sorted ids, the first n of them valid (< 2^42). Replaces the tree.
		void build_sorted(DeviceBuffer& ids, uint32_t n)
		{
			// distinct ids
			DeviceBuffer heads(context, uint64_t(n) * 4), offsets(context, uint64_t(n) * 4), unique_ids(context, uint64_t(n) * 8);
			{
				KeepAlive keep_alive;
				CommandBatch batch(context);
				dispatch(batch, kernels.heads, { &ids, &heads }, n, HeadsPush{ n, 0 });
				batch.barrier();
				primitives.record_exclusive_scan(batch, heads, offsets, n, keep_alive);
				batch.barrier();
				dispatch(batch, kernels.compact, { &ids, &heads, &offsets, &unique_ids }, n, CountPush{ n });
				batch.submit_and_wait();
			}
			const uint32_t distinct = read_word(offsets, n - 1) + read_word(heads, n - 1);

			std::vector<Level> built(kLevelCount);
			DeviceBuffer current = std::move(unique_ids);
			uint32_t count = distinct;
			uint64_t previous_address = 0;
			for (uint32_t level = 0; level < kLevelCount; ++level)
			{
				DeviceBuffer level_heads(context, uint64_t(count) * 4), level_offsets(context, uint64_t(count) * 4);
				{
					KeepAlive keep_alive;
					CommandBatch batch(context);
					dispatch(batch, kernels.heads, { &current, &level_heads }, count, HeadsPush{ count, 3 * (level + 1) });
					batch.barrier();
					primitives.record_exclusive_scan(batch, level_heads, level_offsets, count, keep_alive);
					batch.submit_and_wait();
				}
				const uint64_t blocks = uint64_t(read_word(level_offsets, count - 1)) + read_word(level_heads, count - 1);
				if (level == kRootLevel && blocks != 1)
					throw VulkanError("Octree::build: more than one root block (internal error)");

				DeviceBuffer chunk(context, blocks * block_bytes(kind, level));
				DeviceBuffer next_ids(context, blocks * 8);
				const uint64_t address = chunk.device_address();
				if (address % 16 != 0 || !addressable(address, chunk.size()))
					throw VulkanError("Octree::build: a chunk lies outside the 48 bit / 16 byte aligned address range the nodes can hold (address " + std::to_string(address) + ", size " + std::to_string(chunk.size()) + ")");
				{
					CommandBatch batch(context);
					dispatch(batch, kernels.build, { &current, &level_heads, &level_offsets, &next_ids }, count,
						BuildPush{ count, level, address, previous_address });
					batch.submit_and_wait();
				}
				built[level].chunks.push_back(Chunk{ blocks, std::move(chunk) });
				previous_address = address;
				current = std::move(next_ids);
				count = static_cast<uint32_t>(blocks);
			}
			levels = std::move(built);
			may_have_garbage = false;
			voxels = distinct;
		}

		void build_coordinates(std::span<const OctreeVoxel> input)
		{
			reset_chunks();
			if (input.size() > kMaxVoxels)
				throw VulkanError("Octree::build: too many voxels in one build");
			if (input.empty())
			{
				make_empty();
				return;
			}
			const uint32_t n = static_cast<uint32_t>(input.size());
			static_assert(sizeof(OctreeVoxel) == 12);
			DeviceBuffer coords(context, uint64_t(n) * 12), ids(context, uint64_t(n) * 8), dropped_buffer(context, 4);
			coords.upload(input.data(), uint64_t(n) * 12);
			dropped_buffer.upload<uint32_t>(std::vector<uint32_t>{ 0u });
			{
				DeviceBuffer values(context, uint64_t(n) * 4);
				KeepAlive keep_alive;
				CommandBatch batch(context);
				dispatch(batch, kernels.morton_ids, { &coords, &ids, &dropped_buffer }, n, CountPush{ n });
				batch.barrier();
				if (n > 1)
					primitives.record_sort_pairs64(batch, ids, values, n, 43, keep_alive); // sentinel 2^42 needs 43 bits
				batch.submit_and_wait();
			}
			const uint32_t dropped_now = read_word(dropped_buffer, 0);
			if (n == dropped_now)
				make_empty();
			else
				build_sorted(ids, n - dropped_now);
			dropped = dropped_now;
		}

		void build_ids(std::span<const uint64_t> input)
		{
			reset_chunks();
			std::vector<uint64_t> valid;
			valid.reserve(input.size());
			for (uint64_t id : input)
				if (id < kSentinelId)
					valid.push_back(id);
			if (valid.size() > kMaxVoxels)
				throw VulkanError("Octree::build_from_ids: too many voxels in one build");
			const uint64_t dropped_now = input.size() - valid.size();
			if (valid.empty())
				make_empty();
			else
			{
				const uint32_t n = static_cast<uint32_t>(valid.size());
				DeviceBuffer ids(context, uint64_t(n) * 8);
				ids.upload<uint64_t>(valid);
				sort_and_build(ids, n, n, 42);
			}
			dropped = dropped_now;
		}

		/// Replaces the tree by the given node list (the format of dump()): every node of a block must be listed, a node with
		/// children needs its child block, every block needs a parent. The blocks of a level are distributed over chunks of at
		/// most `max_blocks` blocks (0 = one chunk per level). Everything is checked before the old tree is touched.
		void load_nodes(std::span<const OctreeNode> nodes, uint64_t max_blocks)
		{
			if (max_blocks == 0)
				max_blocks = UINT64_MAX;
			using Slots = std::array<const OctreeNode*, 8>;
			std::vector<std::map<uint64_t, Slots>> blocks(kLevelCount); // per level: block id (= parent id) -> its 8 nodes
			const OctreeNode* root = nullptr;
			for (const OctreeNode& n : nodes)
			{
				if (n.level >= kLevelCount)
					throw VulkanError("Octree::load_nodes: node level out of range");
				if (n.level == kRootLevel)
				{
					if (n.id != 0 || root)
						throw VulkanError("Octree::load_nodes: the root must be listed exactly once with id 0");
					root = &n;
					continue;
				}
				if ((n.id >> (3 * (kRootLevel - n.level))) != 0)
					throw VulkanError("Octree::load_nodes: node id out of range for its level");
				const OctreeNode*& slot = blocks[n.level][n.id >> 3][n.id & 7];
				if (slot)
					throw VulkanError("Octree::load_nodes: a node is listed twice");
				slot = &n;
			}
			if (!root)
				throw VulkanError("Octree::load_nodes: the root is missing");
			if (root->has_children && blocks[kRootLevel - 1].find(0) == blocks[kRootLevel - 1].end())
				throw VulkanError("Octree::load_nodes: the root has children but its child block is not listed");
			for (uint32_t level = 0; level < kRootLevel; ++level)
			{
				for (const auto& [block_id, slots] : blocks[level])
				{
					for (const OctreeNode* node : slots)
						if (!node)
							throw VulkanError("Octree::load_nodes: an incomplete block on level " + std::to_string(level));
					if (level == 0)
						for (const OctreeNode* node : slots)
							if (node->has_children)
								throw VulkanError("Octree::load_nodes: a leaf with children");
				}
				// children of this level's nodes must be exactly the blocks of the level below, the parents are the nodes above
				if (level > 0)
				{
					uint64_t children = 0;
					for (const auto& [block_id, slots] : blocks[level])
						for (const OctreeNode* node : slots)
							if (node->has_children)
							{
								++children;
								if (blocks[level - 1].find(node->id) == blocks[level - 1].end())
									throw VulkanError("Octree::load_nodes: a node has children but its child block is not listed");
							}
					if (children != blocks[level - 1].size())
						throw VulkanError("Octree::load_nodes: a block on level " + std::to_string(level - 1) + " has no parent node");
				}
			}
			{
				// the top level: the blocks of level 13 are the children of the root node only
				const uint64_t expected = root->has_children ? 1 : 0;
				if (blocks[kRootLevel - 1].size() != expected)
					throw VulkanError("Octree::load_nodes: a block on level 13 has no parent node");
			}

			reset_chunks();
			try
			{
				std::vector<std::unordered_map<uint64_t, uint64_t>> address(kLevelCount); // block id -> device address
				for (uint32_t level = 0; level < kRootLevel; ++level)
				{
					const uint64_t block_size = block_bytes(kind, level);
					std::vector<const std::pair<const uint64_t, Slots>*> ordered;
					for (const auto& entry : blocks[level])
						ordered.push_back(&entry);
					for (size_t first = 0; first < ordered.size(); first += max_blocks)
					{
						const size_t count = static_cast<size_t>(std::min<uint64_t>(max_blocks, ordered.size() - first));
						DeviceBuffer buffer(context, count * block_size);
						const uint64_t base = buffer.device_address();
						std::vector<uint8_t> bytes(count * block_size, 0);
						for (size_t b = 0; b < count; ++b)
						{
							const auto& [block_id, slots] = *ordered[first + b];
							address[level][block_id] = base + b * block_size;
							uint8_t* block = bytes.data() + b * block_size;
							for (uint32_t slot = 0; slot < 8; ++slot)
							{
								const OctreeNode& n = *slots[slot];
								HostNode h;
								h.status = n.status;
								h.occupancy = n.occupancy;
								if (level == 0)
									encode_leaf(kind, block, slot, h);
								else
								{
									h.child = n.has_children ? address[level - 1].at(n.id) : 0;
									encode_inner(kind, block, slot, h);
								}
							}
						}
						buffer.upload(bytes.data(), bytes.size());
						add_chunk(level, count, std::move(buffer));
					}
				}
				std::vector<uint8_t> root_block(inner_block_bytes(kind), 0);
				HostNode h;
				h.status = root->status;
				h.occupancy = root->occupancy;
				h.child = root->has_children ? address[kRootLevel - 1].at(0) : 0;
				encode_inner(kind, root_block.data(), 0, h);
				for (uint32_t slot = 1; slot < 8; ++slot)
					encode_inner(kind, root_block.data(), slot, filler(kind, kRootLevel));
				DeviceBuffer root_buffer(context, root_block.size());
				root_buffer.upload(root_block.data(), root_block.size());
				add_chunk(kRootLevel, 1, std::move(root_buffer));
			}
			catch (...)
			{
				make_empty();
				throw;
			}
		}

		// ------------------------------------------------------------------------------------------------ dynamic updates

		/// Batches above this many ids are sorted and made distinct on the GPU (radix sort + the build's compaction), smaller ones on the host: a
		/// round trip costs about as much as sorting a few thousand ids (measured, Agent/Analysis/2026-11-01_octree_dynamic_timings.md).
		uint32_t gpu_sort_limit = 20000;

		/// Node ids of `level` -> level 0 ids (the node id shifted up; ids outside the level are counted as dropped), sorted and distinct, in a device
		/// buffer; `count` = number of ids (0 = nothing to insert).
		DeviceBuffer sorted_ids(std::span<const uint64_t> ids, uint32_t level, uint32_t& count)
		{
			const uint64_t limit = uint64_t(1) << (3 * (kRootLevel - level));
			std::vector<uint64_t> valid;
			valid.reserve(ids.size());
			for (uint64_t id : ids)
				if (id < limit)
					valid.push_back(id << (3 * level));
			dropped += ids.size() - valid.size();
			if (valid.size() > kMaxVoxels)
				throw VulkanError("Octree::insert: too many voxels in one insert");
			count = 0;
			if (valid.empty())
				return DeviceBuffer(context, 8);
			if (valid.size() <= gpu_sort_limit)
			{
				std::sort(valid.begin(), valid.end());
				valid.erase(std::unique(valid.begin(), valid.end()), valid.end());
				count = static_cast<uint32_t>(valid.size());
				DeviceBuffer buffer(context, uint64_t(count) * 8);
				buffer.upload<uint64_t>(valid);
				return buffer;
			}
			const uint32_t n = static_cast<uint32_t>(valid.size());
			DeviceBuffer sorted(context, uint64_t(n) * 8), heads(context, uint64_t(n) * 4), offsets(context, uint64_t(n) * 4), unique_ids(context, uint64_t(n) * 8);
			sorted.upload<uint64_t>(valid);
			{
				DeviceBuffer values(context, uint64_t(n) * 4);
				KeepAlive keep_alive;
				CommandBatch batch(context);
				if (n > 1)
				{
					primitives.record_sort_pairs64(batch, sorted, values, n, 42, keep_alive);
					batch.barrier();
				}
				record_distinct(batch, sorted, heads, offsets, unique_ids, n, keep_alive);
				batch.submit_and_wait();
			}
			count = distinct_count(heads, offsets, n);
			return unique_ids;
		}

		/// Records the distinct ids of the `n` sorted ids of `sorted` into `unique_ids` (heads of the runs, scan, compaction).
		void record_distinct(CommandBatch& batch, DeviceBuffer& sorted, DeviceBuffer& heads, DeviceBuffer& offsets, DeviceBuffer& unique_ids,
			uint32_t n, KeepAlive& keep_alive) const
		{
			dispatch(batch, kernels.heads, { &sorted, &heads }, n, HeadsPush{ n, 0 });
			batch.barrier();
			primitives.record_exclusive_scan(batch, heads, offsets, n, keep_alive);
			batch.barrier();
			dispatch(batch, kernels.compact, { &sorted, &heads, &offsets, &unique_ids }, n, CountPush{ n });
		}

		static uint32_t distinct_count(const DeviceBuffer& heads, const DeviceBuffer& offsets, uint32_t n)
		{
			return read_word(offsets, n - 1) + read_word(heads, n - 1);
		}

		/// Slice S6e: the `n` points first .. first + n - 1 of `points` (packed floats) -> level 0 ids on the GPU, sorted and distinct, in a device buffer;
		/// `count` = number of distinct ids (0 = nothing to insert), `dropped_now` = number of points outside the tree (added to `dropped`).
		DeviceBuffer sorted_point_ids(const DeviceBuffer& points, uint32_t first, uint32_t n, uint32_t& count, uint32_t& dropped_now)
		{
			count = 0;
			dropped_now = 0;
			if (n > kMaxVoxels)
				throw VulkanError("Octree::insert_point_cloud: too many points in one insert");
			if (n == 0)
				return DeviceBuffer(context, 8);
			DeviceBuffer ids(context, uint64_t(n) * 8), dropped_buffer(context, 4);
			{
				DeviceBuffer values(context, uint64_t(n) * 4);
				KeepAlive keep_alive;
				CommandBatch batch(context);
				batch.fill(dropped_buffer, 0);
				batch.barrier();
				dispatch(batch, kernels.points_to_ids, { &points, &ids, &dropped_buffer }, n, PointsPush{ n, first, voxel_side_length });
				batch.barrier();
				if (n > 1)
					primitives.record_sort_pairs64(batch, ids, values, n, 43, keep_alive); // sentinel 2^42 needs 43 bits
				batch.submit_and_wait();
			}
			dropped_now = read_word(dropped_buffer, 0);
			dropped += dropped_now;
			const uint32_t valid = n - dropped_now; // the sentinels sorted behind the valid ids
			if (valid == 0)
				return DeviceBuffer(context, 8);
			DeviceBuffer heads(context, uint64_t(valid) * 4), offsets(context, uint64_t(valid) * 4), unique_ids(context, uint64_t(valid) * 8);
			{
				KeepAlive keep_alive;
				CommandBatch batch(context);
				record_distinct(batch, ids, heads, offsets, unique_ids, valid, keep_alive);
				batch.submit_and_wait();
			}
			count = distinct_count(heads, offsets, valid);
			return unique_ids;
		}

		/// Hard insert of the voxels of `n` points (without propagate). \return the points outside the tree
		uint32_t insert_points(const DeviceBuffer& points, uint32_t first, uint32_t n, OctreeDatum datum)
		{
			uint32_t count = 0, dropped_now = 0;
			DeviceBuffer ids = sorted_point_ids(points, first, n, count, dropped_now);
			if (count > 0)
			{
				insert_sorted(ids, count, 0, datum, OctreeInsertMode::Hard);
				note_pending(count, true);
			}
			return dropped_now;
		}

		/// Slice S6f-1: the distinct voxels of `n` sensor points on the grid of `side` (device buffers: Morton ids, counts, occupancies as int32).
		struct VoxelRuns
		{
			explicit VoxelRuns(Context& c) : ids(c, 8), counts(c, 4), occupancy(c, 4) {}
			DeviceBuffer ids, counts, occupancy;
			uint32_t count = 0;   ///< number of distinct voxels
			uint32_t outside = 0; ///< points that were dropped
		};

		VoxelRuns voxelize(const DeviceBuffer& points, uint32_t n, const Matrix4f& pose, float side, const OctreeSensorModel& model)
		{
			if (!(side > 0.0f) || !std::isfinite(side))
				throw VulkanError("Octree::voxelize_points: the voxel side length must be positive and finite");
			if (n > kMaxVoxels)
				throw VulkanError("Octree::voxelize_points: too many points in one call");
			VoxelRuns runs(context);
			if (n == 0)
				return runs;
			SensorVoxelsPush push{ n, side, { 0, 0 }, {} };
			for (int r = 0; r < 3; ++r)
				for (int c = 0; c < 4; ++c)
					push.rows[4 * r + c] = pose(r, c);
			DeviceBuffer ids(context, uint64_t(n) * 8), dropped_buffer(context, 4);
			{
				DeviceBuffer values(context, uint64_t(n) * 4);
				KeepAlive keep_alive;
				CommandBatch batch(context);
				batch.fill(dropped_buffer, 0);
				batch.barrier();
				dispatch(batch, kernels.sensor_voxels, { &points, &ids, &dropped_buffer }, n, push);
				batch.barrier();
				if (n > 1)
					primitives.record_sort_pairs64(batch, ids, values, n, 43, keep_alive); // sentinel 2^42 needs 43 bits
				batch.submit_and_wait();
			}
			runs.outside = read_word(dropped_buffer, 0);
			const uint32_t valid = n - runs.outside; // the sentinels sorted behind the valid ids
			if (valid == 0)
				return runs;
			DeviceBuffer heads(context, uint64_t(valid) * 4), offsets(context, uint64_t(valid) * 4), unique_ids(context, uint64_t(valid) * 8);
			DeviceBuffer starts(context, uint64_t(valid) * 4), counts(context, uint64_t(valid) * 4), occupancy(context, uint64_t(valid) * 4);
			{
				KeepAlive keep_alive;
				CommandBatch batch(context);
				record_distinct(batch, ids, heads, offsets, unique_ids, valid, keep_alive);
				batch.barrier();
				dispatch(batch, kernels.run_starts, { &heads, &offsets, &starts }, valid, CountPush{ valid });
				batch.barrier();
				dispatch(batch, kernels.run_counts, { &heads, &offsets, &starts, &counts, &occupancy }, valid,
					RunCountsPush{ valid, model.initial_probability, model.update_probability });
				batch.submit_and_wait();
			}
			runs.count = distinct_count(heads, offsets, valid);
			runs.ids = std::move(unique_ids);
			runs.counts = std::move(counts);
			runs.occupancy = std::move(occupancy);
			return runs;
		}

		std::vector<OctreeVoxelRun> voxelize_points(std::span<const Point3f> points, const Matrix4f& pose, float side, const OctreeSensorModel& model,
			uint32_t* outside)
		{
			if (points.size() > kMaxVoxels)
				throw VulkanError("Octree::voxelize_points: too many points in one call");
			std::vector<OctreeVoxelRun> result;
			uint32_t dropped_now = 0;
			if (points.empty())
				voxelize(DeviceBuffer(context, 12), 0, pose, side, model); // validates the side length
			else
			{
				static_assert(sizeof(Point3f) == 12);
				DeviceBuffer buffer(context, points.size_bytes());
				buffer.upload(points.data(), points.size_bytes());
				const VoxelRuns runs = voxelize(buffer, static_cast<uint32_t>(points.size()), pose, side, model);
				dropped_now = runs.outside;
				if (runs.count > 0)
				{
					const auto ids = runs.ids.download_vector<uint64_t>(runs.count);
					const auto counts = runs.counts.download_vector<uint32_t>(runs.count);
					const auto occupancy = runs.occupancy.download_vector<int32_t>(runs.count);
					result.reserve(runs.count);
					for (uint32_t i = 0; i < runs.count; ++i)
						result.push_back({ ids[i], counts[i], static_cast<int8_t>(occupancy[i]) });
				}
			}
			if (outside)
				*outside = dropped_now;
			return result;
		}

		// ------------------------------------------------------------------------------------------------ S6f-2: free-space rays

		uint64_t ray_map_limit = uint64_t(1) << 28; ///< cells of the largest ray map
		uint64_t ray_chunk_steps = uint64_t(1) << 28; ///< walk steps (rays of a chunk times the longest possible ray) per GPU submission
		uint64_t ray_submissions = 0;                 ///< GPU submissions of ray chunks so far (observable for the tests)

		/// The ray hits of `n` voxels (Morton ids in a device buffer) on the box that contains them and `origin`.
		struct RayCounts
		{
			explicit RayCounts(Context& c) : counts(c, 4) {}
			DeviceBuffer counts;
			uint32_t box_min[3] = {}, box_size[3] = {};
			uint64_t cells = 0;
		};

		RayCounts cast_rays(const DeviceBuffer& ids, uint32_t n, const OctreeVoxel& origin)
		{
			if (origin.x >= Octree::kCoordinateLimit || origin.y >= Octree::kCoordinateLimit || origin.z >= Octree::kCoordinateLimit)
				throw VulkanError("Octree::cast_free_space_rays: the sensor origin lies outside the 2^14 grid");
			RayCounts rays(context);
			if (n == 0)
				return rays;
			uint32_t bounds[6] = { UINT32_MAX, UINT32_MAX, UINT32_MAX, 0, 0, 0 };
			{
				DeviceBuffer bounds_buffer(context, sizeof(bounds));
				bounds_buffer.upload(bounds, sizeof(bounds));
				CommandBatch batch(context);
				dispatch(batch, kernels.ray_bounds, { &ids, &bounds_buffer }, n, CountPush{ n });
				batch.submit_and_wait();
				bounds_buffer.download(bounds, sizeof(bounds));
			}
			const uint32_t o[3] = { origin.x, origin.y, origin.z };
			uint64_t cells = 1;
			uint64_t longest = 0; // 1 + dx + dy + dz of the longest ray that fits into the box
			for (int a = 0; a < 3; ++a)
			{
				rays.box_min[a] = std::min(bounds[a], o[a]);
				rays.box_size[a] = std::max(bounds[3 + a], o[a]) - rays.box_min[a] + 1;
				cells *= rays.box_size[a];
				longest += rays.box_size[a] - 1;
			}
			longest += 1;
			if (cells > ray_map_limit)
				throw VulkanError("Octree::cast_free_space_rays: the box of the rays has " + std::to_string(cells) + " cells, more than the limit of " + std::to_string(ray_map_limit));
			rays.cells = cells;
			rays.counts = DeviceBuffer(context, cells * 4);
			{
				CommandBatch batch(context);
				batch.fill(rays.counts, 0);
				batch.submit_and_wait();
			}
			const uint64_t per_chunk = std::max<uint64_t>(1, ray_chunk_steps / longest);
			for (uint64_t first = 0; first < n; first += per_chunk)
			{
				const uint32_t count = static_cast<uint32_t>(std::min<uint64_t>(per_chunk, n - first));
				CommandBatch batch(context);
				dispatch(batch, kernels.ray_cast, { &ids, &rays.counts }, count,
					RayPush{ static_cast<uint32_t>(first), count, { rays.box_min[0], rays.box_min[1], rays.box_min[2] },
						{ rays.box_size[0], rays.box_size[1], rays.box_size[2] }, { o[0], o[1], o[2] } });
				batch.submit_and_wait();
				++ray_submissions;
			}
			return rays;
		}

		OctreeRayMap cast_free_space_rays(std::span<const OctreeVoxel> free_voxels, const OctreeVoxel& origin)
		{
			if (free_voxels.size() > kMaxVoxels)
				throw VulkanError("Octree::cast_free_space_rays: too many voxels in one call");
			std::vector<uint64_t> ids;
			ids.reserve(free_voxels.size());
			for (const OctreeVoxel& v : free_voxels)
			{
				if (v.x >= Octree::kCoordinateLimit || v.y >= Octree::kCoordinateLimit || v.z >= Octree::kCoordinateLimit)
					throw VulkanError("Octree::cast_free_space_rays: a voxel lies outside the 2^14 grid");
				ids.push_back(morton::code60(v.x, v.y, v.z));
			}
			DeviceBuffer buffer(context, std::max<uint64_t>(8, ids.size() * 8));
			if (!ids.empty())
				buffer.upload<uint64_t>(ids);
			const RayCounts rays = cast_rays(buffer, static_cast<uint32_t>(ids.size()), origin);
			OctreeRayMap map;
			if (rays.cells > 0)
			{
				map.min = { rays.box_min[0], rays.box_min[1], rays.box_min[2] };
				map.size = { rays.box_size[0], rays.box_size[1], rays.box_size[2] };
				map.counts = rays.counts.download_vector<uint32_t>(rays.cells);
			}
			return map;
		}

		static OctreeDatum datum_of(uint8_t meaning, const char* function)
		{
			if (meaning == eBVM_OCCUPIED)
				return OctreeDatum::Occupied;
			if (meaning == eBVM_FREE)
				return OctreeDatum::Free;
			throw VulkanError(std::string("Octree::") + function + ": the tree stores occupied and free voxels only (meaning " + std::to_string(meaning) + ")");
		}

		void require_side_length(const char* function) const
		{
			if (!(voxel_side_length > 0.0f))
				throw VulkanError(std::string("Octree::") + function + ": the voxel side length is not set");
		}

		uint32_t insert_point_cloud(std::span<const Point3f> points, uint8_t meaning, bool propagate_now)
		{
			const OctreeDatum datum = datum_of(meaning, "insert_point_cloud");
			require_side_length("insert_point_cloud");
			if (points.size() > kMaxVoxels)
				throw VulkanError("Octree::insert_point_cloud: too many points in one insert");
			uint32_t outside = 0;
			if (!points.empty())
			{
				static_assert(sizeof(Point3f) == 12);
				DeviceBuffer buffer(context, points.size_bytes());
				buffer.upload(points.data(), points.size_bytes());
				outside = insert_points(buffer, 0, static_cast<uint32_t>(points.size()), datum);
			}
			if (propagate_now)
				propagate();
			return outside;
		}

		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings, bool propagate_now)
		{
			if (&cloud.context() != &context)
				throw VulkanError("Octree::insert_meta_point_cloud: different contexts");
			if (meanings.size() != cloud.num_clouds())
				throw VulkanError("Octree::insert_meta_point_cloud: expected one meaning per sub cloud");
			std::vector<OctreeDatum> data;
			for (uint8_t meaning : meanings)
				data.push_back(datum_of(meaning, "insert_meta_point_cloud"));
			require_side_length("insert_meta_point_cloud");
			uint32_t outside = 0;
			for (uint32_t c = 0; c < cloud.num_clouds();)
			{
				uint32_t end = c + 1;
				while (end < cloud.num_clouds() && data[end] == data[c])
					++end;
				outside += insert_points(cloud.points(), cloud.cloud_offset(c), cloud.cloud_offset(end) - cloud.cloud_offset(c), data[c]);
				// a run is propagated before the next one is inserted (as with separate calls): inserting into nodes that still wait for their
				// update gives other trees than inserting into a consistent one
				if (propagate_now || end < cloud.num_clouds())
					propagate();
				c = end;
			}
			return outside;
		}

		void insert_ids(std::span<const uint64_t> node_ids, uint32_t level, OctreeDatum datum, OctreeInsertMode mode, bool propagate_now)
		{
			if (level >= kRootLevel)
				throw VulkanError("Octree::insert_nodes: the level must be in 0..13");
			uint32_t count = 0;
			DeviceBuffer ids = sorted_ids(node_ids, level, count);
			if (count > 0)
			{
				insert_sorted(ids, count, level, datum, mode);
				note_pending(count, !(mode == OctreeInsertMode::Soft && level > 0));
			}
			if (propagate_now)
				propagate();
		}

		/// `ids`: `n` sorted, distinct level 0 ids. The steps are described in the insert shaders; the new blocks of every level from `target` up
		/// become one new chunk each. Two submits and one read back: (1) walk + count of the new blocks of all levels, (2) the numbering of the
		/// blocks (heads + scan per level), linking and the final kernel in one batch.
		void insert_sorted(DeviceBuffer& id_buffer, uint32_t n, uint32_t target, OctreeDatum datum, OctreeInsertMode mode)
		{
			DeviceBuffer info(context, uint64_t(n) * 16), stats(context, 16 * 4);
			{
				CommandBatch batch(context);
				batch.fill(stats, 0);
				dispatch(batch, kernels.insert_walk, { &id_buffer, &info, &stats }, n, InsertWalkPush{ n, target, root_address() });
				batch.barrier();
				dispatch(batch, kernels.insert_count, { &id_buffer, &info, &stats }, n, InsertCountPush{ n, target });
				batch.submit_and_wait();
			}
			uint32_t counted[16];
			stats.download(counted, sizeof(counted));
			if (counted[14] > 0)
				may_have_garbage = true;
			uint64_t totals[kRootLevel] = {};
			for (uint32_t level = 0; level < kRootLevel; ++level)
				totals[level] = counted[level];

			uint64_t base[kRootLevel] = {};
			for (uint32_t level = 0; level < kRootLevel; ++level)
				if (totals[level] > 0)
				{
					DeviceBuffer chunk(context, totals[level] * block_bytes(kind, level));
					base[level] = chunk.device_address();
					add_chunk(level, totals[level], std::move(chunk));
				}

			DeviceBuffer heads_a(context, uint64_t(n) * 4), excl_a(context, uint64_t(n) * 4), heads_b(context, uint64_t(n) * 4), excl_b(context, uint64_t(n) * 4);
			// from the top: number and initialise the new blocks of a level and link them into their parents
			DeviceBuffer* heads_cur = &heads_a;
			DeviceBuffer* excl_cur = &excl_a;
			DeviceBuffer* heads_up = &heads_b;
			DeviceBuffer* excl_up = &excl_b;
			KeepAlive keep_alive;
			CommandBatch batch(context);
			for (int level = int(kRootLevel) - 1; level >= 0; --level)
			{
				if (totals[level] == 0)
					continue;
				dispatch(batch, kernels.insert_heads, { &id_buffer, &info, heads_cur }, n, InsertHeadsPush{ n, uint32_t(level), target });
				batch.barrier();
				primitives.record_exclusive_scan(batch, *heads_cur, *excl_cur, n, keep_alive);
				batch.barrier();
				dispatch(batch, kernels.insert_link, { &id_buffer, &info, heads_cur, excl_cur, heads_up, excl_up }, n,
					InsertLinkPush{ n, uint32_t(level), base[level], level + 1 < int(kRootLevel) ? base[level + 1] : 0 });
				batch.barrier();
				std::swap(heads_cur, heads_up);
				std::swap(excl_cur, excl_up);
			}
			// the arrays of the target level are now in *_up (or unused when no voxel needs a new block there)
			const uint32_t datum_bits = datum == OctreeDatum::Occupied ? uint32_t(kOccupied) : uint32_t(kFree);
			if (target == 0)
				dispatch(batch, kernels.insert_leaves, { &id_buffer, &info, heads_up, excl_up }, n, InsertLeafPush{ n, datum_bits, base[0] });
			else
				dispatch(batch, kernels.insert_nodes, { &id_buffer, &info, heads_up, excl_up }, n,
					InsertNodePush{ n, datum_bits, mode == OctreeInsertMode::Soft ? 1u : 0u, target, base[target] });
			batch.submit_and_wait();
		}

		/// Upper bound for the nodes of one level that carry NEEDS_UPDATE since the last propagate: an inserted id marks at most one node per level (and
		/// only those nodes with PART are visited), so with this bound the work lists fit in buffers allocated up front and propagate needs no
		/// read back (the counts stay on the GPU). Unknown after a soft insert (UPDATE_SUBTREE reaches whole subtrees); big bounds take the exact path.
		uint64_t pending_bound = 0;
		bool pending_bounded = true;
		uint64_t bounded_propagate_limit = uint64_t(1) << 17;
		/// The exact propagate path sizes a work list exactly (count pass first) when 8 entries per node of the level above would take more than this many bytes.
		uint64_t exact_list_threshold = uint64_t(4) << 20;

		void note_pending(uint64_t ids, bool bounded)
		{
			pending_bound += ids;
			pending_bounded = pending_bounded && bounded;
		}

		void forget_pending()
		{
			pending_bound = 0;
			pending_bounded = true;
		}

		void propagate()
		{
			std::vector<uint8_t> root_block(inner_block_bytes(kind));
			levels[kRootLevel].chunks.front().buffer.download(root_block.data(), root_block.size());
			HostNode root = decode_inner(kind, root_block.data(), 0);
			if (!(root.status & kPart) || !(root.flags & kNeedsUpdate))
			{
				if (root.flags & kNeedsUpdate)
				{
					root.flags = uint8_t(root.flags & ~kNeedsUpdate);
					encode_inner(kind, root_block.data(), 0, root);
					levels[kRootLevel].chunks.front().buffer.upload(root_block.data(), root_block.size());
				}
				forget_pending();
				return;
			}
			if (pending_bounded && pending_bound > 0 && pending_bound <= bounded_propagate_limit)
				propagate_bounded();
			else
				propagate_exact();
			forget_pending();
		}

		/// All levels in one batch: the lists are sized from `pending_bound`, the kernels read and write the counts in `counts` (one word per level).
		void propagate_bounded()
		{
			uint64_t capacity[kLevelCount] = {};
			capacity[kRootLevel] = 1;
			for (int level = int(kRootLevel) - 1; level >= 2; --level)
				capacity[level] = std::min<uint64_t>(pending_bound, 8 * levels[level].blocks());
			capacity[1] = std::min<uint64_t>(8 * pending_bound, 8 * levels[1].blocks()); // every PART child of a level 2 node is recomputed
			std::vector<std::optional<DeviceBuffer>> list(kLevelCount);
			for (uint32_t level = 1; level < kLevelCount; ++level)
				list[level] = DeviceBuffer(context, std::max<uint64_t>(16, capacity[level] * 16));
			DeviceBuffer counts(context, 16 * 4);
			std::vector<uint32_t> initial(16, 0u);
			initial[kRootLevel] = 1;
			counts.upload<uint32_t>(initial);
			const uint64_t address = root_address();
			list[kRootLevel]->upload<uint32_t>(std::vector<uint32_t>{ uint32_t(address), uint32_t(address >> 32), 0u, 0u });

			CommandBatch batch(context);
			int lowest = int(kRootLevel) + 1; // lowest level whose top-down step ran
			for (int level = int(kRootLevel); level >= 2 && capacity[level] > 0; --level)
			{
				dispatch(batch, kernels.prop_down, { &*list[level], &*list[level - 1], &counts }, capacity[level] * 8, PropagatePush{ 0, uint32_t(level), 1, uint32_t(capacity[level - 1]), uint32_t(capacity[level]) });
				batch.barrier();
				lowest = level;
			}
			if (lowest == 2 && capacity[1] > 0)
			{
				dispatch(batch, kernels.prop_leaves, { &*list[1], &counts }, capacity[1], PropagateLeavesPush{ 0, 1, uint32_t(capacity[1]) });
				batch.barrier();
			}
			for (uint32_t level = 2; level < kLevelCount; ++level)
				if (capacity[level] > 0)
				{
					dispatch(batch, kernels.prop_up, { &*list[level], &counts }, capacity[level], PropagateUpPush{ 0, level, 1, uint32_t(capacity[level]) });
					batch.barrier();
				}
			batch.submit_and_wait();
			uint32_t reached[16];
			counts.download(reached, sizeof(reached));
			for (uint32_t level = 1; level < kRootLevel; ++level)
				if (reached[level] > capacity[level])
					throw VulkanError("Octree::propagate: more nodes to update on level " + std::to_string(level) + " than the bound allows (internal error)");
		}

		/// The lists are sized from the counts read back after every level (needed when the number of nodes is not bounded by the inserts).
		void propagate_exact()
		{
			std::vector<std::optional<DeviceBuffer>> list(kLevelCount);
			std::vector<uint32_t> count(kLevelCount, 0);
			{
				DeviceBuffer first(context, 16); // a record: address low, address high, inherit UPDATE_SUBTREE, unused
				const uint64_t address = root_address();
				first.upload<uint32_t>(std::vector<uint32_t>{ uint32_t(address), uint32_t(address >> 32), 0u, 0u });
				list[kRootLevel] = std::move(first);
				count[kRootLevel] = 1;
			}
			// top-down: the nodes to update of every level, expanding fresh nodes on the way. The list of the next level needs at most 8 entries per
			// node of this level: small lists are allocated with that size and the number of entries is read back after the pass. A big list is
			// sized exactly instead (a count pass of the kernel, which changes nothing, tells how many nodes will enter it, one more round trip):
			// allocating 8 times the nodes cost 70 % of the time for a million pending ids.
			DeviceBuffer no_list(context, 16); // bound to the next list slot of a count pass, never written
			auto zero_counter = [&]()
			{
				DeviceBuffer counter(context, 16);
				counter.upload<uint32_t>(std::vector<uint32_t>(4, 0u));
				return counter;
			};
			for (int level = int(kRootLevel); level >= 2 && count[level] > 0; --level)
			{
				const uint64_t worst = uint64_t(count[level]) * 8;
				uint64_t capacity = worst;
				if (worst * 16 > exact_list_threshold)
				{
					DeviceBuffer counter = zero_counter();
					CommandBatch batch(context);
					dispatch(batch, kernels.prop_down, { &*list[level], &no_list, &counter }, worst, PropagatePush{ count[level], uint32_t(level), 0, 0, UINT32_MAX, 1 });
					batch.submit_and_wait();
					capacity = read_word(counter, 1);
				}
				DeviceBuffer next(context, std::max<uint64_t>(16, capacity * 16));
				DeviceBuffer counter = zero_counter();
				CommandBatch batch(context);
				dispatch(batch, kernels.prop_down, { &*list[level], &next, &counter }, worst, PropagatePush{ count[level], uint32_t(level), 0, uint32_t(capacity), UINT32_MAX });
				batch.submit_and_wait();
				count[level - 1] = read_word(counter, 0);
				if (count[level - 1] > capacity)
					throw VulkanError("Octree::propagate: the top-down pass of level " + std::to_string(level) + " listed " + std::to_string(count[level - 1]) +
						" nodes where " + std::to_string(capacity) + " fit (internal error)");
				list[level - 1] = std::move(next);
			}
			// bottom-up: level 1 from its leaves, then every level from its children
			DeviceBuffer unused_counts(context, 16 * 4);
			CommandBatch batch(context);
			bool any = false;
			if (count[1] > 0)
			{
				dispatch(batch, kernels.prop_leaves, { &*list[1], &unused_counts }, count[1], PropagateLeavesPush{ count[1], 0, UINT32_MAX });
				any = true;
			}
			for (uint32_t level = 2; level < kLevelCount; ++level)
				if (count[level] > 0)
				{
					if (any)
						batch.barrier();
					dispatch(batch, kernels.prop_up, { &*list[level], &unused_counts }, count[level], PropagateUpPush{ count[level], level, 0, UINT32_MAX });
					any = true;
				}
			if (any)
				batch.submit_and_wait();
		}

		// ------------------------------------------------------------------------------------------------ rebuild

		/// Compacts the tree (see Octree::rebuild): step 1 walks the live blocks level by level from the root and collects their old addresses
		/// (14 small dispatches with a read back of the counter each), step 2 allocates the new chunks with the exact sizes and copies every level
		/// in one batch, fixing the child pointers. The old chunks are released only when everything succeeded.
		void rebuild()
		{
			propagate();
			std::vector<std::optional<DeviceBuffer>> blocks(kLevelCount), links(kLevelCount);
			std::vector<uint64_t> count(kLevelCount, 0);
			{
				const uint64_t root = root_address();
				DeviceBuffer first(context, 8);
				first.upload<uint32_t>(std::vector<uint32_t>{ uint32_t(root), uint32_t(root >> 32) });
				blocks[kRootLevel] = std::move(first);
				count[kRootLevel] = 1;
			}
			for (int level = int(kRootLevel); level >= 1 && count[level] > 0; --level)
			{
				// the live blocks of the level below cannot outnumber its allocated blocks (every block has one parent), nor 8 per block here
				const uint64_t capacity = std::min<uint64_t>(count[level] * 8, levels[level - 1].blocks());
				if (count[level] > UINT32_MAX / 8 || capacity > UINT32_MAX)
					throw VulkanError("Octree::rebuild: too many blocks on level " + std::to_string(level));
				DeviceBuffer next(context, std::max<uint64_t>(8, capacity * 8)), link(context, count[level] * 8 * 4), counter(context, 16);
				counter.upload<uint32_t>(std::vector<uint32_t>(4, 0u));
				CommandBatch batch(context);
				dispatch(batch, kernels.compact_scan, { &*blocks[level], &next, &link, &counter }, count[level] * 8,
					CompactScanPush{ uint32_t(count[level]), uint32_t(capacity) });
				batch.submit_and_wait();
				const uint32_t found = read_word(counter, 0);
				if (found > capacity)
					throw VulkanError("Octree::rebuild: more live blocks on level " + std::to_string(level - 1) + " than were allocated (a block is referenced twice)");
				count[level - 1] = found;
				blocks[level - 1] = std::move(next);
				links[level] = std::move(link);
			}

			// the new chunks: one per level with live blocks (the root keeps its chunk and is rewritten in place)
			std::vector<std::optional<DeviceBuffer>> fresh(kLevelCount);
			uint64_t base[kLevelCount] = {};
			base[kRootLevel] = root_address();
			for (uint32_t level = 0; level < kRootLevel; ++level)
				if (count[level] > 0)
				{
					fresh[level] = DeviceBuffer(context, count[level] * block_bytes(kind, level));
					check_chunk(*fresh[level]);
					base[level] = fresh[level]->device_address();
				}
			DeviceBuffer no_links(context, 16);
			CommandBatch batch(context);
			for (uint32_t level = 0; level <= kRootLevel; ++level)
				if (count[level] > 0)
				{
					const uint32_t child_bytes = level > 0 ? static_cast<uint32_t>(block_bytes(kind, level - 1)) : 0u;
					dispatch(batch, kernels.compact_copy, { &*blocks[level], level > 0 ? &*links[level] : &no_links },
						level > 0 ? count[level] * 8 : count[level],
						CompactCopyPush{ uint32_t(count[level]), level, base[level], level > 0 ? base[level - 1] : 0, child_bytes, 0 });
				}
			batch.submit_and_wait();

			for (uint32_t level = 0; level < kRootLevel; ++level)
			{
				levels[level].chunks.clear();
				if (fresh[level])
					levels[level].chunks.push_back(Chunk{ count[level], std::move(*fresh[level]) });
			}
			may_have_garbage = false;
		}

		// ------------------------------------------------------------------------------------------------ garbage

		/// Slice S6d-6: the allocated blocks no node points to any more (cut subtrees that lost their parent pointer), per level 0 .. 13, counted on the
		/// GPU: the reachable blocks are walked level by level from the root (every node with a child pointer, as the host walk of `analyze` does) in ONE
		/// batch, the counts stay in a small buffer (the kernels read the count of their level from it, the lists are sized by the allocated blocks) and are
		/// read back once. eturn false when a level holds more reachable blocks than were allocated (a block referenced twice: a corrupt tree), `garbage` is
		/// then unchanged and the caller takes the host walk, which reports the problem.
		bool count_garbage_on_gpu(std::array<uint64_t, kLevelCount>& garbage) const
		{
			if (levels[kRootLevel].chunks.size() != 1 || levels[kRootLevel].blocks() != 1)
				return false;
			uint64_t capacity[kLevelCount] = {};
			capacity[kRootLevel] = 1;
			uint64_t widest = 1;
			for (int level = int(kRootLevel); level >= 1; --level)
			{
				capacity[level - 1] = std::min<uint64_t>(capacity[level] * 8, levels[level - 1].blocks());
				if (capacity[level - 1] > UINT32_MAX / 8)
					return false;
				widest = std::max(widest, capacity[level - 1]);
			}
			const uint64_t root = root_address();
			DeviceBuffer root_list(context, 8), list_a(context, widest * 8), list_b(context, widest * 8), counter(context, kLevelCount * 4);
			root_list.upload<uint32_t>(std::vector<uint32_t>{ uint32_t(root), uint32_t(root >> 32) });
			std::vector<uint32_t> start(kLevelCount, 0u);
			start[kRootLevel] = 1;
			counter.upload<uint32_t>(start);
			CommandBatch batch(context);
			for (uint32_t step = 0; step < kRootLevel; ++step)
			{
				const uint32_t level = kRootLevel - step;
				const DeviceBuffer* in = step == 0 ? &root_list : (step % 2 == 1 ? &list_a : &list_b);
				DeviceBuffer* out = step % 2 == 0 ? &list_a : &list_b;
				if (step > 0)
					batch.barrier();
				dispatch(batch, kernels.garbage_scan, { in, out, &counter }, std::max<uint64_t>(capacity[level] * 8, 1),
					GarbageScanPush{ level, uint32_t(capacity[level]), level > 1 ? uint32_t(capacity[level - 1]) : 0u });
			}
			batch.submit_and_wait();
			std::vector<uint32_t> found(kLevelCount, 0u);
			counter.download(found.data(), found.size() * 4);
			for (uint32_t level = 0; level < kRootLevel; ++level)
				if (found[level] > capacity[level])
					return false;
			for (uint32_t level = 0; level < kRootLevel; ++level)
				garbage[level] = levels[level].blocks() - found[level];
			return true;
		}

		// ------------------------------------------------------------------------------------------------ inspection

		struct Snapshot
		{
			std::vector<std::vector<std::vector<uint8_t>>> bytes; ///< per level, per chunk
		};

		Snapshot snapshot() const
		{
			Snapshot s;
			s.bytes.resize(kLevelCount);
			for (uint32_t level = 0; level < kLevelCount; ++level)
			{
				s.bytes[level].resize(levels[level].chunks.size());
				for (size_t c = 0; c < levels[level].chunks.size(); ++c)
				{
					const auto& buffer = levels[level].chunks[c].buffer;
					s.bytes[level][c].resize(buffer.size());
					buffer.download(s.bytes[level][c].data(), s.bytes[level][c].size());
				}
			}
			return s;
		}

		/// Walks the memory tree from the root on the host. Collects the reachable nodes and the problems found. Blocks that are
		/// allocated but not reachable are a problem unless `may_have_garbage` (an insert replaced a child pointer); they are counted
		/// per level in `unreachable_out`. `flagged_out` counts the nodes with NEEDS_UPDATE that are reachable through PART nodes.
		void analyze(const Snapshot& snap, std::vector<OctreeNode>* nodes, std::vector<std::string>& problems,
			std::array<uint64_t, kLevelCount>* unreachable_out = nullptr, uint64_t* flagged_out = nullptr) const
		{
			auto problem = [&](const std::string& text) { if (problems.size() < 64) problems.push_back(text); };
			if (levels[kRootLevel].chunks.size() != 1 || levels[kRootLevel].blocks() != 1)
			{
				problem("the root chunk is missing or does not hold exactly one block");
				return;
			}
			// visited[level][chunk][block]
			std::vector<std::vector<std::vector<uint8_t>>> visited(kLevelCount);
			for (uint32_t level = 0; level < kLevelCount; ++level)
				for (const auto& chunk : levels[level].chunks)
					visited[level].emplace_back(chunk.blocks, uint8_t(0));
			visited[kRootLevel][0][0] = 1;

			struct Walker
			{
				const Impl& self;
				const Snapshot& snap;
				std::vector<OctreeNode>* nodes;
				std::vector<std::vector<std::vector<uint8_t>>>& visited;
				decltype(problem)& report;
				uint64_t* flagged;

				void visit(uint32_t level, uint64_t id, const HostNode& node, bool live = true)
				{
					const auto where = [&]() { return "node (level " + std::to_string(level) + ", id " + std::to_string(id) + "): "; };
					if (nodes)
						nodes->push_back(OctreeNode{ level, id, node.status, node.occupancy, level > 0 && node.child != 0 });
					if (flagged && live && (node.flags & kNeedsUpdate))
						++*flagged; // only through PART nodes: stale subtrees below a cut node keep their flags
					if (level == 0)
					{
						if (node.status & (kPart | kLastLevel))
							report(where() + "a leaf has the PART or LAST_LEVEL bit");
						return;
					}
					if (((node.status & kLastLevel) != 0) != (level == 1))
						report(where() + "LAST_LEVEL bit does not match the level");
					if ((node.status & kPart) && node.child == 0)
						report(where() + "PART without a child block");
					if (node.child == 0)
						return;

					const auto& chunks = self.levels[level - 1].chunks;
					if (chunks.empty())
					{
						report(where() + "points to a level without a chunk");
						return;
					}
					size_t index = chunks.size();
					for (size_t c = 0; c < chunks.size(); ++c)
					{
						const uint64_t chunk_base = chunks[c].buffer.device_address();
						if (node.child >= chunk_base && node.child < chunk_base + chunks[c].buffer.size())
						{
							index = c;
							break;
						}
					}
					if (index == chunks.size())
					{
						report(where() + "child pointer lies outside every chunk of level " + std::to_string(level - 1));
						return;
					}
					const uint64_t base = chunks[index].buffer.device_address();
					const uint64_t block_size = block_bytes(self.kind, level - 1);
					const uint64_t offset = node.child - base;
					if (offset % block_size != 0)
					{
						report(where() + "child pointer is not aligned to a block");
						return;
					}
					const uint64_t block = offset / block_size;
					if (visited[level - 1][index][block])
					{
						report(where() + "its child block is referenced a second time");
						return;
					}
					visited[level - 1][index][block] = 1;

					const uint8_t* data = snap.bytes[level - 1][index].data() + offset;
					HostNode kids[8];
					for (uint32_t s = 0; s < 8; ++s)
						kids[s] = level == 1 ? decode_leaf(self.kind, data, s) : decode_inner(self.kind, data, s);
					const HostNode expected = parent_of(self.kind, kids, level);
					// a node without PART may own stale children (uniform node, cut by an insert): its status is not derived from them; below
					// such a node (not `live`) nothing is recomputed any more either, a subtree created and cut before a propagate stays as it is
					// probabilistic: a dynamic update marks nodes DYNAMIC where the build says STATIC, the propagate rule does not recompute either
					const uint8_t ignore = self.kind == OctreeKind::Probabilistic ? uint8_t(kStatic | kDynamic) : uint8_t(0);
					if (live && (node.status & kPart) && (uint8_t(expected.status & ~ignore) != uint8_t(node.status & ~kCollision & ~ignore) || expected.occupancy != node.occupancy))
						report(where() + "status " + std::to_string(node.status) + " / occupancy " + std::to_string(node.occupancy) +
							" differ from the rule applied to its children (" + std::to_string(expected.status) + " / " + std::to_string(expected.occupancy) + ")");
					for (uint32_t s = 0; s < 8; ++s)
						visit(level - 1, id * 8 + s, kids[s], live && (node.status & kPart) != 0);
				}
			};
			Walker walker{ *this, snap, nodes, visited, problem, flagged_out };
			walker.visit(kRootLevel, 0, decode_inner(kind, snap.bytes[kRootLevel][0].data(), 0));

			for (uint32_t level = 0; level < kRootLevel; ++level)
			{
				if (levels[level].empty())
					continue;
				uint64_t seen = 0;
				for (const auto& chunk : visited[level])
					seen += static_cast<uint64_t>(std::count(chunk.begin(), chunk.end(), uint8_t(1)));
				const uint64_t unreachable = levels[level].blocks() - seen;
				if (unreachable_out)
					(*unreachable_out)[level] = unreachable;
				if (unreachable != 0 && !may_have_garbage)
					problem("level " + std::to_string(level) + ": " + std::to_string(unreachable) + " of " + std::to_string(levels[level].blocks()) +
						" blocks are not reachable from the root");
			}
		}

		// ------------------------------------------------------------------------------------------------ GPU queries

		uint64_t root_address() const { return levels[kRootLevel].chunks.front().buffer.device_address(); }

		std::vector<OctreeLookup> find(std::span<const uint64_t> ids) const
		{
			if (ids.empty())
				return {};
			const uint32_t n = static_cast<uint32_t>(ids.size());
			DeviceBuffer id_buffer(context, uint64_t(n) * 8), result(context, uint64_t(n) * 16);
			id_buffer.upload(ids.data(), uint64_t(n) * 8);
			CommandBatch batch(context);
			dispatch(batch, kernels.find, { &id_buffer, &result }, n, FindPush{ n, 0, root_address() });
			batch.submit_and_wait();
			const auto raw = result.download_vector<uint32_t>(uint64_t(n) * 4);
			std::vector<OctreeLookup> out(n);
			for (uint32_t i = 0; i < n; ++i)
				out[i] = OctreeLookup{ raw[4 * i], static_cast<uint8_t>(raw[4 * i + 1]), static_cast<int8_t>(raw[4 * i + 2]) };
			return out;
		}

		// ------------------------------------------------------------------------------------------------ collisions

		/// mode: 0 bit map, 1 probabilistic map, 2 bit list, 3 Morton ids (see octree_collide.glsl).
		OctreeCollisionResult collide(uint32_t mode, const DeviceBuffer* payload, const DeviceBuffer* positions, uint64_t n, Dim3ui dims,
			bool ids_have_flags, const OctreeCollisionOptions& options)
		{
			if (n > UINT32_MAX)
				throw VulkanError("Octree::collide_with: too many probes");
			if (options.compute_voxel_flags && mode == 1)
				throw VulkanError("Octree::collide_with: compute_voxel_flags needs bit vectors, a ProbVoxelMap has none");
			OctreeCollisionResult result;
			if (n == 0)
				return result;
			DeviceBuffer dummy(context, 16), results(context, 10 * 4);
			results.upload<uint32_t>(std::vector<uint32_t>(10, 0u));
			const uint32_t flags = (options.set_collision_flags ? 1u : 0u) | (options.compute_voxel_flags ? 2u : 0u) |
				(options.count_unknown ? 4u : 0u) | (ids_have_flags ? 8u : 0u);
			CommandBatch batch(context);
			dispatch(batch, kernels.collide, { payload ? payload : &dummy, positions ? positions : &dummy, &results }, n,
				CollidePush{ root_address(), static_cast<uint32_t>(n), mode, options.min_level, flags, options.offset_x, options.offset_y,
					options.offset_z, dims.x, dims.y, options.map_occupancy_threshold, options.tree_occupancy_threshold, 0 });
			batch.submit_and_wait();
			const auto raw = results.download_vector<uint32_t>(10);
			result.collisions = raw[0];
			result.collisions_with_unknown = raw[1];
			for (size_t w = 0; w < 8; ++w)
				result.voxel_flags[w] = raw[2 + w];
			return result;
		}

		// ------------------------------------------------------------------------------------------------ load balancer

		/// Runs a balanced traversal (port of AbstractLoadBalancer::run): `work` is one of the lb_* kernels with the bindings
		/// stacks, stack counts, idle counter, `extra`; `params` are its remaining push constants (the first three are the stack size,
		/// the stack threshold and the idle threshold). The traversal starts with one work item.
		void run_balanced(const ComputePipeline& work, const std::vector<const DeviceBuffer*>& extra, const std::array<uint32_t, 6>& first_item,
			const std::vector<uint32_t>& params) const
		{
			const uint32_t tasks = balancer.tasks, stack = balancer.stack_size;
			const uint32_t stack_threshold = stack + kBalanceItems - kBalanceThreads;
			const uint32_t idle_threshold = std::max<uint32_t>(1, static_cast<uint32_t>(balancer.idle_fraction * float(tasks)));
			const uint64_t stack_bytes = uint64_t(tasks) * stack * kBalanceWords * 4;
			DeviceBuffer stacks_a(context, stack_bytes), stacks_b(context, stack_bytes), counts(context, uint64_t(tasks) * 4), idle(context, 4);
			DeviceBuffer item_sums(context, (uint64_t(tasks) * kBalanceLevels + 1) * 4), scanned(context, (uint64_t(tasks) * kBalanceLevels + 1) * 4);
			DeviceBuffer inter_offsets(context, uint64_t(tasks) * kBalanceLevels * 4), balance_result(context, 4);
			DeviceBuffer* current = &stacks_a;
			DeviceBuffer* other = &stacks_b;

			std::vector<uint32_t> zero_counts(tasks, 0u);
			zero_counts[0] = 1;
			counts.upload<uint32_t>(zero_counts);
			current->upload(first_item.data(), sizeof(uint32_t) * first_item.size());
			balancer_stats = {};

			uint64_t total = 1;
			while (total > 0)
			{
				const uint32_t zero = 0;
				idle.upload(&zero, 4, 0);
				{
					std::vector<const DeviceBuffer*> buffers = { current, &counts, &idle };
					buffers.insert(buffers.end(), extra.begin(), extra.end());
					std::vector<uint32_t> push = { stack, stack_threshold, idle_threshold };
					push.insert(push.end(), params.begin(), params.end());
					CommandBatch batch(context);
					batch.dispatch(work, std::span<const DeviceBuffer* const>(buffers.data(), buffers.size()), tasks, 1, 1, push.data(), push.size() * 4);
					batch.submit_and_wait();
					++balancer_stats.kernel_launches;
				}

				// redistribute the items: count per level and stack, scan, deal out round robin
				balance_result.upload(&zero, 4);
				item_sums.upload(&zero, 4, uint64_t(tasks) * kBalanceLevels * 4); // the entry behind the sums receives the total of the scan
				{
					KeepAlive keep_alive;
					CommandBatch batch(context);
					const BalancePush push{ stack, tasks };
					batch.dispatch(kernels.lb_count, { current, &counts, &item_sums, &inter_offsets }, tasks, 1, 1, &push, sizeof(push));
					batch.barrier();
					primitives.record_exclusive_scan(batch, item_sums, scanned, tasks * kBalanceLevels + 1, keep_alive);
					batch.barrier();
					batch.dispatch(kernels.lb_move, { current, other, &counts, &scanned, &inter_offsets, &balance_result }, tasks, 1, 1, &push, sizeof(push));
					batch.submit_and_wait();
				}
				++balancer_stats.balance_steps;
				total = read_word(balance_result, 0);
				std::swap(current, other);
				if (total > 0 && (total + tasks - 1) / tasks >= stack_threshold)
					throw VulkanError("Octree load balancer: the work stacks cannot hold the work items (increase tasks or stack_size)");
			}
		}

		/// Tree vs tree by a level-synchronous wavefront (see octree_collide_trees.glsl). Returns the volume in voxels.
		uint64_t collide_trees_with(Impl& other, const OctreeTreeCollisionOptions& options)
		{
			if (other.kind != kind)
				throw VulkanError("Octree::collide_with: both trees must be of the same kind");
			if (&other == this)
				throw VulkanError("Octree::collide_with: a tree cannot collide with itself");

			auto root_of = [](const Impl& t)
			{
				std::vector<uint8_t> block(inner_block_bytes(t.kind));
				t.levels[kRootLevel].chunks.front().buffer.download(block.data(), block.size());
				return decode_inner(t.kind, block.data(), 0);
			};
			auto occupied = [&](const HostNode& n)
			{
				return kind == OctreeKind::Probabilistic ? (n.occupancy != kProbUnknown && n.occupancy >= options.occupancy_threshold)
					: (n.status & kOccupied) != 0;
			};
			const HostNode root_a = root_of(*this), root_b = root_of(other);
			const bool a_part = (root_a.status & kPart) && root_a.child != 0, b_part = (root_b.status & kPart) && root_b.child != 0;
			uint64_t box = 1;
			for (uint32_t i = 0; i < kRootLevel; ++i)
				box *= 8;
			if (!a_part && !b_part)
			{
				if (!(occupied(root_a) && occupied(root_b)))
					return 0;
				if (options.mark_collisions)
				{
					std::vector<uint8_t> block(inner_block_bytes(kind));
					levels[kRootLevel].chunks.front().buffer.download(block.data(), block.size());
					HostNode marked = root_a;
					marked.status |= kCollision;
					encode_inner(kind, block.data(), 0, marked);
					levels[kRootLevel].chunks.front().buffer.upload(block.data(), block.size());
				}
				return box;
			}

			if (options.use_load_balancer)
			{
				const uint64_t a_first = a_part ? root_a.child : root_address();
				const uint64_t b_first = b_part ? root_b.child : other.root_address();
				DeviceBuffer counters(context, 16 * 4);
				counters.upload<uint32_t>(std::vector<uint32_t>(16, 0u));
				const std::array<uint32_t, 6> first = { uint32_t(a_first), uint32_t(a_first >> 32), uint32_t(b_first), uint32_t(b_first >> 32),
					(a_part ? 1u : 0u) | (b_part ? 2u : 0u) | ((kRootLevel - 1) << 8), 0u };
				run_balanced(kernels.lb_trees, { &counters }, first, { options.min_level, options.mark_collisions ? 1u : 0u, uint32_t(options.occupancy_threshold) });
				const auto counts = counters.download_vector<uint32_t>(16);
				uint64_t volume = 0, weight = 1;
				for (uint32_t level = 0; level < kLevelCount; ++level, weight *= 8)
					volume += uint64_t(counts[1 + level]) * weight;
				return volume;
			}

			uint64_t max_items = 1;
			for (uint32_t level = 1; level < kRootLevel; ++level)
				max_items = std::max(max_items, levels[level].blocks() + other.levels[level].blocks());
			DeviceBuffer items_a(context, max_items * 24), items_b(context, max_items * 24), counters(context, 16 * 4);
			DeviceBuffer* current = &items_a;
			DeviceBuffer* next = &items_b;
			const uint64_t a_start = a_part ? root_a.child : root_address();
			const uint64_t b_start = b_part ? root_b.child : other.root_address();
			const uint32_t first[6] = { uint32_t(a_start), uint32_t(a_start >> 32), uint32_t(b_start), uint32_t(b_start >> 32),
				(a_part ? 1u : 0u) | (b_part ? 2u : 0u), 0u };
			current->upload(first, sizeof(first));
			counters.upload<uint32_t>(std::vector<uint32_t>(16, 0u));
			uint32_t item_count = 1;
			for (int level = int(kRootLevel) - 1; level >= 1 && item_count > 0; --level)
			{
				const uint32_t zero = 0;
				counters.upload(&zero, 4, 0);
				const uint64_t capacity = level >= 2 ? levels[level - 1].blocks() + other.levels[level - 1].blocks() : 0;
				CommandBatch batch(context);
				dispatch(batch, kernels.collide_trees, { current, next, &counters }, uint64_t(item_count) * 8,
					TreePush{ item_count, uint32_t(level), options.min_level, options.mark_collisions ? 1u : 0u, options.occupancy_threshold, uint32_t(capacity) });
				batch.submit_and_wait();
				const uint32_t produced = read_word(counters, 0);
				if (produced > capacity)
					throw VulkanError("Octree::collide_with: more work items than blocks (corrupt tree)");
				std::swap(current, next);
				item_count = produced;
			}
			const auto counts = counters.download_vector<uint32_t>(16);
			uint64_t volume = 0, weight = 1;
			for (uint32_t level = 0; level < kLevelCount; ++level, weight *= 8)
				volume += uint64_t(counts[1 + level]) * weight;
			return volume;
		}

		void clear_collision_flags()
		{
			CommandBatch batch(context);
			bool any = false;
			for (uint32_t level = 0; level < kLevelCount; ++level)
				for (auto& chunk : levels[level].chunks)
				{
					any = true;
					DeviceBuffer& buffer = chunk.buffer;
					const uint64_t blocks = chunk.blocks;
					ClearPush push{};
					if (level > 0)
					{
						push = { static_cast<uint32_t>(blocks * 8), uint32_t(inner_block_bytes(kind) / 32), uint32_t(kCollision) };
					}
					else
						push = { static_cast<uint32_t>(blocks * leaf_block_bytes(kind) / 4), 1u,
							kind == OctreeKind::Probabilistic ? 0x00200020u : 0x20202020u };
					dispatch(batch, kernels.clear_collision, { &buffer }, push.count, push);
				}
			if (any)
				batch.submit_and_wait();
		}

		std::vector<OctreeCube> extract(const OctreeExtractOptions& options) const
		{
			if (options.min_level >= kRootLevel)
				throw VulkanError("Octree::extract: min_level must be in 0..13");
			if (!options.status_selection.empty() && options.status_selection.size() != 256)
				throw VulkanError("Octree::extract: status_selection needs 256 entries");

			// the root is never emitted; its children are the first work item
			std::vector<uint8_t> root_block(inner_block_bytes(kind));
			levels[kRootLevel].chunks.front().buffer.download(root_block.data(), root_block.size());
			const HostNode root = decode_inner(kind, root_block.data(), 0);
			if (!(root.status & kPart) || root.child == 0)
				return {};

			std::vector<uint32_t> selection(256, 1u);
			for (size_t i = 0; i < options.status_selection.size(); ++i)
				selection[i] = options.status_selection[i] ? 1u : 0u;
			DeviceBuffer selection_buffer(context, 256 * 4);
			selection_buffer.upload<uint32_t>(selection);

			uint64_t max_blocks = 1;
			for (uint32_t level = 0; level < kRootLevel; ++level)
				max_blocks = std::max(max_blocks, levels[level].blocks());

			auto traverse = [&](DeviceBuffer& out, uint32_t capacity) -> uint32_t
			{
				if (options.use_load_balancer)
				{
					DeviceBuffer counters(context, 8);
					counters.upload<uint32_t>(std::vector<uint32_t>{ 0u, 0u });
					const std::array<uint32_t, 6> first = { uint32_t(root.child), uint32_t(root.child >> 32), 0u, 0u, (kRootLevel - 1) << 8, 0u };
					run_balanced(kernels.lb_extract, { &out, &counters, &selection_buffer }, first, { options.min_level, capacity });
					return read_word(counters, 1);
				}
				DeviceBuffer items_a(context, max_blocks * 16), items_b(context, max_blocks * 16), counters(context, 8);
				DeviceBuffer* current = &items_a;
				DeviceBuffer* next = &items_b;
				const uint32_t first[4] = { uint32_t(root.child), uint32_t(root.child >> 32), 0, 0 };
				current->upload(first, sizeof(first));
				counters.upload<uint32_t>(std::vector<uint32_t>{ 0u, 0u });
				uint32_t item_count = 1;
				for (int level = int(kRootLevel) - 1; level >= int(std::max(options.min_level, 1u)) && item_count > 0; --level)
				{
					const uint32_t zero = 0;
					counters.upload(&zero, 4, 0);
					const uint64_t next_capacity = level >= 1 ? levels[level - 1].blocks() : 0;
					CommandBatch batch(context);
					dispatch(batch, kernels.extract, { current, next, &out, &counters, &selection_buffer }, uint64_t(item_count) * 8,
						ExtractPush{ item_count, uint32_t(level), options.min_level, capacity, uint32_t(next_capacity) });
					batch.submit_and_wait();
					const uint32_t produced = read_word(counters, 0);
					if (produced > next_capacity)
						throw VulkanError("Octree::extract: more work items than blocks (corrupt tree)");
					std::swap(current, next);
					item_count = produced;
				}
				return read_word(counters, 1);
			};

			DeviceBuffer dummy(context, 16);
			const uint32_t total = traverse(dummy, 0);
			if (total == 0)
				return {};
			DeviceBuffer out(context, uint64_t(total) * 16);
			const uint32_t written = traverse(out, total);
			if (written != total)
				throw VulkanError("Octree::extract: the second pass found a different number of nodes");
			const auto raw = out.download_vector<uint32_t>(uint64_t(total) * 4);
			std::vector<OctreeCube> cubes(total);
			for (uint32_t i = 0; i < total; ++i)
			{
				const uint32_t w = raw[4 * i + 2];
				cubes[i] = OctreeCube{ w & 255u, (uint64_t(raw[4 * i + 1]) << 32) | raw[4 * i], static_cast<uint8_t>(w >> 8), static_cast<int8_t>(w >> 16) };
			}
			return cubes;
		}
	};

	// ---------------------------------------------------------------------------------------------------------- public

	namespace
	{
		void check_side_length(float voxel_side_length)
		{
			if (!std::isfinite(voxel_side_length) || voxel_side_length < 0.0f)
				throw VulkanError("Octree: the voxel side length must be positive (0 = not set)");
		}
	}

	Octree::Octree(Context& context, OctreeKind kind, float voxel_side_length)
	{
		if (!context.info().buffer_device_address)
			throw VulkanError("Octree: the device does not support bufferDeviceAddress, which the pointer-based nodes need");
		check_side_length(voxel_side_length);
		impl_ = std::make_unique<Impl>(context, kind);
		impl_->voxel_side_length = voxel_side_length;
	}

	Octree::~Octree() = default;
	Octree::Octree(Octree&&) noexcept = default;
	Octree& Octree::operator=(Octree&&) noexcept = default;

	DeviceBuffer& Octree::chunk(uint32_t level)
	{
		return chunk(level, 0);
	}

	DeviceBuffer& Octree::chunk(uint32_t level, uint32_t index)
	{
		if (level >= kLevels || index >= impl_->levels[level].chunks.size())
			throw VulkanError("Octree::chunk: no such chunk");
		return impl_->levels[level].chunks[index].buffer;
	}

	uint32_t Octree::chunk_count(uint32_t level) const
	{
		return level >= kLevels ? 0u : static_cast<uint32_t>(impl_->levels[level].chunks.size());
	}

	void Octree::set_max_groups_x(uint32_t groups) { impl_->max_groups_x = groups; }
	void Octree::test_set_host_garbage_count(bool host) { impl_->host_garbage_count = host; }
	uint64_t Octree::test_garbage_host_fallbacks() const { return impl_->garbage_fallbacks; }
	void Octree::set_gpu_sort_limit(uint32_t ids) { impl_->gpu_sort_limit = ids; }
	void Octree::set_bounded_propagate_limit(uint64_t ids) { impl_->bounded_propagate_limit = ids; }
	void Octree::set_exact_list_threshold(uint64_t bytes) { impl_->exact_list_threshold = bytes; }
	void Octree::test_set_pending_bound(uint64_t ids) { impl_->pending_bound = ids; impl_->pending_bounded = true; }

	void Octree::set_balancer_config(const OctreeBalancerConfig& config)
	{
		if (config.tasks == 0 || config.tasks > 65535 || config.stack_size < kBalanceThreads || config.stack_size > (1u << 20) ||
			!(config.idle_fraction > 0.0f && config.idle_fraction <= 1.0f))
			throw VulkanError("Octree::set_balancer_config: tasks must be 1..65535, stack_size 64..2^20 and idle_fraction in (0, 1]");
		impl_->balancer = config;
	}

	OctreeBalancerStats Octree::balancer_stats() const { return impl_->balancer_stats; }

	OctreeKind Octree::kind() const { return impl_->kind; }
	Context& Octree::context() const { return impl_->context; }
	float Octree::voxel_side_length() const { return impl_->voxel_side_length; }

	void Octree::set_voxel_side_length(float voxel_side_length)
	{
		check_side_length(voxel_side_length);
		impl_->voxel_side_length = voxel_side_length;
	}
	void Octree::build(std::span<const OctreeVoxel> voxels) { impl_->build_coordinates(voxels); }
	void Octree::build_from_ids(std::span<const uint64_t> morton_ids) { impl_->build_ids(morton_ids); }
	void Octree::load_nodes(std::span<const OctreeNode> nodes, uint64_t max_blocks_per_chunk) { impl_->load_nodes(nodes, max_blocks_per_chunk); }
	void Octree::clear() { impl_->make_empty(); }

	void Octree::insert_ids(std::span<const uint64_t> morton_ids, OctreeDatum datum, bool propagate_now)
	{
		impl_->insert_ids(morton_ids, 0, datum, OctreeInsertMode::Hard, propagate_now);
	}

	void Octree::insert_nodes(uint32_t level, std::span<const uint64_t> node_ids, OctreeDatum datum, OctreeInsertMode mode, bool propagate_now)
	{
		impl_->insert_ids(node_ids, level, datum, mode, propagate_now);
	}

	void Octree::insert(std::span<const OctreeVoxel> voxels, OctreeDatum datum, bool propagate_now)
	{
		std::vector<uint64_t> ids;
		ids.reserve(voxels.size());
		for (const OctreeVoxel& v : voxels)
			ids.push_back(v.x < (1u << 14) && v.y < (1u << 14) && v.z < (1u << 14) ? morton::code60(v.x, v.y, v.z) : kSentinelId);
		impl_->insert_ids(ids, 0, datum, OctreeInsertMode::Hard, propagate_now);
	}

	uint32_t Octree::insert_point_cloud(std::span<const Point3f> points, uint8_t meaning, bool propagate_now)
	{
		return impl_->insert_point_cloud(points, meaning, propagate_now);
	}

	uint32_t Octree::insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning, bool propagate_now)
	{
		static_assert(sizeof(Coord3ui) == sizeof(OctreeVoxel));
		const OctreeDatum datum = Impl::datum_of(meaning, "insert_coordinate_list");
		const uint64_t dropped_before = impl_->dropped;
		std::vector<OctreeVoxel> voxels;
		voxels.reserve(coordinates.size());
		for (const Coord3ui& c : coordinates)
			voxels.push_back({ c.x, c.y, c.z });
		insert(voxels, datum, propagate_now);
		return static_cast<uint32_t>(impl_->dropped - dropped_before);
	}

	uint32_t Octree::insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning, bool propagate_now)
	{
		const std::vector<uint8_t> meanings(cloud.num_clouds(), meaning);
		return impl_->insert_meta_point_cloud(cloud, meanings, propagate_now);
	}

	uint32_t Octree::insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings, bool propagate_now)
	{
		return impl_->insert_meta_point_cloud(cloud, meanings, propagate_now);
	}

	std::vector<OctreeVoxelRun> Octree::voxelize_points(std::span<const Point3f> points, const Matrix4f& pose, float voxel_side_length,
		const OctreeSensorModel& model, uint32_t* outside)
	{
		return impl_->voxelize_points(points, pose, voxel_side_length, model, outside);
	}

	OctreeRayMap Octree::cast_free_space_rays(std::span<const OctreeVoxel> free_voxels, const OctreeVoxel& origin)
	{
		return impl_->cast_free_space_rays(free_voxels, origin);
	}

	void Octree::set_ray_map_limit(uint64_t cells) { impl_->ray_map_limit = cells; }
	void Octree::test_set_ray_chunk_steps(uint64_t steps) { impl_->ray_chunk_steps = steps; }
	uint64_t Octree::test_ray_submissions() const { return impl_->ray_submissions; }

	void Octree::propagate() { impl_->propagate(); }
	void Octree::rebuild() { impl_->rebuild(); }

	bool Octree::needs_rebuild() const
	{
		const OctreeRebuildPolicy& policy = impl_->policy;
		uint64_t bytes = 0;
		for (const auto& level : impl_->levels)
			bytes += level.bytes();
		if (policy.max_bytes != 0 && bytes >= policy.max_bytes)
			return true;
		if (policy.max_garbage_fraction > 0.0 && impl_->may_have_garbage)
		{
			const OctreeStats stats = memory_usage();
			uint64_t garbage = 0;
			for (uint32_t level = 0; level < kLevels; ++level)
				garbage += stats.garbage_blocks[level] * octree_format::block_bytes(impl_->kind, level);
			return stats.bytes > 0 && double(garbage) >= policy.max_garbage_fraction * double(stats.bytes);
		}
		return false;
	}

	void Octree::set_rebuild_policy(const OctreeRebuildPolicy& policy)
	{
		if (!(policy.max_garbage_fraction >= 0.0 && policy.max_garbage_fraction <= 1.0))
			throw VulkanError("Octree::set_rebuild_policy: max_garbage_fraction must be in 0 .. 1");
		impl_->policy = policy;
	}

	OctreeRebuildPolicy Octree::rebuild_policy() const { return impl_->policy; }

	uint64_t Octree::pending_update_count() const
	{
		std::vector<std::string> problems;
		uint64_t flagged = 0;
		impl_->analyze(impl_->snapshot(), nullptr, problems, nullptr, &flagged);
		return flagged;
	}
	uint32_t Octree::voxel_count() const { return impl_->voxels; }
	uint64_t Octree::dropped_count() const { return impl_->dropped; }

	OctreeStats Octree::memory_usage() const
	{
		OctreeStats stats;
		if (impl_->may_have_garbage)
		{
			std::array<uint64_t, kLevelCount> unreachable{};
			const bool on_host = impl_->host_garbage_count || !impl_->count_garbage_on_gpu(unreachable);
			if (on_host && !impl_->host_garbage_count)
				++impl_->garbage_fallbacks;
			if (on_host)
			{
				std::vector<std::string> problems;
				impl_->analyze(impl_->snapshot(), nullptr, problems, &unreachable);
			}
			for (uint32_t level = 0; level < kLevelCount; ++level)
				stats.garbage_blocks[level] = unreachable[level];
		}
		for (uint32_t level = 0; level < kLevels; ++level)
		{
			stats.bytes += impl_->levels[level].bytes();
			stats.blocks[level] = impl_->levels[level].blocks();
			stats.chunks += static_cast<uint32_t>(impl_->levels[level].chunks.size());
		}
		return stats;
	}

	OctreeCollisionResult Octree::collide_with(const BitVoxelMap& map, const OctreeCollisionOptions& options)
	{
		return impl_->collide(0, &map.voxel_buffer(), nullptr, map.dimensions().voxel_count(), map.dimensions(), false, options);
	}

	OctreeCollisionResult Octree::collide_with(const ProbVoxelMap& map, const OctreeCollisionOptions& options)
	{
		return impl_->collide(1, &map.voxel_buffer(), nullptr, map.dimensions().voxel_count(), map.dimensions(), false, options);
	}

	OctreeCollisionResult Octree::collide_with(const BitVoxelList& list, const OctreeCollisionOptions& options)
	{
		return impl_->collide(2, &list.voxel_buffer(), &list.coordinates_buffer(), list.size(), Dim3ui{ 1, 1, 1 }, false, options);
	}

	OctreeCollisionResult Octree::collide_with_ids(std::span<const uint64_t> morton_ids, std::span<const BitVector256> flags,
		const OctreeCollisionOptions& options)
	{
		if (!flags.empty() && flags.size() != morton_ids.size())
			throw VulkanError("Octree::collide_with_ids: flags and ids differ in size");
		if (morton_ids.empty())
			return {};
		DeviceBuffer ids(impl_->context, morton_ids.size() * 8);
		ids.upload(morton_ids.data(), morton_ids.size() * 8);
		if (flags.empty())
			return impl_->collide(3, nullptr, &ids, morton_ids.size(), Dim3ui{ 1, 1, 1 }, false, options);
		DeviceBuffer flag_buffer(impl_->context, flags.size() * 32);
		flag_buffer.upload(flags.data(), flags.size() * 32);
		return impl_->collide(3, &flag_buffer, &ids, morton_ids.size(), Dim3ui{ 1, 1, 1 }, true, options);
	}

	uint64_t Octree::collide_with(Octree& other, const OctreeTreeCollisionOptions& options) { return impl_->collide_trees_with(*other.impl_, options); }

	void Octree::clear_collision_flags() { impl_->clear_collision_flags(); }

	std::vector<OctreeLookup> Octree::find(std::span<const uint64_t> morton_ids) const { return impl_->find(morton_ids); }
	std::vector<OctreeCube> Octree::extract(const OctreeExtractOptions& options) const { return impl_->extract(options); }

	std::vector<OctreeNode> Octree::dump() const
	{
		std::vector<OctreeNode> nodes;
		std::vector<std::string> problems;
		impl_->analyze(impl_->snapshot(), &nodes, problems);
		std::sort(nodes.begin(), nodes.end(), [](const OctreeNode& a, const OctreeNode& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
		return nodes;
	}

	void Octree::check_tree() const
	{
		std::vector<std::string> problems;
		impl_->analyze(impl_->snapshot(), nullptr, problems);
		if (!problems.empty())
		{
			std::ostringstream message;
			message << "Octree::check_tree: " << problems.size() << " problem(s):";
			for (size_t i = 0; i < problems.size() && i < 10; ++i)
				message << "\n  " << problems[i];
			throw VulkanError(message.str());
		}
	}
}
