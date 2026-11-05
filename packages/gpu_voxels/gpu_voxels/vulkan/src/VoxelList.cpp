// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/VoxelList.h>

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <string>

#include <gpu_voxels/vulkan/Log.h>
#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/Primitives.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kLocalSize = 64;                   // local_size_x of all list kernels
		constexpr uint32_t kMaxAxisDimension = 1u << 24;      // exactly representable in a float (bounds checks)
		constexpr std::size_t kMaxItemsPerUpload = 4u << 20;  // points / coordinates per upload buffer
		constexpr uint32_t kBitWords = 8;

		struct EntriesPointsPush { uint32_t dim_x, dim_y, dim_z, count, meaning, kind; float voxel_side_length; };
		struct EntriesCoordsPush { uint32_t dim_x, dim_y, dim_z, count, meaning, kind; };
		struct EntriesMetaPush { uint32_t dim_x, dim_y, dim_z, count, sub_cloud_count, kind; float voxel_side_length; };
		struct CompactPush { uint32_t count, words, dst_offset; };
		struct CountPush { uint32_t n; };
		struct MergePush { uint32_t n, words, mode; };
		struct MembershipPush { uint32_t n, other_n, use_threshold; int32_t threshold; };
		struct CountGePush { uint32_t n; int32_t threshold; };
		struct GatherPush { uint32_t voxel_count, dim_x, dim_y, dim_z, dst_offset; };
		struct CollidePush
		{
			uint32_t n, dim_x, dim_y, dim_z;
			int32_t off_x, off_y, off_z;
			uint32_t mode, map_kind;
			int32_t level;
			float threshold;
			uint32_t mask[8];
		};
		constexpr uint32_t kCollideResultWords = 9; // count + the OR of the bits of 8 words

		uint32_t group_count(const Context& context, uint64_t items)
		{
			const uint64_t needed = (items + kLocalSize - 1) / kLocalSize;
			const uint64_t cap = std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], 65535);
			return static_cast<uint32_t>(std::clamp<uint64_t>(needed, 1, cap));
		}

		uint32_t words_of(VoxelListBase::Kind kind)
		{
			return kind == VoxelListBase::Kind::Bit ? kBitWords : 1u;
		}

		/// The three parallel arrays of a list. Buffers hold at least one element so that empty lists stay bindable.
		struct Entries
		{
			Entries(Context& context, uint32_t n, uint32_t words)
				: ids(context, sizeof(uint32_t) * std::max<uint64_t>(n, 1)),
				  coords(context, 3 * sizeof(uint32_t) * std::max<uint64_t>(n, 1)),
				  voxels(context, sizeof(uint32_t) * words * std::max<uint64_t>(n, 1)),
				  count(n)
			{
			}

			DeviceBuffer ids;
			DeviceBuffer coords;
			DeviceBuffer voxels;
			uint32_t count;
		};

		DeviceBuffer make_words(Context& context, uint64_t words)
		{
			return DeviceBuffer(context, sizeof(uint32_t) * std::max<uint64_t>(words, 1));
		}

		/// Last element of an exclusive scan plus the last flag = sum of all flags.
		uint32_t total_of_scan(const DeviceBuffer& flags, const DeviceBuffer& scan, uint32_t n)
		{
			const uint64_t offset = static_cast<uint64_t>(n - 1) * sizeof(uint32_t);
			uint32_t flag = 0;
			uint32_t position = 0;
			flags.download(&flag, sizeof(uint32_t), offset);
			scan.download(&position, sizeof(uint32_t), offset);
			return flag + position;
		}
	}

	struct VoxelListBase::Impl
	{
		Impl(Context& ctx, Dim3ui dims, Kind list_kind)
			: context(ctx), kind(list_kind), words(words_of(list_kind)),
			  key_bits(std::max<uint32_t>(1, std::bit_width(static_cast<uint32_t>(dims.voxel_count() - 1)))),
			  primitives(ctx),
			  entries_points(ctx, GVL_SHADER_SPAN(list_entries_points_comp), 5, sizeof(EntriesPointsPush)),
			  entries_coords(ctx, GVL_SHADER_SPAN(list_entries_coords_comp), 5, sizeof(EntriesCoordsPush)),
			  entries_meta(ctx, GVL_SHADER_SPAN(list_entries_meta_comp), 7, sizeof(EntriesMetaPush)),
			  compact(ctx, GVL_SHADER_SPAN(list_compact_comp), 8, sizeof(CompactPush)),
			  unique_heads(ctx, GVL_SHADER_SPAN(list_unique_heads_comp), 2, sizeof(CountPush)),
			  unique_merge(ctx, GVL_SHADER_SPAN(list_unique_merge_comp), 8, sizeof(MergePush)),
			  normalize_counts(ctx, GVL_SHADER_SPAN(list_normalize_counts_comp), 1, sizeof(CountPush)),
			  membership(ctx, GVL_SHADER_SPAN(list_membership_comp), 6, sizeof(MembershipPush)),
			  collide(ctx, GVL_SHADER_SPAN(list_collide_comp), 4, sizeof(CollidePush)),
			  flags_map_occupied(ctx, GVL_SHADER_SPAN(list_flags_map_occupied_comp), 2, sizeof(CountPush)),
			  gather_map(ctx, GVL_SHADER_SPAN(list_gather_map_comp), 6, sizeof(GatherPush)),
			  flags_count_ge(ctx, GVL_SHADER_SPAN(list_flags_count_ge_comp), 2, sizeof(CountGePush)),
			  entries(ctx, 0, words),
			  counter(ctx, sizeof(uint32_t)),
			  collide_result(ctx, kCollideResultWords * sizeof(uint32_t))
		{
		}

		/// Writes the entries with flags[i] != 0 (pos = exclusive scan of the flags) to dst[pos[i] + dst_offset].
		void record_compact(CommandBatch& batch, const Entries& src, const DeviceBuffer& flags, const DeviceBuffer& pos,
			Entries& dst, uint32_t dst_offset)
		{
			batch.dispatch(compact, { &src.ids, &src.coords, &src.voxels, &flags, &pos, &dst.ids, &dst.coords, &dst.voxels },
				group_count(context, src.count), CompactPush{ src.count, words, dst_offset });
		}

		/// Replaces `entries` by those with flags[i] != 0. `kept` = number of flagged entries.
		void keep_flagged(const DeviceBuffer& flags, const DeviceBuffer& pos, uint32_t kept)
		{
			if (kept == entries.count)
				return;
			Entries filtered(context, kept, words);
			if (kept > 0)
			{
				CommandBatch batch(context);
				record_compact(batch, entries, flags, pos, filtered, 0);
				batch.submit_and_wait();
			}
			entries = std::move(filtered);
		}

		/// Returns [existing entries | new entries], the new ones written by `write_new(batch, combined, offset)`.
		Entries combine_with_existing(uint32_t new_count, const std::function<void(CommandBatch&, Entries&, uint32_t)>& write_new)
		{
			const uint64_t total = static_cast<uint64_t>(entries.count) + new_count;
			if (total > kMaxEntries)
				throw VulkanError("VoxelList: more than " + std::to_string(kMaxEntries) + " entries are not supported");
			Entries combined(context, static_cast<uint32_t>(total), words);
			CommandBatch batch(context);
			if (entries.count > 0)
			{
				batch.copy(entries.ids, combined.ids, static_cast<uint64_t>(entries.count) * sizeof(uint32_t));
				batch.copy(entries.coords, combined.coords, static_cast<uint64_t>(entries.count) * 3 * sizeof(uint32_t));
				batch.copy(entries.voxels, combined.voxels, static_cast<uint64_t>(entries.count) * words * sizeof(uint32_t));
			}
			write_new(batch, combined, entries.count);
			batch.submit_and_wait();
			return combined;
		}

		/// Sorts by id and merges entries with equal ids (OR for bit voxels, wrapping int8 sum for counting voxels).
		Entries make_unique(Entries input)
		{
			const uint32_t m = input.count;
			if (m <= 1)
				return input;

			DeviceBuffer keys = make_words(context, m);
			DeviceBuffer order = make_words(context, m);
			DeviceBuffer heads = make_words(context, m);
			DeviceBuffer pos = make_words(context, m);
			KeepAlive keep_alive;
			{
				CommandBatch batch(context);
				batch.copy(input.ids, keys, static_cast<uint64_t>(m) * sizeof(uint32_t));
				primitives.record_iota(batch, order, m);
				batch.barrier();
				primitives.record_sort_pairs(batch, keys, order, m, key_bits, keep_alive);
				batch.barrier();
				batch.dispatch(unique_heads, { &keys, &heads }, group_count(context, m), CountPush{ m });
				batch.barrier();
				primitives.record_exclusive_scan(batch, heads, pos, m, keep_alive);
				batch.submit_and_wait();
			}
			const uint32_t unique_count = total_of_scan(heads, pos, m);

			Entries result(context, unique_count, words);
			CommandBatch batch(context);
			batch.fill(result.voxels, 0);
			batch.dispatch(unique_merge, { &keys, &order, &pos, &input.coords, &input.voxels, &result.ids, &result.coords, &result.voxels },
				group_count(context, m), MergePush{ m, words, kind == Kind::Bit ? 0u : 1u });
			if (kind == Kind::Counting)
			{
				batch.barrier();
				batch.dispatch(normalize_counts, { &result.voxels }, group_count(context, unique_count), CountPush{ unique_count });
			}
			batch.submit_and_wait();
			return result;
		}

		/// Appends `n` freshly created entries: `produce` fills entries + flags (flag 0 = drop), then the kept ones are
		/// added to the list and merged. \return number of dropped entries
		uint32_t append(uint32_t n, const std::function<void(CommandBatch&, Entries&, DeviceBuffer&)>& produce)
		{
			if (n == 0)
				return 0;
			Entries fresh(context, n, words);
			DeviceBuffer flags = make_words(context, n);
			DeviceBuffer pos = make_words(context, n);
			KeepAlive keep_alive;
			{
				CommandBatch batch(context);
				produce(batch, fresh, flags);
				batch.barrier();
				primitives.record_exclusive_scan(batch, flags, pos, n, keep_alive);
				batch.submit_and_wait();
			}
			const uint32_t kept = total_of_scan(flags, pos, n);
			if (kept > 0)
			{
				Entries combined = combine_with_existing(kept, [&](CommandBatch& batch, Entries& dst, uint32_t offset)
				{
					record_compact(batch, fresh, flags, pos, dst, offset);
				});
				entries = make_unique(std::move(combined));
			}
			return n - kept;
		}

		void check_other(const VoxelListBase& owner, const VoxelListBase& other) const
		{
			if (&other.context() != &context)
				throw VulkanError("VoxelList: different contexts");
			if (!(other.reference_dimensions() == owner.reference_dimensions()))
				throw VulkanError("VoxelList: reference dimensions differ");
		}

		Context& context;
		Kind kind;
		uint32_t words;
		uint32_t key_bits;
		Primitives primitives;
		ComputePipeline entries_points;
		ComputePipeline entries_coords;
		ComputePipeline entries_meta;
		ComputePipeline compact;
		ComputePipeline unique_heads;
		ComputePipeline unique_merge;
		ComputePipeline normalize_counts;
		ComputePipeline membership;
		ComputePipeline collide;
		ComputePipeline flags_map_occupied;
		ComputePipeline gather_map;
		ComputePipeline flags_count_ge;
		Entries entries;
		DeviceBuffer counter; ///< one uint32 scratch counter
		DeviceBuffer collide_result; ///< kCollideResultWords words
	};

	VoxelListBase::VoxelListBase(Context& context, Dim3ui reference_dimensions, float voxel_side_length, Kind kind)
		: context_(&context), dimensions_(reference_dimensions), voxel_side_length_(voxel_side_length), kind_(kind)
	{
		if (reference_dimensions.x == 0 || reference_dimensions.y == 0 || reference_dimensions.z == 0)
			throw VulkanError("VoxelList: empty reference dimensions");
		if (reference_dimensions.x > kMaxAxisDimension || reference_dimensions.y > kMaxAxisDimension
			|| reference_dimensions.z > kMaxAxisDimension)
			throw VulkanError("VoxelList: more than 2^24 voxels along one axis");
		if (reference_dimensions.voxel_count() > UINT32_MAX)
			throw VulkanError("VoxelList: voxel ids do not fit into 32 bits");
		if (!(voxel_side_length > 0.0f))
			throw VulkanError("VoxelList: voxel side length must be positive");
		impl_ = std::make_unique<Impl>(context, reference_dimensions, kind);
	}

	VoxelListBase::~VoxelListBase() = default;
	VoxelListBase::VoxelListBase(VoxelListBase&&) noexcept = default;
	VoxelListBase& VoxelListBase::operator=(VoxelListBase&&) noexcept = default;

	uint32_t VoxelListBase::size() const { return impl_->entries.count; }

	uint64_t VoxelListBase::memory_usage_bytes() const
	{
		return static_cast<uint64_t>(size()) * (sizeof(uint32_t) + 3 * sizeof(uint32_t) + impl_->words * sizeof(uint32_t));
	}

	void VoxelListBase::clear()
	{
		impl_->entries = Entries(*context_, 0, impl_->words);
	}

	void VoxelListBase::make_unique()
	{
		impl_->entries = impl_->make_unique(std::move(impl_->entries));
	}

	uint32_t VoxelListBase::insert_point_cloud(std::span<const Point3f> points, uint8_t meaning)
	{
		static_assert(sizeof(Point3f) == 12);
		uint32_t dropped = 0;
		for (std::size_t first = 0; first < points.size(); first += kMaxItemsPerUpload)
		{
			const auto chunk = points.subspan(first, std::min(kMaxItemsPerUpload, points.size() - first));
			const uint32_t n = static_cast<uint32_t>(chunk.size());
			DeviceBuffer point_buffer(*context_, chunk.size_bytes());
			point_buffer.upload(chunk.data(), chunk.size_bytes());
			const EntriesPointsPush push{ dimensions_.x, dimensions_.y, dimensions_.z, n, meaning,
				static_cast<uint32_t>(kind_), voxel_side_length_ };
			dropped += impl_->append(n, [&](CommandBatch& batch, Entries& fresh, DeviceBuffer& flags)
			{
				batch.dispatch(impl_->entries_points, { &point_buffer, &fresh.ids, &fresh.coords, &fresh.voxels, &flags },
					group_count(*context_, n), push);
			});
		}
		if (dropped > 0)
			GVL_LOG_WARNING(VoxelList, dropped << " point(s) lie outside the reference map dimensions!");
		return dropped;
	}

	uint32_t VoxelListBase::insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning)
	{
		static_assert(sizeof(Coord3ui) == 12);
		uint32_t dropped = 0;
		for (std::size_t first = 0; first < coordinates.size(); first += kMaxItemsPerUpload)
		{
			const auto chunk = coordinates.subspan(first, std::min(kMaxItemsPerUpload, coordinates.size() - first));
			const uint32_t n = static_cast<uint32_t>(chunk.size());
			DeviceBuffer coord_buffer(*context_, chunk.size_bytes());
			coord_buffer.upload(chunk.data(), chunk.size_bytes());
			const EntriesCoordsPush push{ dimensions_.x, dimensions_.y, dimensions_.z, n, meaning, static_cast<uint32_t>(kind_) };
			dropped += impl_->append(n, [&](CommandBatch& batch, Entries& fresh, DeviceBuffer& flags)
			{
				batch.dispatch(impl_->entries_coords, { &coord_buffer, &fresh.ids, &fresh.coords, &fresh.voxels, &flags },
					group_count(*context_, n), push);
			});
		}
		if (dropped > 0)
			GVL_LOG_WARNING(VoxelList, dropped << " coordinate(s) lie outside the reference map dimensions!");
		return dropped;
	}

	uint32_t VoxelListBase::insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning)
	{
		const std::vector<uint8_t> meanings(cloud.num_clouds(), meaning);
		return insert_meta_point_cloud(cloud, meanings);
	}

	uint32_t VoxelListBase::insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings)
	{
		if (meanings.size() != cloud.num_clouds())
			throw VulkanError("VoxelList::insert_meta_point_cloud: expected one meaning per sub cloud");
		if (&cloud.context() != context_)
			throw VulkanError("VoxelList::insert_meta_point_cloud: different contexts");
		if (cloud.empty())
			return 0;

		const std::vector<uint32_t> meaning_words(meanings.begin(), meanings.end());
		DeviceBuffer offsets(*context_, cloud.offsets().size() * sizeof(uint32_t));
		offsets.upload(cloud.offsets().data(), cloud.offsets().size() * sizeof(uint32_t));
		DeviceBuffer meaning_buffer(*context_, meaning_words.size() * sizeof(uint32_t));
		meaning_buffer.upload(meaning_words.data(), meaning_words.size() * sizeof(uint32_t));

		const uint32_t n = cloud.accumulated_size();
		const EntriesMetaPush push{ dimensions_.x, dimensions_.y, dimensions_.z, n, cloud.num_clouds(),
			static_cast<uint32_t>(kind_), voxel_side_length_ };
		const uint32_t dropped = impl_->append(n, [&](CommandBatch& batch, Entries& fresh, DeviceBuffer& flags)
		{
			batch.dispatch(impl_->entries_meta,
				{ &cloud.points(), &offsets, &meaning_buffer, &fresh.ids, &fresh.coords, &fresh.voxels, &flags },
				group_count(*context_, n), push);
		});
		if (dropped > 0)
			GVL_LOG_WARNING(VoxelList, dropped << " point(s) lie outside the reference map dimensions!");
		return dropped;
	}

	std::vector<uint32_t> VoxelListBase::download_ids() const
	{
		const uint32_t n = size();
		return n == 0 ? std::vector<uint32_t>{} : impl_->entries.ids.download_vector<uint32_t>(n);
	}

	std::vector<Coord3ui> VoxelListBase::download_coordinates() const
	{
		const uint32_t n = size();
		return n == 0 ? std::vector<Coord3ui>{} : impl_->entries.coords.download_vector<Coord3ui>(n);
	}

	std::vector<uint32_t> VoxelListBase::download_voxel_words() const
	{
		const uint32_t n = size();
		return n == 0 ? std::vector<uint32_t>{} : impl_->entries.voxels.download_vector<uint32_t>(static_cast<std::size_t>(n) * impl_->words);
	}

	const DeviceBuffer& VoxelListBase::ids_buffer() const { return impl_->entries.ids; }
	const DeviceBuffer& VoxelListBase::coordinates_buffer() const { return impl_->entries.coords; }
	const DeviceBuffer& VoxelListBase::voxel_buffer() const { return impl_->entries.voxels; }

	uint32_t VoxelListBase::count_common_entries(const VoxelListBase& other, const int8_t* threshold) const
	{
		impl_->check_other(*this, other);
		const Entries& mine = impl_->entries;
		const Entries& theirs = other.impl_->entries;
		if (mine.count == 0 || theirs.count == 0)
			return 0;

		DeviceBuffer found = make_words(*context_, mine.count);
		DeviceBuffer keep = make_words(*context_, mine.count);
		CommandBatch batch(*context_);
		batch.fill(impl_->counter, 0);
		batch.dispatch(impl_->membership, { &mine.ids, &theirs.ids, &mine.voxels, &found, &keep, &impl_->counter },
			group_count(*context_, mine.count),
			MembershipPush{ mine.count, theirs.count, threshold ? 1u : 0u, threshold ? static_cast<int32_t>(*threshold) : 0 });
		batch.submit_and_wait();
		return impl_->counter.download_vector<uint32_t>(1).front();
	}

	void VoxelListBase::subtract_entries_of(const VoxelListBase& other)
	{
		impl_->check_other(*this, other);
		const Entries& mine = impl_->entries;
		const Entries& theirs = other.impl_->entries;
		if (mine.count == 0 || theirs.count == 0)
			return;

		DeviceBuffer found = make_words(*context_, mine.count);
		DeviceBuffer keep = make_words(*context_, mine.count);
		DeviceBuffer pos = make_words(*context_, mine.count);
		KeepAlive keep_alive;
		{
			CommandBatch batch(*context_);
			batch.fill(impl_->counter, 0);
			batch.dispatch(impl_->membership, { &mine.ids, &theirs.ids, &mine.voxels, &found, &keep, &impl_->counter },
				group_count(*context_, mine.count), MembershipPush{ mine.count, theirs.count, 0, 0 });
			batch.barrier();
			impl_->primitives.record_exclusive_scan(batch, keep, pos, mine.count, keep_alive);
			batch.submit_and_wait();
		}
		impl_->keep_flagged(keep, pos, total_of_scan(keep, pos, mine.count));
	}

	void VoxelListBase::remove_counts_below(int8_t threshold)
	{
		if (kind_ != Kind::Counting)
			throw VulkanError("VoxelList: only counting lists have counts");
		const uint32_t n = size();
		if (n == 0)
			return;
		DeviceBuffer flags = make_words(*context_, n);
		DeviceBuffer pos = make_words(*context_, n);
		KeepAlive keep_alive;
		{
			CommandBatch batch(*context_);
			batch.dispatch(impl_->flags_count_ge, { &impl_->entries.voxels, &flags }, group_count(*context_, n),
				CountGePush{ n, static_cast<int32_t>(threshold) });
			batch.barrier();
			impl_->primitives.record_exclusive_scan(batch, flags, pos, n, keep_alive);
			batch.submit_and_wait();
		}
		impl_->keep_flagged(flags, pos, total_of_scan(flags, pos, n));
	}

	VoxelListBase::MapCollision VoxelListBase::collide_with_map(const DeviceBuffer& map_voxels, Dim3ui map_dimensions, bool probabilistic,
		MapCollisionMode mode, Offset3i offset, int32_t level, float threshold, const BitVector256* mask)
	{
		if (kind_ != Kind::Bit)
			throw VulkanError("VoxelList: only bit lists collide with maps");
		if (!(map_dimensions == dimensions_))
			throw VulkanError("VoxelList: the reference dimensions do not match the map dimensions");
		const Entries& mine = impl_->entries;
		MapCollision result;
		// An offset of a whole axis or more moves every entry out of the map. Smaller offsets cannot overflow in the shader.
		const auto beyond = [](int32_t shift, uint32_t extent) { return shift >= 0 ? static_cast<uint32_t>(shift) >= extent : -static_cast<int64_t>(shift) >= static_cast<int64_t>(extent); };
		if (mine.count == 0 || beyond(offset.x, dimensions_.x) || beyond(offset.y, dimensions_.y) || beyond(offset.z, dimensions_.z))
			return result;

		CollidePush push{ mine.count, dimensions_.x, dimensions_.y, dimensions_.z, offset.x, offset.y, offset.z,
			static_cast<uint32_t>(mode), probabilistic ? 1u : 0u, level, threshold, {} };
		if (mask != nullptr)
			std::copy(mask->begin(), mask->end(), push.mask);
		CommandBatch batch(*context_);
		batch.fill(impl_->collide_result, 0);
		batch.dispatch(impl_->collide, { &mine.coords, &mine.voxels, &map_voxels, &impl_->collide_result }, group_count(*context_, mine.count), push);
		batch.submit_and_wait();
		const std::vector<uint32_t> words = impl_->collide_result.download_vector<uint32_t>(kCollideResultWords);
		result.count = words[0];
		std::copy(words.begin() + 1, words.end(), result.types.begin());
		return result;
	}

	uint32_t VoxelListBase::append_occupied_voxels(const BitVoxelMap& map)
	{
		if (kind_ != Kind::Bit)
			throw VulkanError("VoxelList: only bit lists can take voxels from bit maps");
		if (!(map.dimensions() == dimensions_))
			throw VulkanError("VoxelList: the reference dimensions do not match the map dimensions");
		if (map.voxel_count() > Primitives::kMaxElements)
			throw VulkanError("VoxelList: maps with more than 2^24 voxels are not supported yet");
		const uint32_t voxel_count = static_cast<uint32_t>(map.voxel_count());

		DeviceBuffer flags = make_words(*context_, voxel_count);
		DeviceBuffer pos = make_words(*context_, voxel_count);
		KeepAlive keep_alive;
		{
			CommandBatch batch(*context_);
			batch.dispatch(impl_->flags_map_occupied, { &map.voxel_buffer(), &flags }, group_count(*context_, voxel_count),
				CountPush{ voxel_count });
			batch.barrier();
			impl_->primitives.record_exclusive_scan(batch, flags, pos, voxel_count, keep_alive);
			batch.submit_and_wait();
		}
		const uint32_t found = total_of_scan(flags, pos, voxel_count);
		if (found == 0)
			return 0;
		Entries combined = impl_->combine_with_existing(found, [&](CommandBatch& batch, Entries& dst, uint32_t offset)
		{
			batch.dispatch(impl_->gather_map, { &map.voxel_buffer(), &flags, &pos, &dst.ids, &dst.coords, &dst.voxels },
				group_count(*context_, voxel_count), GatherPush{ voxel_count, dimensions_.x, dimensions_.y, dimensions_.z, offset });
		});
		impl_->entries = impl_->make_unique(std::move(combined));
		return found;
	}

	uint32_t VoxelListBase::merge_entries(const VoxelListBase& other, Offset3i offset, std::optional<uint8_t> meaning)
	{
		if (&other == this)
			throw VulkanError("VoxelList: a list cannot be merged into itself");
		impl_->check_other(*this, other);
		if (other.voxel_side_length() != voxel_side_length_)
			throw VulkanError("VoxelList: the voxel side lengths differ");
		const uint32_t n = other.size();
		if (n == 0)
			return 0;

		const std::vector<Coord3ui> coords = other.download_coordinates();
		const std::vector<uint32_t> source = other.download_voxel_words();
		const uint32_t words = impl_->words;
		const uint32_t source_words = words_of(other.kind());
		std::vector<uint32_t> ids;
		std::vector<Coord3ui> moved;
		std::vector<uint32_t> payload;
		ids.reserve(n);
		moved.reserve(n);
		payload.reserve(static_cast<std::size_t>(n) * words);
		for (uint32_t i = 0; i < n; ++i)
		{
			const int64_t x = int64_t(coords[i].x) + offset.x;
			const int64_t y = int64_t(coords[i].y) + offset.y;
			const int64_t z = int64_t(coords[i].z) + offset.z;
			if (x < 0 || y < 0 || z < 0 || x >= int64_t(dimensions_.x) || y >= int64_t(dimensions_.y) || z >= int64_t(dimensions_.z))
				continue;
			moved.push_back({ static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z) });
			ids.push_back(static_cast<uint32_t>((uint64_t(z) * dimensions_.y + uint64_t(y)) * dimensions_.x + uint64_t(x)));
			if (kind_ == Kind::Bit)
			{
				std::array<uint32_t, kBitWords> bits{};
				if (meaning)
					bits[*meaning / 32] |= 1u << (*meaning % 32);
				else if (other.kind() == Kind::Bit)
					std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(i) * source_words, kBitWords, bits.begin());
				payload.insert(payload.end(), bits.begin(), bits.end());
			}
			else
				payload.push_back(other.kind() == Kind::Counting ? source[i] : 0u);
		}
		const uint32_t kept = static_cast<uint32_t>(ids.size());
		if (kept == 0)
			return n;

		Entries fresh(*context_, kept, words);
		fresh.ids.upload(ids.data(), ids.size() * sizeof(uint32_t));
		fresh.coords.upload(moved.data(), moved.size() * sizeof(Coord3ui));
		fresh.voxels.upload(payload.data(), payload.size() * sizeof(uint32_t));
		Entries combined = impl_->combine_with_existing(kept, [&](CommandBatch& batch, Entries& dst, uint32_t first)
		{
			batch.copy(fresh.ids, dst.ids, uint64_t(kept) * sizeof(uint32_t), 0, uint64_t(first) * sizeof(uint32_t));
			batch.copy(fresh.coords, dst.coords, uint64_t(kept) * 3 * sizeof(uint32_t), 0, uint64_t(first) * 3 * sizeof(uint32_t));
			batch.copy(fresh.voxels, dst.voxels, uint64_t(kept) * words * sizeof(uint32_t), 0, uint64_t(first) * words * sizeof(uint32_t));
		});
		impl_->entries = impl_->make_unique(std::move(combined));
		if (n > kept)
			GVL_LOG_WARNING(VoxelList, (n - kept) << " entrie(s) lie outside the reference map dimensions after the offset!");
		return n - kept;
	}

	std::optional<std::string> VoxelListBase::entries_problem(Dim3ui dims, Kind kind, std::span<const uint32_t> ids,
		std::span<const Coord3ui> coordinates, std::span<const uint32_t> payload)
	{
		const std::size_t n = ids.size();
		if (coordinates.size() != n || payload.size() != n * words_of(kind))
			return "ids, coordinates and payload do not have the same number of entries";
		if (n > kMaxEntries)
			return "more than " + std::to_string(kMaxEntries) + " entries are not supported";
		for (std::size_t i = 0; i < n; ++i)
		{
			const Coord3ui c = coordinates[i];
			if (c.x >= dims.x || c.y >= dims.y || c.z >= dims.z)
				return "entry " + std::to_string(i) + " lies outside the reference map";
			if (ids[i] != (uint64_t(c.z) * dims.y + c.y) * dims.x + c.x)
				return "the id of entry " + std::to_string(i) + " is not the index of its coordinates";
			if (i > 0 && ids[i] <= ids[i - 1])
				return "the ids are not strictly ascending at entry " + std::to_string(i);
			if (kind == Kind::Counting && static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(payload[i] & 0xFFu))) != payload[i])
				return "the count of entry " + std::to_string(i) + " is not a sign extended int8";
		}
		return std::nullopt;
	}

	void VoxelListBase::assign_entries(std::span<const uint32_t> ids, std::span<const Coord3ui> coordinates, std::span<const uint32_t> payload)
	{
		if (const auto problem = entries_problem(dimensions_, kind_, ids, coordinates, payload))
			throw VulkanError("VoxelList::assign_entries: " + *problem);
		const std::size_t n = ids.size();
		const uint32_t words = impl_->words;
		Entries fresh(*context_, static_cast<uint32_t>(n), words);
		if (n > 0)
		{
			fresh.ids.upload(ids.data(), n * sizeof(uint32_t));
			fresh.coords.upload(coordinates.data(), n * sizeof(Coord3ui));
			fresh.voxels.upload(payload.data(), payload.size() * sizeof(uint32_t));
		}
		impl_->entries = std::move(fresh);
	}

	// ---- BitVoxelList ----

	BitVoxelList::BitVoxelList(Context& context, Dim3ui reference_dimensions, float voxel_side_length)
		: VoxelListBase(context, reference_dimensions, voxel_side_length, Kind::Bit)
	{
	}

	uint32_t BitVoxelList::collide_with(const BitVoxelList& other) const
	{
		return count_common_entries(other, nullptr);
	}

	namespace
	{
		template<class Map>
		void check_context(const Context* context, const Map& map)
		{
			if (&map.context() != context)
				throw VulkanError("VoxelList: the map belongs to another context");
		}
	}

	uint32_t BitVoxelList::collide_with(const BitVoxelMap& map, Offset3i offset)
	{
		check_context(&context(), map);
		return collide_with_map(map.voxel_buffer(), map.dimensions(), false, MapCollisionMode::Default, offset, 0, 0.0f, nullptr).count;
	}

	uint32_t BitVoxelList::collide_with(const ProbVoxelMap& map, float threshold, Offset3i offset)
	{
		check_context(&context(), map);
		return collide_with_map(map.voxel_buffer(), map.dimensions(), true, MapCollisionMode::Default, offset,
			ProbVoxelMap::probability_from_float(threshold), 0.0f, nullptr).count;
	}

	BitVoxelList::TypeCollision BitVoxelList::collide_with_types(const BitVoxelMap& map, float threshold, Offset3i offset)
	{
		check_context(&context(), map);
		const MapCollision r = collide_with_map(map.voxel_buffer(), map.dimensions(), false, MapCollisionMode::Types, offset, 0, threshold, nullptr);
		return { r.count, r.types };
	}

	BitVoxelList::TypeCollision BitVoxelList::collide_with_types(const ProbVoxelMap& map, float threshold, Offset3i offset)
	{
		check_context(&context(), map);
		const MapCollision r = collide_with_map(map.voxel_buffer(), map.dimensions(), true, MapCollisionMode::Types, offset, 0, threshold, nullptr);
		return { r.count, r.types };
	}

	uint32_t BitVoxelList::collide_with_type_mask(const BitVoxelMap& map, const BitVector256& types_to_check, float threshold, Offset3i offset)
	{
		check_context(&context(), map);
		return collide_with_map(map.voxel_buffer(), map.dimensions(), false, MapCollisionMode::TypeMask, offset, 0, threshold, &types_to_check).count;
	}

	uint32_t BitVoxelList::collide_with_type_mask(const ProbVoxelMap& map, const BitVector256& types_to_check, float threshold, Offset3i offset)
	{
		check_context(&context(), map);
		return collide_with_map(map.voxel_buffer(), map.dimensions(), true, MapCollisionMode::TypeMask, offset, 0, threshold, &types_to_check).count;
	}

	void BitVoxelList::subtract(const BitVoxelList& other)
	{
		subtract_entries_of(other);
	}

	uint32_t BitVoxelList::insert_occupied_voxels(const BitVoxelMap& map)
	{
		return append_occupied_voxels(map);
	}

	uint32_t BitVoxelList::merge(const BitVoxelList& other, Offset3i offset, std::optional<uint8_t> new_meaning)
	{
		return merge_entries(other, offset, new_meaning);
	}

	uint32_t BitVoxelList::merge(const CountingVoxelList& other, Offset3i offset, std::optional<uint8_t> new_meaning)
	{
		return merge_entries(other, offset, new_meaning);
	}

	uint32_t BitVoxelList::merge(const BitVoxelMap& other, std::optional<uint8_t> new_meaning)
	{
		if (!new_meaning)
		{
			append_occupied_voxels(other);
			return 0;
		}
		// The voxels of the map in a list of their own, so that only they get the new meaning.
		BitVoxelList taken(context(), reference_dimensions(), voxel_side_length());
		taken.append_occupied_voxels(other);
		return merge_entries(taken, {}, new_meaning);
	}

	std::vector<BitVector256> BitVoxelList::download_bit_vectors() const
	{
		const uint32_t n = size();
		return n == 0 ? std::vector<BitVector256>{} : voxel_buffer().download_vector<BitVector256>(n);
	}

	// ---- CountingVoxelList ----

	CountingVoxelList::CountingVoxelList(Context& context, Dim3ui reference_dimensions, float voxel_side_length)
		: VoxelListBase(context, reference_dimensions, voxel_side_length, Kind::Counting)
	{
	}

	uint32_t CountingVoxelList::collide_with(const BitVoxelList& other, int8_t threshold) const
	{
		return count_common_entries(other, &threshold);
	}

	void CountingVoxelList::remove_underpopulated(int8_t threshold)
	{
		remove_counts_below(threshold);
	}

	void CountingVoxelList::subtract(const CountingVoxelList& other)
	{
		subtract_entries_of(other);
	}

	void CountingVoxelList::subtract_from_counting_list(const BitVoxelList& other)
	{
		subtract_entries_of(other);
	}

	uint32_t CountingVoxelList::merge(const CountingVoxelList& other, Offset3i offset)
	{
		return merge_entries(other, offset, std::nullopt);
	}

	uint32_t CountingVoxelList::merge(const BitVoxelList& other, Offset3i offset)
	{
		return merge_entries(other, offset, std::nullopt);
	}

	std::vector<int8_t> CountingVoxelList::download_counts() const
	{
		const auto words = download_voxel_words();
		std::vector<int8_t> counts(words.size());
		for (std::size_t i = 0; i < words.size(); ++i)
			counts[i] = static_cast<int8_t>(words[i] & 0xFFu);
		return counts;
	}
}
