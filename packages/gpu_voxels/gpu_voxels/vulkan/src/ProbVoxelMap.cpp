// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/ProbVoxelMap.h>

#include <algorithm>
#include <bit>

#include <gpu_voxels/logging/logging_vulkan.h>
#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/Primitives.h>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kLocalSize = 64;                      // local_size_x of all ProbVoxelMap kernels
		constexpr uint32_t kMaxAxisDimension = 1u << 24;         // exactly representable in a float (bounds checks)
		constexpr std::size_t kMaxItemsPerCall = 4u << 20;       // measurements per upload buffer / scan
		constexpr uint32_t kUnknownWord = 0xFFFFFF80u;           // int8 -128, sign extended

		/// Must match the push block of shaders/prob_push.glsl.
		struct EventsPush
		{
			uint32_t dim_x, dim_y, dim_z;
			uint32_t first, count, input_kind;
			float side;
			float origin_x, origin_y, origin_z;
			uint32_t raycast, require_origin, cut_robot, robot_bit;
			int32_t free_inc, hit_inc;
			uint32_t mode, assign_unknown, base, batch_first;
		};
		struct CountPush { uint32_t n; };
		struct CollidePush { uint32_t voxel_count, other_is_bit; int32_t threshold_this, threshold_other, mark_increment; };

		uint32_t group_count(const Context& context, uint64_t items)
		{
			const uint64_t needed = (items + kLocalSize - 1) / kLocalSize;
			const uint64_t cap = std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], 65535);
			return static_cast<uint32_t>(std::clamp<uint64_t>(needed, 1, cap));
		}

		void check_model(const SensorModel& model)
		{
			if (model.free_update < ProbVoxelMap::kMinProbability || model.occupied_update < ProbVoxelMap::kMinProbability)
				throw VulkanError("ProbVoxelMap: sensor model updates must lie in [-127, 127]");
		}

		struct MeaningUpdate
		{
			bool assign_unknown;
			int32_t increment;
			bool operator==(const MeaningUpdate&) const = default;
		};

		/// CUDA ProbabilisticVoxel::insert(BitVoxelMeaning).
		MeaningUpdate update_of_meaning(uint8_t meaning, const SensorModel& model)
		{
			switch (meaning)
			{
			case eBVM_FREE: return { false, model.free_update };
			case eBVM_OCCUPIED:
			case eBVM_COLLISION: return { false, model.occupied_update };
			case eBVM_UNKNOWN:
			case eBVM_UNDEFINED: return { true, 0 };
			default: return { false, static_cast<int32_t>(meaning) - eBVM_UNCERTAIN_OCC_PROB }; // swept volume 4..254
			}
		}

		uint32_t read_word(const DeviceBuffer& buffer, uint64_t index)
		{
			uint32_t word = 0;
			buffer.download(&word, sizeof(uint32_t), index * sizeof(uint32_t));
			return word;
		}
	}

	struct ProbVoxelMap::Kernels
	{
		explicit Kernels(Context& context)
			: points_apply(context, GVL_SHADER_SPAN(prob_points_apply_comp), 6, sizeof(EventsPush)),
			  events_count(context, GVL_SHADER_SPAN(prob_events_count_comp), 4, sizeof(EventsPush)),
			  events_write(context, GVL_SHADER_SPAN(prob_events_write_comp), 5, sizeof(EventsPush)),
			  events_fold(context, GVL_SHADER_SPAN(prob_events_fold_comp), 3, sizeof(CountPush)),
			  apply_accumulated(context, GVL_SHADER_SPAN(prob_apply_accumulated_comp), 3, sizeof(CountPush)),
			  collide(context, GVL_SHADER_SPAN(prob_collide_comp), 3, sizeof(CollidePush))
		{
		}

		ComputePipeline points_apply;
		ComputePipeline events_count;
		ComputePipeline events_write;
		ComputePipeline events_fold;
		ComputePipeline apply_accumulated;
		ComputePipeline collide;
	};

	/// A range of measurements inside a device buffer.
	struct ProbVoxelMap::EventSource
	{
		const DeviceBuffer* buffer;
		uint32_t first;       ///< first measurement (a measurement is 3 words)
		uint32_t count;
		uint32_t input_kind;  ///< 0: float points, 1: uint voxel coordinates
	};

	struct ProbVoxelMap::EventParams
	{
		Point3f origin{ 0, 0, 0 };
		bool raycast = false;
		bool require_origin = false;
		const BitVoxelMap* robot = nullptr;
		uint8_t robot_bit = 0;
		int32_t free_inc = 0;
		int32_t hit_inc = 0;
		bool assign_unknown = false;
	};

	ProbVoxelMap::ProbVoxelMap(Context& context, Dim3ui dimensions, float voxel_side_length)
		: context_(&context), dimensions_(dimensions), voxel_side_length_(voxel_side_length),
		  voxels_(context, [&]
		  {
			  if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0)
				  throw VulkanError("ProbVoxelMap: empty dimensions");
			  if (dimensions.x > kMaxAxisDimension || dimensions.y > kMaxAxisDimension || dimensions.z > kMaxAxisDimension)
				  throw VulkanError("ProbVoxelMap: more than 2^24 voxels along one axis");
			  if (!(voxel_side_length > 0.0f))
				  throw VulkanError("ProbVoxelMap: voxel side length must be positive");
			  if (dimensions.voxel_count() > UINT32_MAX)
				  throw VulkanError("ProbVoxelMap: voxel index does not fit into 32 bits");
			  return dimensions.voxel_count() * kBytesPerVoxel; // DeviceBuffer rejects sizes above maxStorageBufferRange
		  }()),
		  scratch_counters_(context, 2 * sizeof(uint32_t)),
		  kernels_(std::make_shared<Kernels>(context))
	{
		clear_map();
	}

	ProbVoxelMap::~ProbVoxelMap() = default;
	ProbVoxelMap::ProbVoxelMap(ProbVoxelMap&&) noexcept = default;
	ProbVoxelMap& ProbVoxelMap::operator=(ProbVoxelMap&&) noexcept = default;

	Primitives& ProbVoxelMap::primitives()
	{
		if (!primitives_)
			primitives_ = std::make_unique<Primitives>(*context_);
		return *primitives_;
	}

	int8_t ProbVoxelMap::probability_from_float(float value)
	{
		float tmp = (std::max)((std::min)(1.0f, value), 0.0f);
		tmp = (tmp * (static_cast<float>(kMaxProbability) - static_cast<float>(kMinProbability))) + static_cast<float>(kMinProbability);
		return static_cast<int8_t>(tmp);
	}

	float ProbVoxelMap::probability_to_float(int8_t value)
	{
		return (static_cast<float>(value) - static_cast<float>(kMinProbability)) /
			(static_cast<float>(kMaxProbability) - static_cast<float>(kMinProbability));
	}

	void ProbVoxelMap::clear_map()
	{
		voxels_.fill(kUnknownWord);
	}

	void ProbVoxelMap::clear_bit_voxel_meaning(uint8_t meaning)
	{
		if (meaning != eBVM_OCCUPIED)
			throw VulkanError("ProbVoxelMap::clear_bit_voxel_meaning: only eBVM_OCCUPIED is supported (clears the map)");
		clear_map();
	}

	std::vector<int8_t> ProbVoxelMap::download_occupancy(uint64_t first_voxel, uint64_t voxel_count_in) const
	{
		if (first_voxel > voxel_count())
			throw VulkanError("ProbVoxelMap::download_occupancy: first voxel beyond the map");
		const uint64_t count = std::min(voxel_count_in, voxel_count() - first_voxel);
		if (count == 0)
			return {};
		const auto words = voxels_.download_vector<uint32_t>(count, first_voxel * kBytesPerVoxel);
		std::vector<int8_t> values(words.size());
		for (std::size_t i = 0; i < words.size(); ++i)
			values[i] = static_cast<int8_t>(words[i] & 0xFFu);
		return values;
	}

	ProbVoxelMap::SensorResult ProbVoxelMap::apply_events(const EventSource& source, const EventParams& params, const UpdateOptions& options)
	{
		SensorResult total;
		for (uint32_t done = 0; done < source.count; done += static_cast<uint32_t>(kMaxItemsPerCall))
		{
			EventSource chunk = source;
			chunk.first = source.first + done;
			chunk.count = std::min<uint32_t>(static_cast<uint32_t>(kMaxItemsPerCall), source.count - done);
			const SensorResult part = apply_event_chunk(chunk, params, options);
			total.outside += part.outside;
			total.cut_by_robot += part.cut_by_robot;
		}
		return total;
	}

	namespace
	{
		EventsPush make_push(const Dim3ui& dims, float side, bool cut, uint8_t robot_bit,
			uint32_t first, uint32_t count, uint32_t input_kind, Point3f origin, bool raycast, bool require_origin,
			int32_t free_inc, int32_t hit_inc, uint32_t mode, bool assign_unknown)
		{
			return EventsPush{ dims.x, dims.y, dims.z, first, count, input_kind, side, origin.x, origin.y, origin.z,
				raycast ? 1u : 0u, require_origin ? 1u : 0u, cut ? 1u : 0u, robot_bit, free_inc, hit_inc, mode,
				assign_unknown ? 1u : 0u, 0u, first };
		}
	}

	ProbVoxelMap::SensorResult ProbVoxelMap::apply_event_chunk(const EventSource& source, const EventParams& params, const UpdateOptions& options)
	{
		if (source.count == 0)
			return {};
		if (options.mode == UpdateMode::Sequential && !params.assign_unknown)
			return apply_sequential(source, params, options.max_sequential_events);

		const bool accumulate = options.mode == UpdateMode::Accumulate && !params.assign_unknown;
		if (accumulate && !accumulator_)
		{
			accumulator_ = std::make_unique<DeviceBuffer>(*context_, voxel_count() * sizeof(uint32_t));
			touched_ = std::make_unique<DeviceBuffer>(*context_, ((voxel_count() + 31) / 32) * sizeof(uint32_t));
			accumulator_->fill(0);
			touched_->fill(0);
		}
		// Atomic mode never touches the accumulators; the counter buffer is a valid stand-in for the binding.
		const DeviceBuffer& acc = accumulate ? *accumulator_ : scratch_counters_;
		const DeviceBuffer& touched = accumulate ? *touched_ : scratch_counters_;
		const DeviceBuffer& robot = params.robot ? params.robot->voxel_buffer() : voxels_;

		const EventsPush push = make_push(dimensions_, voxel_side_length_, params.robot != nullptr, params.robot_bit,
			source.first, source.count, source.input_kind, params.origin, params.raycast, params.require_origin,
			params.free_inc, params.hit_inc, accumulate ? 1u : 0u, params.assign_unknown);
		CommandBatch batch(*context_);
		batch.fill(scratch_counters_, 0);
		batch.dispatch(kernels_->points_apply, { source.buffer, &robot, &voxels_, &acc, &touched, &scratch_counters_ },
			group_count(*context_, source.count), push);
		if (accumulate)
		{
			batch.barrier();
			batch.dispatch(kernels_->apply_accumulated, { &voxels_, accumulator_.get(), touched_.get() },
				group_count(*context_, voxel_count()), CountPush{ static_cast<uint32_t>(voxel_count()) });
		}
		batch.submit_and_wait();
		const auto counters = scratch_counters_.download_vector<uint32_t>(2);
		return SensorResult{ counters[0], counters[1] };
	}

	ProbVoxelMap::SensorResult ProbVoxelMap::apply_sequential(const EventSource& source, const EventParams& params, uint32_t max_events)
	{
		if (max_events == 0 || max_events > kMaxSequentialEvents)
			throw VulkanError("ProbVoxelMap: max_sequential_events must be in 1 .. kMaxSequentialEvents");
		Primitives& prim = primitives();
		const uint32_t n = source.count;
		const DeviceBuffer& robot = params.robot ? params.robot->voxel_buffer() : voxels_;
		const EventsPush push = make_push(dimensions_, voxel_side_length_, params.robot != nullptr, params.robot_bit,
			source.first, n, source.input_kind, params.origin, params.raycast, params.require_origin,
			params.free_inc, params.hit_inc, 0, false);

		// 1. number of updates per measurement and their exclusive scan
		DeviceBuffer counts(*context_, static_cast<uint64_t>(n) * sizeof(uint32_t));
		DeviceBuffer offsets(*context_, static_cast<uint64_t>(n) * sizeof(uint32_t));
		KeepAlive keep_alive;
		{
			CommandBatch batch(*context_);
			batch.fill(scratch_counters_, 0);
			batch.dispatch(kernels_->events_count, { source.buffer, &robot, &counts, &scratch_counters_ },
				group_count(*context_, n), push);
			batch.barrier();
			prim.record_exclusive_scan(batch, counts, offsets, n, keep_alive);
			batch.submit_and_wait();
		}
		const auto skipped = scratch_counters_.download_vector<uint32_t>(2);
		const SensorResult result{ skipped[0], skipped[1] };
		const uint64_t total = static_cast<uint64_t>(read_word(offsets, n - 1)) + read_word(counts, n - 1);
		if (total == 0)
			return result;

		// 2. chunks of consecutive measurements with at most kMaxSequentialEvents updates (applied in order)
		std::vector<uint32_t> offset_values; // only needed on the host when several chunks are required
		auto offset_of = [&](uint32_t i) -> uint64_t
		{
			if (i == n)
				return total;
			return offset_values.empty() ? 0 : offset_values[i];
		};
		std::vector<uint32_t> bounds{ 0, n };
		if (total > max_events)
		{
			offset_values = offsets.download_vector<uint32_t>(n);
			bounds = { 0 };
			while (bounds.back() < n)
			{
				const uint32_t a = bounds.back();
				if (offset_of(a + 1) - offset_of(a) > max_events)
					throw VulkanError("ProbVoxelMap: one measurement needs more voxel updates than a Sequential chunk can hold");
				uint32_t lo = a + 1, hi = n; // largest b with offset_of(b) - offset_of(a) <= cap
				while (lo < hi)
				{
					const uint32_t mid = lo + (hi - lo + 1) / 2;
					if (offset_of(mid) - offset_of(a) <= max_events)
						lo = mid;
					else
						hi = mid - 1;
				}
				bounds.push_back(lo);
			}
		}

		const uint32_t key_bits = std::max<uint32_t>(1, std::bit_width(static_cast<uint32_t>(voxel_count() - 1)));
		for (std::size_t c = 0; c + 1 < bounds.size(); ++c)
		{
			const uint32_t a = bounds[c], b = bounds[c + 1];
			const uint64_t base = offset_of(a);
			const uint64_t events = offset_of(b) - base;
			if (events == 0)
				continue;
			DeviceBuffer keys(*context_, events * sizeof(uint32_t));
			DeviceBuffer vals(*context_, events * sizeof(uint32_t));
			EventsPush write_push = push;
			write_push.first = source.first + a;
			write_push.count = b - a;
			write_push.base = static_cast<uint32_t>(base);
			write_push.batch_first = source.first;
			KeepAlive sort_keep_alive;
			CommandBatch batch(*context_);
			batch.dispatch(kernels_->events_write, { source.buffer, &robot, &offsets, &keys, &vals },
				group_count(*context_, b - a), write_push);
			batch.barrier();
			prim.record_sort_pairs(batch, keys, vals, static_cast<uint32_t>(events), key_bits, sort_keep_alive);
			batch.barrier();
			batch.dispatch(kernels_->events_fold, { &keys, &vals, &voxels_ }, group_count(*context_, events),
				CountPush{ static_cast<uint32_t>(events) });
			batch.submit_and_wait();
		}
		return result;
	}

	uint32_t ProbVoxelMap::insert_point_cloud(std::span<const Point3f> points, uint8_t meaning, const UpdateOptions& options)
	{
		static_assert(sizeof(Point3f) == 12);
		check_model(options.model);
		const MeaningUpdate update = update_of_meaning(meaning, options.model);
		EventParams params;
		params.hit_inc = update.increment;
		params.assign_unknown = update.assign_unknown;
		uint32_t outside = 0;
		for (std::size_t first = 0; first < points.size(); first += kMaxItemsPerCall)
		{
			const auto chunk = points.subspan(first, std::min(kMaxItemsPerCall, points.size() - first));
			DeviceBuffer buffer(*context_, chunk.size_bytes());
			buffer.upload(chunk.data(), chunk.size_bytes());
			outside += apply_events({ &buffer, 0, static_cast<uint32_t>(chunk.size()), 0 }, params, options).outside;
		}
		if (outside > 0)
			LOGGING_WARNING_C(VulkanLog, ProbVoxelMap, outside << " point(s) lie outside the map dimensions!" << endl);
		return outside;
	}

	uint32_t ProbVoxelMap::insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning, const UpdateOptions& options)
	{
		static_assert(sizeof(Coord3ui) == 12);
		check_model(options.model);
		const MeaningUpdate update = update_of_meaning(meaning, options.model);
		EventParams params;
		params.hit_inc = update.increment;
		params.assign_unknown = update.assign_unknown;
		uint32_t outside = 0;
		for (std::size_t first = 0; first < coordinates.size(); first += kMaxItemsPerCall)
		{
			const auto chunk = coordinates.subspan(first, std::min(kMaxItemsPerCall, coordinates.size() - first));
			DeviceBuffer buffer(*context_, chunk.size_bytes());
			buffer.upload(chunk.data(), chunk.size_bytes());
			outside += apply_events({ &buffer, 0, static_cast<uint32_t>(chunk.size()), 1 }, params, options).outside;
		}
		if (outside > 0)
			LOGGING_WARNING_C(VulkanLog, ProbVoxelMap, outside << " coordinate(s) lie outside the map dimensions!" << endl);
		return outside;
	}

	uint32_t ProbVoxelMap::insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning, const UpdateOptions& options)
	{
		const std::vector<uint8_t> meanings(cloud.num_clouds(), meaning);
		return insert_meta_point_cloud(cloud, meanings, options);
	}

	uint32_t ProbVoxelMap::insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings, const UpdateOptions& options)
	{
		if (meanings.size() != cloud.num_clouds())
			throw VulkanError("ProbVoxelMap::insert_meta_point_cloud: expected one meaning per sub cloud");
		if (&cloud.context() != context_)
			throw VulkanError("ProbVoxelMap::insert_meta_point_cloud: different contexts");
		check_model(options.model);

		// Sub clouds are contiguous in the packed buffer: neighbours with the same update form one batch.
		uint32_t outside = 0;
		uint32_t k = 0;
		while (k < cloud.num_clouds())
		{
			const MeaningUpdate update = update_of_meaning(meanings[k], options.model);
			uint32_t end = k + 1;
			while (end < cloud.num_clouds() && update_of_meaning(meanings[end], options.model) == update)
				++end;
			EventParams params;
			params.hit_inc = update.increment;
			params.assign_unknown = update.assign_unknown;
			const uint32_t first = cloud.cloud_offset(k);
			const uint32_t count = cloud.cloud_offset(end) - first;
			if (count > 0)
				outside += apply_events({ &cloud.points(), first, count, 0 }, params, options).outside;
			k = end;
		}
		if (outside > 0)
			LOGGING_WARNING_C(VulkanLog, ProbVoxelMap, outside << " point(s) lie outside the map dimensions!" << endl);
		return outside;
	}

	ProbVoxelMap::SensorResult ProbVoxelMap::insert_sensor_data(std::span<const Point3f> points, Point3f sensor_pose, bool raycasting,
		const BitVoxelMap* robot_map, uint8_t robot_voxel_meaning, const UpdateOptions& options)
	{
		check_model(options.model);
		if (robot_map)
		{
			if (&robot_map->context() != context_)
				throw VulkanError("ProbVoxelMap::insert_sensor_data: different contexts");
			if (!(robot_map->dimensions() == dimensions_))
				throw VulkanError("ProbVoxelMap::insert_sensor_data: the robot map must have the same dimensions");
		}
		EventParams params;
		params.origin = sensor_pose;
		params.raycast = raycasting;
		params.require_origin = true;
		params.robot = robot_map;
		params.robot_bit = robot_voxel_meaning;
		params.free_inc = options.model.free_update;
		params.hit_inc = options.model.occupied_update;
		SensorResult total;
		for (std::size_t first = 0; first < points.size(); first += kMaxItemsPerCall)
		{
			const auto chunk = points.subspan(first, std::min(kMaxItemsPerCall, points.size() - first));
			DeviceBuffer buffer(*context_, chunk.size_bytes());
			buffer.upload(chunk.data(), chunk.size_bytes());
			const SensorResult part = apply_events({ &buffer, 0, static_cast<uint32_t>(chunk.size()), 0 }, params, options);
			total.outside += part.outside;
			total.cut_by_robot += part.cut_by_robot;
		}
		if (total.outside > 0)
			LOGGING_WARNING_C(VulkanLog, ProbVoxelMap, total.outside << " measurement(s) were dropped: hit or sensor outside the map dimensions!" << endl);
		return total;
	}

	uint32_t ProbVoxelMap::collide(const DeviceBuffer& other, bool other_is_bit, float threshold, const SensorModel& model)
	{
		check_model(model);
		const int32_t level = probability_from_float(threshold);
		const CollidePush push{ static_cast<uint32_t>(voxel_count()), other_is_bit ? 1u : 0u, level, level, model.occupied_update };
		CommandBatch batch(*context_);
		batch.fill(scratch_counters_, 0);
		batch.dispatch(kernels_->collide, { &voxels_, &other, &scratch_counters_ }, group_count(*context_, voxel_count()), push);
		batch.submit_and_wait();
		return read_word(scratch_counters_, 0);
	}

	uint32_t ProbVoxelMap::collide_with(const BitVoxelMap& other, float threshold, const SensorModel& model)
	{
		if (&other.context() != context_)
			throw VulkanError("ProbVoxelMap::collide_with: different contexts");
		if (!(other.dimensions() == dimensions_))
			throw VulkanError("ProbVoxelMap::collide_with: dimensions differ");
		return collide(other.voxel_buffer(), true, threshold, model);
	}

	uint32_t ProbVoxelMap::collide_with(const ProbVoxelMap& other, float threshold, const SensorModel& model)
	{
		if (&other == this)
			throw VulkanError("ProbVoxelMap::collide_with: a map cannot collide with itself");
		if (other.context_ != context_)
			throw VulkanError("ProbVoxelMap::collide_with: different contexts");
		if (!(other.dimensions_ == dimensions_))
			throw VulkanError("ProbVoxelMap::collide_with: dimensions differ");
		return collide(other.voxels_, false, threshold, model);
	}
}
