// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/DistanceVoxelMap.h>

#include <algorithm>
#include <cmath>

#include <gpu_voxels/vulkan/Log.h>
#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kLocalSize = 64;                      // local_size_x of all DistanceVoxelMap kernels
		constexpr std::size_t kMaxItemsPerUpload = 4u << 20;     // points / coordinates per upload buffer

		struct InsertPush { uint32_t dim_x, dim_y, dim_z, count, input_kind; float voxel_side_length; };
		struct MergePush { uint32_t dim_x, dim_y, dim_z, voxel_count; int32_t level; };
		struct ZPush { uint32_t dim_x, dim_y, dim_z, lines; };
		struct LinePush { uint32_t dim_x, dim_y, dim_z, lines, axis; };
		struct ExtractPush { uint32_t dim_x, dim_y, dim_z, voxel_count; int32_t radius; };
		struct QueryPush { uint32_t dim_x, dim_y, dim_z, count; };
		struct PbaBandPush { uint32_t nx, ny, nz, band_size, count; };
		struct PbaStackPush { uint32_t nx, ny, nz, band_size, count, bands; };
		struct PbaDistancePush { uint32_t nx, ny, nz, band_count; };
		struct PbaVolumePush { uint32_t nx, ny, nz, count; };

		uint32_t group_count(const Context& context, uint64_t items)
		{
			const uint64_t needed = (items + kLocalSize - 1) / kLocalSize;
			const uint64_t cap = std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], 65535);
			return static_cast<uint32_t>(std::clamp<uint64_t>(needed, 1, cap));
		}

		/// CUDA DistanceVoxel::squaredObstacleDistance
		int32_t squared_distance_of(uint32_t word, Coord3ui position)
		{
			const Coord3ui o = DistanceVoxelMap::unpack(word);
			constexpr uint32_t u = DistanceVoxelMap::kUninitialisedCoordinate;
			if (position.x == u || position.y == u || position.z == u || o.x == u || o.y == u || o.z == u)
				return DistanceVoxelMap::kMaxObstacleDistance;
			const int64_t dx = static_cast<int64_t>(position.x) - o.x;
			const int64_t dy = static_cast<int64_t>(position.y) - o.y;
			const int64_t dz = static_cast<int64_t>(position.z) - o.z;
			return static_cast<int32_t>(dx * dx + dy * dy + dz * dz);
		}

		int32_t truncated_root(int32_t squared)
		{
			return static_cast<int32_t>(std::sqrt(static_cast<double>(squared)));
		}
	}

	struct DistanceVoxelMap::Kernels
	{
		explicit Kernels(Context& context)
			: insert(context, GVL_SHADER_SPAN(dist_insert_comp), 3, sizeof(InsertPush)),
			  merge_occupied(context, GVL_SHADER_SPAN(dist_merge_occupied_comp), 2, sizeof(MergePush)),
			  edt_z(context, GVL_SHADER_SPAN(dist_edt_z_comp), 3, sizeof(ZPush)),
			  edt_line(context, GVL_SHADER_SPAN(dist_edt_line_comp), 4, sizeof(LinePush)),
			  extract(context, GVL_SHADER_SPAN(dist_extract_comp), 2, sizeof(ExtractPush)),
			  query(context, GVL_SHADER_SPAN(dist_query_comp), 3, sizeof(QueryPush))
		{
		}

		ComputePipeline insert;
		ComputePipeline merge_occupied;
		ComputePipeline edt_z;
		ComputePipeline edt_line;
		ComputePipeline extract;
		ComputePipeline query;
	};

	struct DistanceVoxelMap::PbaKernels
	{
		explicit PbaKernels(Context& context)
			: sanitize(context, GVL_SHADER_SPAN(pba_sanitize_comp), 1, sizeof(PbaVolumePush)),
			  flood(context, GVL_SHADER_SPAN(pba_p1_flood_comp), 1, sizeof(PbaBandPush)),
			  propagate(context, GVL_SHADER_SPAN(pba_p1_propagate_comp), 2, sizeof(PbaBandPush)),
			  update(context, GVL_SHADER_SPAN(pba_p1_update_comp), 2, sizeof(PbaBandPush)),
			  backpointers(context, GVL_SHADER_SPAN(pba_p2_backpointers_comp), 2, sizeof(PbaStackPush)),
			  forward(context, GVL_SHADER_SPAN(pba_p2_forward_comp), 2, sizeof(PbaStackPush)),
			  merge(context, GVL_SHADER_SPAN(pba_p2_merge_comp), 2, sizeof(PbaStackPush)),
			  distances(context, GVL_SHADER_SPAN(pba_p3_distances_comp), 2, sizeof(PbaDistancePush)),
			  transpose(context, GVL_SHADER_SPAN(pba_transpose_comp), 2, sizeof(PbaVolumePush))
		{
		}

		ComputePipeline sanitize;
		ComputePipeline flood;
		ComputePipeline propagate;
		ComputePipeline update;
		ComputePipeline backpointers;
		ComputePipeline forward;
		ComputePipeline merge;
		ComputePipeline distances;
		ComputePipeline transpose;
	};

	DistanceVoxelMap::DistanceVoxelMap(Context& context, Dim3ui dimensions, float voxel_side_length)
		: context_(&context), dimensions_(dimensions), voxel_side_length_(voxel_side_length),
		  voxels_(context, [&]
		  {
			  if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0)
				  throw VulkanError("DistanceVoxelMap: empty dimensions");
			  if (dimensions.x > kMaxAxisDimension || dimensions.y > kMaxAxisDimension || dimensions.z > kMaxAxisDimension)
				  throw VulkanError("DistanceVoxelMap: more than 1023 voxels along one axis (10 bit obstacle coordinates)");
			  if (!(voxel_side_length > 0.0f))
				  throw VulkanError("DistanceVoxelMap: voxel side length must be positive");
			  if (dimensions.voxel_count() > UINT32_MAX)
				  throw VulkanError("DistanceVoxelMap: voxel index does not fit into 32 bits");
			  return dimensions.voxel_count() * kBytesPerVoxel; // DeviceBuffer rejects sizes above maxStorageBufferRange
		  }()),
		  scratch_counter_(context, sizeof(uint32_t)),
		  kernels_(std::make_shared<Kernels>(context))
	{
		clear_map();
	}

	DistanceVoxelMap::~DistanceVoxelMap() = default;
	DistanceVoxelMap::DistanceVoxelMap(DistanceVoxelMap&&) noexcept = default;
	DistanceVoxelMap& DistanceVoxelMap::operator=(DistanceVoxelMap&&) noexcept = default;

	void DistanceVoxelMap::clear_map()
	{
		voxels_.fill(uninitialised_word());
	}

	uint32_t DistanceVoxelMap::insert_raw(const DeviceBuffer& buffer, uint32_t count, uint32_t input_kind)
	{
		const InsertPush push{ dimensions_.x, dimensions_.y, dimensions_.z, count, input_kind, voxel_side_length_ };
		CommandBatch batch(*context_);
		batch.fill(scratch_counter_, 0);
		batch.dispatch(kernels_->insert, { &buffer, &voxels_, &scratch_counter_ }, group_count(*context_, count), push);
		batch.submit_and_wait();
		uint32_t outside = 0;
		scratch_counter_.download(&outside, sizeof(uint32_t));
		return outside;
	}

	uint32_t DistanceVoxelMap::insert_point_cloud(std::span<const Point3f> points, uint8_t meaning)
	{
		static_assert(sizeof(Point3f) == 12);
		if (meaning != eBVM_OCCUPIED)
			throw VulkanError("DistanceVoxelMap: only eBVM_OCCUPIED can be inserted");
		uint32_t outside = 0;
		for (std::size_t first = 0; first < points.size(); first += kMaxItemsPerUpload)
		{
			const auto chunk = points.subspan(first, std::min(kMaxItemsPerUpload, points.size() - first));
			DeviceBuffer buffer(*context_, chunk.size_bytes());
			buffer.upload(chunk.data(), chunk.size_bytes());
			outside += insert_raw(buffer, static_cast<uint32_t>(chunk.size()), 0);
		}
		if (outside > 0)
			GVL_LOG_WARNING(DistanceVoxelMap, outside << " point(s) lie outside the map dimensions!");
		return outside;
	}

	uint32_t DistanceVoxelMap::insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning)
	{
		static_assert(sizeof(Coord3ui) == 12);
		if (meaning != eBVM_OCCUPIED)
			throw VulkanError("DistanceVoxelMap: only eBVM_OCCUPIED can be inserted");
		uint32_t outside = 0;
		for (std::size_t first = 0; first < coordinates.size(); first += kMaxItemsPerUpload)
		{
			const auto chunk = coordinates.subspan(first, std::min(kMaxItemsPerUpload, coordinates.size() - first));
			DeviceBuffer buffer(*context_, chunk.size_bytes());
			buffer.upload(chunk.data(), chunk.size_bytes());
			outside += insert_raw(buffer, static_cast<uint32_t>(chunk.size()), 1);
		}
		if (outside > 0)
			GVL_LOG_WARNING(DistanceVoxelMap, outside << " coordinate(s) lie outside the map dimensions!");
		return outside;
	}

	uint32_t DistanceVoxelMap::insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning)
	{
		if (meaning != eBVM_OCCUPIED)
			throw VulkanError("DistanceVoxelMap: only eBVM_OCCUPIED can be inserted");
		if (&cloud.context() != context_)
			throw VulkanError("DistanceVoxelMap::insert_meta_point_cloud: different contexts");
		if (cloud.empty())
			return 0;
		const uint32_t outside = insert_raw(cloud.points(), cloud.accumulated_size(), 0);
		if (outside > 0)
			GVL_LOG_WARNING(DistanceVoxelMap, outside << " point(s) lie outside the map dimensions!");
		return outside;
	}

	void DistanceVoxelMap::merge_occupied(const ProbVoxelMap& other, float occupancy_threshold)
	{
		if (&other.context() != context_)
			throw VulkanError("DistanceVoxelMap::merge_occupied: different contexts");
		if (!(other.dimensions() == dimensions_))
			throw VulkanError("DistanceVoxelMap::merge_occupied: dimensions differ");
		const MergePush push{ dimensions_.x, dimensions_.y, dimensions_.z, static_cast<uint32_t>(voxel_count()),
			ProbVoxelMap::probability_from_float(occupancy_threshold) };
		CommandBatch batch(*context_);
		batch.dispatch(kernels_->merge_occupied, { &other.voxel_buffer(), &voxels_ }, group_count(*context_, voxel_count()), push);
		batch.submit_and_wait();
	}

	void DistanceVoxelMap::compute_distances(DistanceAlgorithm algorithm, const ParallelBandingOptions& pba)
	{
		if (algorithm == DistanceAlgorithm::ParallelBanding)
			compute_parallel_banding(pba);
		else
			compute_exact();
	}

	void DistanceVoxelMap::compute_exact()
	{
		const uint64_t bytes = voxel_count() * sizeof(uint32_t);
		DeviceBuffer cost_a(*context_, bytes), site_a(*context_, bytes), cost_b(*context_, bytes), site_b(*context_, bytes);
		CommandBatch batch(*context_);
		// pass 1 along z (map -> a), pass 2 along y (a -> b), pass 3 along x (b -> map)
		batch.dispatch(kernels_->edt_z, { &voxels_, &cost_a, &site_a }, group_count(*context_, static_cast<uint64_t>(dimensions_.x) * dimensions_.y),
			ZPush{ dimensions_.x, dimensions_.y, dimensions_.z, dimensions_.x * dimensions_.y });
		batch.barrier();
		const uint32_t y_lines = dimensions_.x * dimensions_.z;
		batch.dispatch(kernels_->edt_line, { &cost_a, &site_a, &cost_b, &site_b }, group_count(*context_, y_lines),
			LinePush{ dimensions_.x, dimensions_.y, dimensions_.z, y_lines, 1 });
		batch.barrier();
		const uint32_t x_lines = dimensions_.y * dimensions_.z;
		batch.dispatch(kernels_->edt_line, { &cost_b, &site_b, &cost_a, &voxels_ }, group_count(*context_, x_lines),
			LinePush{ dimensions_.x, dimensions_.y, dimensions_.z, x_lines, 0 });
		batch.submit_and_wait();
	}

	void DistanceVoxelMap::compute_parallel_banding(const ParallelBandingOptions& options)
	{
		const uint32_t dx = dimensions_.x, dy = dimensions_.y, dz = dimensions_.z;
		auto is_power_of_two = [](uint32_t v) { return v != 0 && (v & (v - 1)) == 0; };
		if (options.m1 == 0 || options.m1 > dz || dz % options.m1 != 0)
			throw VulkanError("DistanceVoxelMap::compute_distances: m1 must be between 1 and dim_z and divide dim_z");
		// With one layer per band the bottom and the top voxel of a band are the same voxel and the inter-band step can only
		// keep one of its two neighbours (a limitation of the CUDA algorithm that makes its result wrong).
		if (options.m1 > 1 && dz / options.m1 < 2)
			throw VulkanError("DistanceVoxelMap::compute_distances: with m1 > 1 a band needs at least two layers");
		if (!is_power_of_two(options.m2) || options.m2 > std::min(dx, dy) || dx % options.m2 != 0 || dy % options.m2 != 0)
			throw VulkanError("DistanceVoxelMap::compute_distances: m2 must be a power of two that divides dim_x and dim_y");
		if (options.m3 == 0 || options.m3 > 16 || dx % options.m3 != 0 || dy % options.m3 != 0)
			throw VulkanError("DistanceVoxelMap::compute_distances: m3 must be between 1 and 16 and divide dim_x and dim_y");
		if (!pba_kernels_)
			pba_kernels_ = std::make_shared<PbaKernels>(*context_);
		PbaKernels& k = *pba_kernels_;

		const uint64_t bytes = voxel_count() * sizeof(uint32_t);
		DeviceBuffer initial(*context_, bytes), forward(*context_, bytes), transposed(*context_, bytes);
		const uint32_t count = static_cast<uint32_t>(voxel_count());
		CommandBatch batch(*context_);

		// phases 2 and 3 along y of a volume nx * ny * nz stored in `data` (result written to `out`)
		auto pass_along_y = [&](const DeviceBuffer& data, DeviceBuffer& out, uint32_t nx, uint32_t ny, uint32_t nz)
		{
			const uint32_t m2 = options.m2;
			const uint32_t lines = nx * m2 * nz;
			batch.dispatch(k.backpointers, { &data, &initial }, group_count(*context_, lines), PbaStackPush{ nx, ny, nz, ny / m2, lines, m2 });
			batch.barrier();
			if (m2 > 1)
			{
				batch.dispatch(k.forward, { &initial, &forward }, group_count(*context_, lines), PbaStackPush{ nx, ny, nz, ny / m2, lines, m2 });
				batch.barrier();
				for (uint32_t bands = m2; bands > 1; bands /= 2)
				{
					const uint32_t pairs = bands / 2;
					const uint32_t merge_lines = nx * pairs * nz;
					batch.dispatch(k.merge, { &initial, &forward }, group_count(*context_, merge_lines),
						PbaStackPush{ nx, ny, nz, ny / bands, merge_lines, pairs });
					batch.barrier();
				}
			}
			const PbaDistancePush push{ nx, ny, nz, options.m3 };
			batch.dispatch(k.distances, { &initial, &out }, (nx + 15) / 16, 1, nz, &push, static_cast<uint32_t>(sizeof(push)));
			batch.barrier();
		};

		// obstacles = voxels that store their own position; the copy keeps them for the inter-band step of phase 1
		batch.dispatch(k.sanitize, { &voxels_ }, group_count(*context_, count), PbaVolumePush{ dx, dy, dz, count });
		batch.barrier();
		batch.copy(voxels_, initial, bytes);

		// phase 1: flood along z in m1 bands
		const uint32_t flood_threads = dx * dy * options.m1;
		const PbaBandPush band_push{ dx, dy, dz, dz / options.m1, flood_threads };
		batch.dispatch(k.flood, { &voxels_ }, group_count(*context_, flood_threads), band_push);
		batch.barrier();
		if (options.m1 > 1)
		{
			batch.dispatch(k.propagate, { &voxels_, &initial }, group_count(*context_, flood_threads), band_push);
			batch.barrier();
			batch.dispatch(k.update, { &initial, &voxels_ }, group_count(*context_, flood_threads), band_push);
			batch.barrier();
		}

		// first pass along y, transpose, second pass along y (the original x), transpose back
		pass_along_y(voxels_, voxels_, dx, dy, dz);
		batch.dispatch(k.transpose, { &voxels_, &transposed }, group_count(*context_, count), PbaVolumePush{ dx, dy, dz, count });
		batch.barrier();
		pass_along_y(transposed, transposed, dy, dx, dz);
		batch.dispatch(k.transpose, { &transposed, &voxels_ }, group_count(*context_, count), PbaVolumePush{ dy, dx, dz, count });
		batch.submit_and_wait();
	}

	int32_t DistanceVoxelMap::squared_obstacle_distance(Coord3ui position) const
	{
		if (position.x >= dimensions_.x || position.y >= dimensions_.y || position.z >= dimensions_.z)
			throw VulkanError("DistanceVoxelMap: position outside the map");
		uint32_t word = 0;
		voxels_.download(&word, sizeof(uint32_t), voxel_index(dimensions_, position) * sizeof(uint32_t));
		return squared_distance_of(word, position);
	}

	int32_t DistanceVoxelMap::obstacle_distance(Coord3ui position) const
	{
		return truncated_root(squared_obstacle_distance(position));
	}

	std::vector<int32_t> DistanceVoxelMap::squared_distances(std::span<const uint32_t> indices) const
	{
		if (indices.empty())
			return {};
		for (const uint32_t index : indices)
			if (index >= voxel_count())
				throw VulkanError("DistanceVoxelMap: voxel index outside the map");
		DeviceBuffer index_buffer(*context_, indices.size_bytes());
		index_buffer.upload(indices.data(), indices.size_bytes());
		DeviceBuffer out(*context_, indices.size() * sizeof(int32_t));
		CommandBatch batch(*context_);
		batch.dispatch(kernels_->query, { &voxels_, &index_buffer, &out }, group_count(*context_, indices.size()),
			QueryPush{ dimensions_.x, dimensions_.y, dimensions_.z, static_cast<uint32_t>(indices.size()) });
		batch.submit_and_wait();
		return out.download_vector<int32_t>(indices.size());
	}

	std::vector<int32_t> DistanceVoxelMap::distances(std::span<const uint32_t> indices) const
	{
		std::vector<int32_t> values = squared_distances(indices);
		for (int32_t& v : values)
			v = truncated_root(v);
		return values;
	}

	std::vector<int8_t> DistanceVoxelMap::extract_distances(int32_t robot_radius) const
	{
		DeviceBuffer out(*context_, voxel_count() * sizeof(uint32_t));
		CommandBatch batch(*context_);
		batch.dispatch(kernels_->extract, { &voxels_, &out }, group_count(*context_, voxel_count()),
			ExtractPush{ dimensions_.x, dimensions_.y, dimensions_.z, static_cast<uint32_t>(voxel_count()), robot_radius });
		batch.submit_and_wait();
		const auto words = out.download_vector<uint32_t>(voxel_count());
		std::vector<int8_t> result(words.size());
		for (std::size_t i = 0; i < words.size(); ++i)
			result[i] = static_cast<int8_t>(words[i]);
		return result;
	}

	std::vector<Coord3ui> DistanceVoxelMap::download_obstacles() const
	{
		const auto words = download_words();
		std::vector<Coord3ui> obstacles(words.size());
		for (std::size_t i = 0; i < words.size(); ++i)
			obstacles[i] = unpack(words[i]);
		return obstacles;
	}

	void DistanceVoxelMap::upload_words(std::span<const uint32_t> words)
	{
		if (words.size() != voxel_count())
			throw VulkanError("DistanceVoxelMap::upload_words: expected " + std::to_string(voxel_count()) + " words, got " + std::to_string(words.size()));
		for (const uint32_t word : words)
			if ((word >> 30) != 0)
				throw VulkanError("DistanceVoxelMap::upload_words: a word has bits above bit 29");
		voxels_.upload(words.data(), words.size_bytes());
	}

	std::vector<uint32_t> DistanceVoxelMap::download_words() const
	{
		return voxels_.download_vector<uint32_t>(voxel_count());
	}
}
