// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/BitVoxelMap.h>

#include <algorithm>

#include <gpu_voxels/vulkan/Log.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kLocalSize = 64;                     // local_size_x of all BitVoxelMap kernels
		constexpr uint32_t kMaxAxisDimension = 1u << 24;        // exactly representable in a float (bounds checks)
		constexpr std::size_t kMaxItemsPerUpload = 4u << 20;    // points / coordinates per upload buffer

		struct InsertPointsPush { uint32_t dim_x, dim_y, dim_z, count, meaning; float voxel_side_length; };
		struct InsertCoordsPush { uint32_t dim_x, dim_y, dim_z, count, meaning; };
		struct ClearMeaningPush { uint32_t voxel_count, meaning; };
		struct CollidePush { uint32_t voxel_count; };
		struct CollideProbPush { uint32_t voxel_count; int32_t threshold; };
		struct InsertMetaPush { uint32_t dim_x, dim_y, dim_z, point_count, sub_cloud_count; float voxel_side_length; };
		struct SelfCollPush { uint32_t dim_x, dim_y, dim_z, first_point, point_count, sub_cloud, meaning; float voxel_side_length; };

		/// Grid-stride kernels: enough groups to cover `items`, capped by the device limit.
		uint32_t group_count(const Context& context, uint64_t items)
		{
			const uint64_t needed = (items + kLocalSize - 1) / kLocalSize;
			const uint64_t cap = std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], 65535);
			return static_cast<uint32_t>(std::clamp<uint64_t>(needed, 1, cap));
		}
	}

	struct BitVoxelMap::Kernels
	{
		explicit Kernels(Context& context)
			: insert_points(context, GVL_SHADER_SPAN(bit_voxel_insert_points_comp), 3, sizeof(InsertPointsPush)),
			  insert_coords(context, GVL_SHADER_SPAN(bit_voxel_insert_coords_comp), 3, sizeof(InsertCoordsPush)),
			  clear_meaning(context, GVL_SHADER_SPAN(bit_voxel_clear_meaning_comp), 1, sizeof(ClearMeaningPush)),
			  collide(context, GVL_SHADER_SPAN(bit_voxel_collide_comp), 3, sizeof(CollidePush)),
			  collide_types(context, GVL_SHADER_SPAN(bit_voxel_collide_types_comp), 3, sizeof(CollidePush)),
			  collide_prob(context, GVL_SHADER_SPAN(bit_voxel_collide_prob_comp), 3, sizeof(CollideProbPush)),
			  insert_meta(context, GVL_SHADER_SPAN(bit_voxel_insert_meta_comp), 5, sizeof(InsertMetaPush)),
			  insert_meta_selfcoll(context, GVL_SHADER_SPAN(bit_voxel_insert_meta_selfcoll_comp), 5, sizeof(SelfCollPush))
		{
		}

		ComputePipeline insert_points;
		ComputePipeline insert_coords;
		ComputePipeline clear_meaning;
		ComputePipeline collide;
		ComputePipeline collide_types;
		ComputePipeline collide_prob;
		ComputePipeline insert_meta;
		ComputePipeline insert_meta_selfcoll;
	};

	BitVoxelMap::BitVoxelMap(Context& context, Dim3ui dimensions, float voxel_side_length)
		: context_(context), dimensions_(dimensions), voxel_side_length_(voxel_side_length),
		  voxels_(context, [&]
		  {
			  if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0)
				  throw VulkanError("BitVoxelMap: empty dimensions");
			  if (dimensions.x > kMaxAxisDimension || dimensions.y > kMaxAxisDimension || dimensions.z > kMaxAxisDimension)
				  throw VulkanError("BitVoxelMap: more than 2^24 voxels along one axis");
			  if (!(voxel_side_length > 0.0f))
				  throw VulkanError("BitVoxelMap: voxel side length must be positive");
			  const uint64_t words = dimensions.voxel_count() * kWordsPerVoxel;
			  if (words > UINT32_MAX)
				  throw VulkanError("BitVoxelMap: voxel word index does not fit into 32 bits");
			  return dimensions.voxel_count() * kBytesPerVoxel; // DeviceBuffer rejects sizes above maxStorageBufferRange
		  }()),
		  scratch_counter_(context, sizeof(uint32_t)),
		  kernels_(std::make_shared<Kernels>(context))
	{
		clear_map();
	}

	void BitVoxelMap::clear_map()
	{
		voxels_.fill(0);
	}

	void BitVoxelMap::clear_bit_voxel_meaning(uint8_t meaning)
	{
		const ClearMeaningPush push{ static_cast<uint32_t>(voxel_count()), meaning };
		CommandBatch batch(context_);
		batch.dispatch(kernels_->clear_meaning, { &voxels_ }, group_count(context_, voxel_count()), push);
		batch.submit_and_wait();
	}

	uint32_t BitVoxelMap::insert_point_cloud(std::span<const Point3f> points, uint8_t meaning)
	{
		static_assert(sizeof(Point3f) == 12);
		uint32_t outside = 0;
		for (std::size_t first = 0; first < points.size(); first += kMaxItemsPerUpload)
		{
			const auto chunk = points.subspan(first, std::min(kMaxItemsPerUpload, points.size() - first));
			DeviceBuffer point_buffer(context_, chunk.size_bytes());
			point_buffer.upload(chunk.data(), chunk.size_bytes());

			const InsertPointsPush push{ dimensions_.x, dimensions_.y, dimensions_.z,
				static_cast<uint32_t>(chunk.size()), meaning, voxel_side_length_ };
			CommandBatch batch(context_);
			batch.fill(scratch_counter_, 0);
			batch.dispatch(kernels_->insert_points, { &voxels_, &point_buffer, &scratch_counter_ },
				group_count(context_, chunk.size()), push);
			batch.submit_and_wait();
			outside += scratch_counter_.download_vector<uint32_t>(1).front();
		}
		if (outside > 0)
			GVL_LOG_WARNING(BitVoxelMap, outside << " point(s) lie outside the map dimensions!");
		return outside;
	}

	uint32_t BitVoxelMap::insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning)
	{
		static_assert(sizeof(Coord3ui) == 12);
		uint32_t outside = 0;
		for (std::size_t first = 0; first < coordinates.size(); first += kMaxItemsPerUpload)
		{
			const auto chunk = coordinates.subspan(first, std::min(kMaxItemsPerUpload, coordinates.size() - first));
			DeviceBuffer coord_buffer(context_, chunk.size_bytes());
			coord_buffer.upload(chunk.data(), chunk.size_bytes());

			const InsertCoordsPush push{ dimensions_.x, dimensions_.y, dimensions_.z,
				static_cast<uint32_t>(chunk.size()), meaning };
			CommandBatch batch(context_);
			batch.fill(scratch_counter_, 0);
			batch.dispatch(kernels_->insert_coords, { &voxels_, &coord_buffer, &scratch_counter_ },
				group_count(context_, chunk.size()), push);
			batch.submit_and_wait();
			outside += scratch_counter_.download_vector<uint32_t>(1).front();
		}
		if (outside > 0)
			GVL_LOG_WARNING(BitVoxelMap, outside << " coordinate(s) lie outside the map dimensions!");
		return outside;
	}

	uint32_t BitVoxelMap::collide_with(const ProbVoxelMap& other, float threshold)
	{
		if (&other.context() != &context_)
			throw VulkanError("BitVoxelMap::collide_with: different contexts");
		if (!(other.dimensions() == dimensions_))
			throw VulkanError("BitVoxelMap::collide_with: dimensions differ");
		const CollideProbPush push{ static_cast<uint32_t>(voxel_count()), ProbVoxelMap::probability_from_float(threshold) };
		CommandBatch batch(context_);
		batch.fill(scratch_counter_, 0);
		batch.dispatch(kernels_->collide_prob, { &voxels_, &other.voxel_buffer(), &scratch_counter_ },
			group_count(context_, voxel_count()), push);
		batch.submit_and_wait();
		return scratch_counter_.download_vector<uint32_t>(1).front();
	}

	uint32_t BitVoxelMap::merge_voxel_list(const VoxelListBase& list)
	{
		if (&list.context() != &context_)
			throw VulkanError("BitVoxelMap::merge_voxel_list: different contexts");
		const uint32_t count = list.size();
		if (count == 0)
			return 0;
		const InsertCoordsPush push{ dimensions_.x, dimensions_.y, dimensions_.z, count, eBVM_OCCUPIED };
		CommandBatch batch(context_);
		batch.fill(scratch_counter_, 0);
		batch.dispatch(kernels_->insert_coords, { &voxels_, &list.coordinates_buffer(), &scratch_counter_ },
			group_count(context_, count), push);
		batch.submit_and_wait();
		const uint32_t outside = scratch_counter_.download_vector<uint32_t>(1).front();
		if (outside > 0)
			GVL_LOG_WARNING(BitVoxelMap, outside << " list coordinate(s) lie outside the map dimensions!");
		return outside;
	}

	uint32_t BitVoxelMap::insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning)
	{
		const std::vector<uint8_t> meanings(cloud.num_clouds(), meaning);
		return insert_meta_point_cloud(cloud, meanings);
	}

	uint32_t BitVoxelMap::insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings)
	{
		if (meanings.size() != cloud.num_clouds())
			throw VulkanError("BitVoxelMap::insert_meta_point_cloud: expected one meaning per sub cloud");
		if (&cloud.context() != &context_)
			throw VulkanError("BitVoxelMap::insert_meta_point_cloud: different contexts");
		if (cloud.empty())
			return 0;

		std::vector<uint32_t> meaning_words(meanings.begin(), meanings.end());
		DeviceBuffer offsets(context_, cloud.offsets().size() * sizeof(uint32_t));
		offsets.upload(cloud.offsets().data(), cloud.offsets().size() * sizeof(uint32_t));
		DeviceBuffer meaning_buffer(context_, meaning_words.size() * sizeof(uint32_t));
		meaning_buffer.upload(meaning_words.data(), meaning_words.size() * sizeof(uint32_t));

		const InsertMetaPush push{ dimensions_.x, dimensions_.y, dimensions_.z, cloud.accumulated_size(),
			cloud.num_clouds(), voxel_side_length_ };
		CommandBatch batch(context_);
		batch.fill(scratch_counter_, 0);
		batch.dispatch(kernels_->insert_meta, { &voxels_, &cloud.points(), &offsets, &meaning_buffer, &scratch_counter_ },
			group_count(context_, cloud.accumulated_size()), push);
		batch.submit_and_wait();
		const uint32_t outside = scratch_counter_.download_vector<uint32_t>(1).front();
		if (outside > 0)
			GVL_LOG_WARNING(BitVoxelMap, outside << " point(s) lie outside the map dimensions!");
		return outside;
	}

	BitVoxelMap::SelfCollisionResult BitVoxelMap::insert_meta_point_cloud_with_self_collision_check(
		const MetaPointCloud& cloud, std::span<const uint8_t> meanings, std::span<const BitVector256> masks)
	{
		const uint32_t links = cloud.num_clouds();
		if (&cloud.context() != &context_)
			throw VulkanError("BitVoxelMap: different contexts");

		std::vector<uint8_t> meaning_list(meanings.begin(), meanings.end());
		if (meaning_list.empty())
		{
			if (links > eBVM_SWEPT_VOLUME_END - eBVM_SWEPT_VOLUME_START)
				throw VulkanError("BitVoxelMap: too many sub clouds for the default swept volume meanings");
			for (uint32_t k = 0; k < links; ++k)
				meaning_list.push_back(static_cast<uint8_t>(eBVM_SWEPT_VOLUME_START + k));
		}
		else if (meaning_list.size() != links)
			throw VulkanError("BitVoxelMap: expected one meaning per sub cloud");

		std::vector<BitVector256> mask_list(masks.begin(), masks.end());
		if (mask_list.empty())
		{
			for (uint32_t k = 0; k < links; ++k)
			{
				BitVector256 mask;
				mask.fill(0xFFFFFFFFu);
				mask[eBVM_COLLISION / 32] &= ~(1u << (eBVM_COLLISION % 32));       // never collide with the collision marker
				mask[meaning_list[k] / 32] &= ~(1u << (meaning_list[k] % 32));      // nor with the own meaning
				mask_list.push_back(mask);
			}
		}
		else if (mask_list.size() != links)
			throw VulkanError("BitVoxelMap: expected one collision mask per sub cloud");

		SelfCollisionResult result;
		if (cloud.empty())
			return result;

		DeviceBuffer mask_buffer(context_, mask_list.size() * sizeof(BitVector256));
		mask_buffer.upload(mask_list.data(), mask_list.size() * sizeof(BitVector256));
		DeviceBuffer colliding(context_, sizeof(BitVector256));

		CommandBatch batch(context_);
		batch.fill(scratch_counter_, 0);
		batch.fill(colliding, 0);
		// One dispatch per sub cloud, ordered by barriers: sub cloud k must see the voxels written by sub clouds < k.
		for (uint32_t k = 0; k < links; ++k)
		{
			if (cloud.cloud_size(k) == 0)
				continue;
			const SelfCollPush push{ dimensions_.x, dimensions_.y, dimensions_.z, cloud.cloud_offset(k), cloud.cloud_size(k), k,
				meaning_list[k], voxel_side_length_ };
			batch.dispatch(kernels_->insert_meta_selfcoll, { &voxels_, &cloud.points(), &mask_buffer, &colliding, &scratch_counter_ },
				group_count(context_, cloud.cloud_size(k)), push);
			batch.barrier();
		}
		batch.submit_and_wait();

		result.points_outside = scratch_counter_.download_vector<uint32_t>(1).front();
		const auto words = colliding.download_vector<uint32_t>(kWordsPerVoxel);
		std::copy(words.begin(), words.end(), result.colliding_meanings.begin());
		result.colliding = std::any_of(words.begin(), words.end(), [](uint32_t w) { return w != 0; });
		if (result.points_outside > 0)
			GVL_LOG_WARNING(BitVoxelMap, result.points_outside << " point(s) lie outside the map dimensions!");
		return result;
	}

	uint32_t BitVoxelMap::collide_with(const BitVoxelMap& other)
	{
		if (&other == this)
			throw VulkanError("BitVoxelMap::collide_with: a map cannot be collided with itself");
		if (!(other.dimensions_ == dimensions_))
			throw VulkanError("BitVoxelMap::collide_with: dimensions differ");

		const CollidePush push{ static_cast<uint32_t>(voxel_count()) };
		CommandBatch batch(context_);
		batch.fill(scratch_counter_, 0);
		batch.dispatch(kernels_->collide, { &voxels_, &other.voxels_, &scratch_counter_ },
			group_count(context_, voxel_count()), push);
		batch.submit_and_wait();
		return scratch_counter_.download_vector<uint32_t>(1).front();
	}

	BitVoxelMap::TypeCollision BitVoxelMap::collide_with_types(const BitVoxelMap& other)
	{
		if (&other == this)
			throw VulkanError("BitVoxelMap::collide_with_types: a map cannot be collided with itself");
		if (!(other.dimensions_ == dimensions_))
			throw VulkanError("BitVoxelMap::collide_with_types: dimensions differ");

		DeviceBuffer result(context_, (1 + kWordsPerVoxel) * sizeof(uint32_t));
		const CollidePush push{ static_cast<uint32_t>(voxel_count()) };
		CommandBatch batch(context_);
		batch.fill(result, 0);
		batch.dispatch(kernels_->collide_types, { &voxels_, &other.voxels_, &result },
			group_count(context_, voxel_count()), push);
		batch.submit_and_wait();

		const auto words = result.download_vector<uint32_t>(1 + kWordsPerVoxel);
		TypeCollision out;
		out.count = words[0];
		std::copy(words.begin() + 1, words.end(), out.types.begin());
		return out;
	}

	void BitVoxelMap::upload_words(std::span<const uint32_t> words)
	{
		if (words.size() != voxel_count() * kWordsPerVoxel)
			throw VulkanError("BitVoxelMap::upload_words: expected " + std::to_string(voxel_count() * kWordsPerVoxel) + " words, got " + std::to_string(words.size()));
		voxels_.upload(words.data(), words.size_bytes());
	}

	std::vector<uint32_t> BitVoxelMap::download_words(uint64_t first_voxel, uint64_t count) const
	{
		if (first_voxel > voxel_count())
			throw VulkanError("BitVoxelMap::download_words: first voxel out of range");
		count = std::min(count, voxel_count() - first_voxel);
		std::vector<uint32_t> words(count * kWordsPerVoxel);
		if (count > 0)
			voxels_.download(words.data(), count * kBytesPerVoxel, first_voxel * kBytesPerVoxel);
		return words;
	}
}
