// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Dense 3D map of BitVoxel<256> on the Vulkan backend (slice S1 of the migration plan).
 *
 * Semantics follow the CUDA classes voxelmap::BitVoxelMap / TemplateVoxelMap<BitVectorVoxel>
 * (kernelInsertGlobalPointCloud, kernelInsertCoordinateTuples, kernelClearVoxelMap, kernelCollideVoxelMapsDebug).
 * Documented deviations from CUDA are listed in Agent/knowledge_base/architecture/vulkan_bit_voxel_map.md.
 */
#ifndef GPU_VOXELS_VULKAN_BIT_VOXEL_MAP_H_INCLUDED
#define GPU_VOXELS_VULKAN_BIT_VOXEL_MAP_H_INCLUDED

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	class VoxelListBase;
	class ProbVoxelMap;

	class BitVoxelMap
	{
	public:
		static constexpr uint32_t kBitLength = 256;
		static constexpr uint32_t kWordsPerVoxel = kBitLength / 32;
		static constexpr uint32_t kBytesPerVoxel = kBitLength / 8;

		/// All voxels start as zero (eBVM_FREE state, nothing set).
		/// \throws VulkanError for empty dimensions, dimensions above 2^24 per axis, or a map larger than
		///         maxStorageBufferRange (maps are bound as one buffer; range binding is not implemented yet)
		BitVoxelMap(Context& context, Dim3ui dimensions, float voxel_side_length);

		[[nodiscard]] Context& context() const { return context_; }
		[[nodiscard]] Dim3ui dimensions() const { return dimensions_; }
		[[nodiscard]] float voxel_side_length() const { return voxel_side_length_; }
		[[nodiscard]] uint64_t voxel_count() const { return dimensions_.voxel_count(); }
		[[nodiscard]] uint64_t memory_usage_bytes() const { return voxel_count() * kBytesPerVoxel; }

		/// Sets every voxel to zero.
		void clear_map();
		/// Clears one meaning (bit) in all voxels.
		void clear_bit_voxel_meaning(uint8_t meaning);

		/// Inserts each point into voxel floor(p / voxel_side_length) with the given meaning.
		/// \return number of points that were not inserted because they lie outside the map
		///         (negative or too large coordinates, NaN, inf)
		uint32_t insert_point_cloud(std::span<const Point3f> points, uint8_t meaning);
		/// Inserts voxel coordinates. \return number of coordinates outside the map.
		uint32_t insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning);

		/// Inserts all sub clouds of a MetaPointCloud with the same meaning. \return number of points outside the map.
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning);
		/// Sub cloud k is inserted with meanings[k] (one meaning per sub cloud). \return number of points outside the map.
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings);

		struct SelfCollisionResult
		{
			bool colliding = false;               ///< any sub cloud hit voxels of another (unmasked) sub cloud
			BitVector256 colliding_meanings{};    ///< meanings involved in self collisions
			uint32_t points_outside = 0;
		};

		/// Inserts the sub clouds one after the other and detects collisions between them
		/// (CUDA insertMetaPointCloudWithSelfCollisionCheck).
		/// \param meanings empty: meaning of sub cloud k = eBVM_SWEPT_VOLUME_START + k, otherwise one meaning per sub cloud
		/// \param masks    empty: every sub cloud collides with everything except itself and eBVM_COLLISION, otherwise
		///                one 256-bit mask per sub cloud (bits that count as collision partners)
		/// The result does not depend on thread scheduling as long as no mask contains the own meaning bit of its sub
		/// cloud or eBVM_COLLISION (true for the default masks); otherwise it depends on the insertion order, like in CUDA.
		SelfCollisionResult insert_meta_point_cloud_with_self_collision_check(const MetaPointCloud& cloud,
			std::span<const uint8_t> meanings = {}, std::span<const BitVector256> masks = {});

		/// Voxel-wise collision with another map of the same dimensions. A voxel is occupied if any bit except
		/// eBVM_FREE is set; the voxels where both maps are occupied collide.
		/// Side effect (as in CUDA): colliding voxels of *this* map get the eBVM_COLLISION bit.
		/// \return number of colliding voxels
		/// \throws VulkanError if the dimensions differ or `other` is this map
		uint32_t collide_with(const BitVoxelMap& other);

		/// Result of collide_with_types().
		struct TypeCollision
		{
			uint32_t count = 0;                                ///< number of colliding voxels
			std::array<uint32_t, kWordsPerVoxel> types{};      ///< OR of the colliding bits (bit i = meaning i)
		};

		/// Type-aware collision (CUDA collideWithTypes with SVCollider, window 0): two voxels collide when they share
		/// a bit >= 4, i.e. any meaning except FREE / OCCUPIED / COLLISION / UNKNOWN; `types` aggregates the shared bits.
		/// Side effect (as in CUDA): colliding voxels of *this* map get the eBVM_COLLISION bit.
		/// Not implemented: SVCollider window size > 0 and sv_offset (not reachable through the CUDA public API).
		TypeCollision collide_with_types(const BitVoxelMap& other);

		/// CUDA merge(VoxelList) without offset: inserts the coordinates of every list entry with eBVM_OCCUPIED (the payload
		/// of the list is ignored, as in CUDA). \return number of coordinates outside this map
		uint32_t merge_voxel_list(const VoxelListBase& list);

		/// Collision with a probabilistic map of the same dimensions (CUDA DefaultCollider(Bit, Prob)): this voxel has a bit
		/// except FREE and the other voxel is >= ProbVoxelMap::probability_from_float(threshold) (CUDA default 1.0 -> 127).
		/// Side effect (as in CUDA): colliding voxels of *this* map get eBVM_COLLISION. \return number of colliding voxels
		uint32_t collide_with(const ProbVoxelMap& other, float threshold = 1.0f);

		/// Copies voxel words to the host: `voxel_count` voxels starting at `first_voxel`, kWordsPerVoxel words each.
		/// Bit i of a voxel is bit (i % 32) of word (i / 32).
		[[nodiscard]] std::vector<uint32_t> download_words(uint64_t first_voxel = 0, uint64_t voxel_count = UINT64_MAX) const;

		/// Device buffer with the voxels (kWordsPerVoxel words each) for other Vulkan code; read-only for callers.
		[[nodiscard]] const DeviceBuffer& voxel_buffer() const { return voxels_; }

		/// Linear voxel index (CUDA getVoxelIndexUnsigned): z * dim_x * dim_y + y * dim_x + x.
		[[nodiscard]] static uint64_t voxel_index(Dim3ui dimensions, Coord3ui c)
		{
			return (static_cast<uint64_t>(c.z) * dimensions.y + c.y) * dimensions.x + c.x;
		}

	private:
		struct Kernels;

		Context& context_;
		Dim3ui dimensions_;
		float voxel_side_length_;
		DeviceBuffer voxels_;
		DeviceBuffer scratch_counter_; ///< one uint32: outside / collision counter
		std::shared_ptr<Kernels> kernels_;
	};
}

#endif
