// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Dense 3D map storing for every voxel the position of its nearest obstacle (slice S5).
 *
 * Semantics follow the CUDA classes voxelmap::DistanceVoxelMap / DistanceVoxel (parallelBanding3D, squaredObstacleDistance,
 * extract_distances, mergeOccupied). Documented deviations from CUDA are listed in
 * Agent/knowledge_base/architecture/vulkan_distance_map.md.
 */
#ifndef GPU_VOXELS_VULKAN_DISTANCE_VOXEL_MAP_H_INCLUDED
#define GPU_VOXELS_VULKAN_DISTANCE_VOXEL_MAP_H_INCLUDED

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	class ProbVoxelMap;

	/// Algorithm of DistanceVoxelMap::compute_distances().
	enum class DistanceAlgorithm
	{
		/// Three exact separable passes (z, y, x), one invocation per line, integer lower envelope (Meijster et al.). Any dimensions up to 1023.
		Exact,
		/// Parallel Banding Algorithm (Cao et al. 2010) as in CUDA parallelBanding3D: flood along z, Voronoi diagram along y,
		/// transpose, Voronoi diagram along y again, transpose back; every phase can be split into bands that are processed in
		/// parallel. More parallelism per line than Exact (useful for few, long lines); needs 3 extra buffers of the map size.
		ParallelBanding
	};

	/// Band counts of the ParallelBanding algorithm (CUDA parallelBanding3D m1, m2, m3). Invalid values throw (CUDA falls back
	/// to 1 with a warning, or silently computes garbage for a band count that is no power of two).
	struct ParallelBandingOptions
	{
		uint32_t m1 = 1; ///< bands of phase 1 (along z); must divide dim_z; with m1 > 1 every band needs >= 2 layers
		uint32_t m2 = 1; ///< bands of phase 2 (along y, later x); a power of two that divides dim_x and dim_y
		uint32_t m3 = 1; ///< invocations per line in phase 3; at most 16 and a divisor of dim_x and dim_y
	};

	class DistanceVoxelMap
	{
	public:
		/// CUDA packs the nearest obstacle as 3 x 10 bit; 1023 marks "uninitialised", so coordinates are at most 1022.
		static constexpr uint32_t kMaxAxisDimension = 1023;
		static constexpr uint32_t kUninitialisedCoordinate = 1023;
		/// squaredObstacleDistance of a voxel without a known obstacle (CUDA MAX_OBSTACLE_DISTANCE).
		static constexpr int32_t kMaxObstacleDistance = 2147483647;
		static constexpr uint32_t kBytesPerVoxel = 4;

		/// All voxels start uninitialised (no obstacle).
		/// \throws VulkanError for empty dimensions, more than 1023 voxels along an axis, a non-positive side length, or a
		///         map larger than maxStorageBufferRange
		DistanceVoxelMap(Context& context, Dim3ui dimensions, float voxel_side_length);
		~DistanceVoxelMap();
		DistanceVoxelMap(DistanceVoxelMap&&) noexcept;
		DistanceVoxelMap& operator=(DistanceVoxelMap&&) noexcept;
		DistanceVoxelMap(const DistanceVoxelMap&) = delete;
		DistanceVoxelMap& operator=(const DistanceVoxelMap&) = delete;

		[[nodiscard]] Context& context() const { return *context_; }
		[[nodiscard]] Dim3ui dimensions() const { return dimensions_; }
		[[nodiscard]] float voxel_side_length() const { return voxel_side_length_; }
		[[nodiscard]] uint64_t voxel_count() const { return dimensions_.voxel_count(); }
		[[nodiscard]] uint64_t memory_usage_bytes() const { return voxel_count() * kBytesPerVoxel; }

		/// Sets every voxel to uninitialised.
		void clear_map();

		/// Marks the voxel floor(p / voxel_side_length) of every point as obstacle (the voxel stores its own position).
		/// \param meaning must be eBVM_OCCUPIED; CUDA prints an error and un-marks the voxel for anything else, here other
		///        meanings throw
		/// \return number of points outside the map
		uint32_t insert_point_cloud(std::span<const Point3f> points, uint8_t meaning = eBVM_OCCUPIED);
		/// \return number of coordinates outside the map
		uint32_t insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning = eBVM_OCCUPIED);
		/// All sub clouds as obstacles. \return number of points outside the map.
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning = eBVM_OCCUPIED);

		/// CUDA mergeOccupied (without offset): every voxel of `other` with occupancy > probability_from_float(threshold)
		/// becomes an obstacle. \throws VulkanError if the dimensions differ
		void merge_occupied(const ProbVoxelMap& other, float occupancy_threshold = 0.5f);

		/// Computes for every voxel the nearest obstacle (CUDA parallelBanding3D). Obstacles are the voxels that store their
		/// own position; all other content is ignored and overwritten, so the call can be repeated after inserting more
		/// obstacles (CUDA requires a clear map in between). The squared Euclidean distance is exact; among equally near
		/// obstacles the choice is deterministic but unspecified. Without any obstacle all voxels stay uninitialised.
		void compute_distances(DistanceAlgorithm algorithm = DistanceAlgorithm::Exact, const ParallelBandingOptions& pba = {});

		/// CUDA DistanceVoxel::squaredObstacleDistance of one voxel (kMaxObstacleDistance if uninitialised).
		/// \throws VulkanError if the position lies outside the map
		[[nodiscard]] int32_t squared_obstacle_distance(Coord3ui position) const;
		/// Truncated square root of the above (CUDA getObstacleDistance).
		[[nodiscard]] int32_t obstacle_distance(Coord3ui position) const;
		/// Squared distances of the voxels with the given linear indices (z * dx * dy + y * dx + x).
		[[nodiscard]] std::vector<int32_t> squared_distances(std::span<const uint32_t> indices) const;
		/// Truncated distances of the voxels with the given linear indices.
		[[nodiscard]] std::vector<int32_t> distances(std::span<const uint32_t> indices) const;

		/// CUDA extract_distances: int8 free-space distance per voxel for a robot radius in voxels: voxels closer than
		/// `robot_radius` to a map border get their distance to the border, all others the truncated distance to the nearest
		/// obstacle capped at 127 (uninitialised voxels: 127).
		[[nodiscard]] std::vector<int8_t> extract_distances(int32_t robot_radius) const;

		/// Nearest obstacle of every voxel (all coordinates 1023 = uninitialised).
		[[nodiscard]] std::vector<Coord3ui> download_obstacles() const;
		/// Packed voxels (x | y << 10 | z << 20).
		[[nodiscard]] std::vector<uint32_t> download_words() const;

		/// Replaces all voxels by packed words (voxel_count() words, the layout of download_words(): x | y << 10 | z << 20).
		/// 	hrows VulkanError for another number of words or a word with bits above bit 29; the map is unchanged then
		void upload_words(std::span<const uint32_t> words);

		/// Device buffer with the packed voxels for other Vulkan code; read-only for callers.
		[[nodiscard]] const DeviceBuffer& voxel_buffer() const { return voxels_; }

		[[nodiscard]] static uint32_t pack(Coord3ui c) { return c.x | (c.y << 10) | (c.z << 20); }
		[[nodiscard]] static Coord3ui unpack(uint32_t word) { return { word & 1023u, (word >> 10) & 1023u, word >> 20 }; }
		[[nodiscard]] static constexpr uint32_t uninitialised_word() { return 0x3FFFFFFFu; }
		/// Linear voxel index: z * dim_x * dim_y + y * dim_x + x.
		[[nodiscard]] static uint64_t voxel_index(Dim3ui dimensions, Coord3ui c)
		{
			return (static_cast<uint64_t>(c.z) * dimensions.y + c.y) * dimensions.x + c.x;
		}

	private:
		struct Kernels;
		struct PbaKernels;

		void compute_exact();
		void compute_parallel_banding(const ParallelBandingOptions& options);

		uint32_t insert_raw(const DeviceBuffer& buffer, uint32_t count, uint32_t input_kind);

		Context* context_;
		Dim3ui dimensions_;
		float voxel_side_length_;
		DeviceBuffer voxels_;
		DeviceBuffer scratch_counter_;
		std::shared_ptr<Kernels> kernels_;
		std::shared_ptr<PbaKernels> pba_kernels_; ///< created on first ParallelBanding use
	};
}

#endif
