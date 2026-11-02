// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Dense 3D map of probabilistic voxels (int8 log-odds occupancy) on the Vulkan backend (slice S3).
 *
 * Semantics follow the CUDA classes voxelmap::ProbVoxelMap / TemplateVoxelMap<ProbabilisticVoxel>
 * (ProbabilisticVoxel::updateOccupancy / insert, kernelInsertSensorData, RayCaster, DefaultCollider).
 * Documented deviations from CUDA are listed in Agent/knowledge_base/architecture/vulkan_prob_voxel_map.md.
 */
#ifndef GPU_VOXELS_VULKAN_PROB_VOXEL_MAP_H_INCLUDED
#define GPU_VOXELS_VULKAN_PROB_VOXEL_MAP_H_INCLUDED

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
	class BitVoxelMap;
	class Primitives;

	/// How concurrent voxel updates of one call are combined (the analogue of a memory order for the voxel updates).
	/// The update rule of a single update is CUDA's `updateOccupancy`: an unknown voxel takes the increment, a known
	/// voxel adds it and saturates at -127 / +127. Because of the saturation the result depends on the order of the
	/// updates of one voxel whenever a voxel saturates inside one call.
	enum class UpdateMode
	{
		/// Every update is one atomic read-modify-write (no lost updates, unlike CUDA). The interleaving of updates of
		/// the same voxel is unspecified: results are reproducible only if no voxel saturates; otherwise the result is
		/// that of SOME serial order of the updates. Cost: proportional to the number of updates.
		Atomic,
		/// The updates of a call are summed per voxel and applied once: clamp(old + sum), an unknown voxel counts as 0.
		/// Deterministic and independent of the order. Equals Sequential unless a voxel saturates inside the call.
		/// Cost: updates plus one pass over the whole map and 5 extra bytes per voxel (allocated on first use).
		Accumulate,
		/// Exactly the sequential result: updates are applied in measurement order (a measurement's free-space voxels in
		/// ray order, then its hit voxel), saturating after every step. Deterministic and equal to a single-threaded CPU
		/// run. Cost: generate, sort and replay all updates (in chunks, see UpdateOptions::max_sequential_events).
		Sequential
	};

	/// Occupancy updates in int8 log-odds units (CUDA cSENSOR_MODEL_FREE / cSENSOR_MODEL_OCCUPIED). Both must lie in
	/// [-127, 127].
	struct SensorModel
	{
		int8_t free_update = -10;      ///< added to voxels a ray passes through (eBVM_FREE)
		int8_t occupied_update = 72;   ///< added to the measured voxel (eBVM_OCCUPIED, eBVM_COLLISION)
	};

	struct UpdateOptions
	{
		SensorModel model{};
		UpdateMode mode = UpdateMode::Accumulate;
		/// Sequential mode: upper limit of voxel updates held at once. Chunks of consecutive measurements with at most this
		/// many updates are applied one after the other (same result); a single measurement with more updates is rejected.
		/// 1 .. ProbVoxelMap::kMaxSequentialEvents.
		uint32_t max_sequential_events = 1u << 22;
	};

	class ProbVoxelMap
	{
	public:
		static constexpr int8_t kUnknown = -128;
		static constexpr int8_t kMinProbability = -127;
		static constexpr int8_t kMaxProbability = 127;
		/// Largest allowed UpdateOptions::max_sequential_events (about 40 bytes per update while sorting).
		static constexpr uint32_t kMaxSequentialEvents = 1u << 22;
		/// Bytes per voxel (one 32-bit word holding the sign extended int8; packing is an open optimisation).
		static constexpr uint32_t kBytesPerVoxel = 4;

		/// All voxels start unknown.
		/// \throws VulkanError for empty dimensions, dimensions above 2^24 per axis, a non-positive side length, or a map
		///         larger than maxStorageBufferRange
		ProbVoxelMap(Context& context, Dim3ui dimensions, float voxel_side_length);
		~ProbVoxelMap();
		ProbVoxelMap(ProbVoxelMap&&) noexcept;
		ProbVoxelMap& operator=(ProbVoxelMap&&) noexcept;
		ProbVoxelMap(const ProbVoxelMap&) = delete;
		ProbVoxelMap& operator=(const ProbVoxelMap&) = delete;

		[[nodiscard]] Context& context() const { return *context_; }
		[[nodiscard]] Dim3ui dimensions() const { return dimensions_; }
		[[nodiscard]] float voxel_side_length() const { return voxel_side_length_; }
		[[nodiscard]] uint64_t voxel_count() const { return dimensions_.voxel_count(); }
		[[nodiscard]] uint64_t memory_usage_bytes() const { return voxel_count() * kBytesPerVoxel; }

		/// Sets every voxel to unknown.
		void clear_map();
		/// CUDA only supports eBVM_OCCUPIED (clears the whole map). \throws VulkanError for every other meaning.
		void clear_bit_voxel_meaning(uint8_t meaning);

		/// Updates the voxel floor(p / voxel_side_length) of every point with the update of `meaning`
		/// (ProbabilisticVoxel::insert): FREE -> model.free_update, OCCUPIED / COLLISION -> model.occupied_update,
		/// UNKNOWN / UNDEFINED -> set unknown, swept volume meanings 4..254 -> meaning - 129.
		/// \return number of points outside the map (negative, too large, NaN, inf)
		uint32_t insert_point_cloud(std::span<const Point3f> points, uint8_t meaning, const UpdateOptions& options = {});
		/// \return number of coordinates outside the map
		uint32_t insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning, const UpdateOptions& options = {});
		/// All sub clouds with the same meaning. \return number of points outside the map.
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning, const UpdateOptions& options = {});
		/// Sub cloud k with meanings[k]; sub clouds are applied one after the other in this order (matters for UNKNOWN).
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings, const UpdateOptions& options = {});

		struct SensorResult
		{
			uint32_t outside = 0;      ///< measurements dropped: hit or sensor origin outside the map, or NaN
			uint32_t cut_by_robot = 0; ///< measurements dropped because their voxel is marked in the robot map
		};

		/// CUDA ProbVoxelMap::insertSensorData for points that are already in map coordinates. A measurement whose hit
		/// voxel or whose sensor voxel lies outside the map is dropped completely. Otherwise, with `raycasting`, every
		/// voxel of the 6-connected walk from the sensor voxel to the hit voxel (both included) gets
		/// model.free_update, and afterwards the hit voxel gets model.occupied_update.
		/// \param robot_map if not null (cut_real_robot): measurements in voxels where the robot map has bit
		///        `robot_voxel_meaning` set are dropped; must have the same dimensions
		/// \throws VulkanError if a single measurement needs more than options.max_sequential_events updates in Sequential
		///         mode. Note: a ray is walked by ONE shader invocation; rays of millions of voxels can exceed the GPU
		///         watchdog (observed: device lost for a 16.7 M voxel ray on the Intel UHD).
		SensorResult insert_sensor_data(std::span<const Point3f> points, Point3f sensor_pose, bool raycasting,
			const BitVoxelMap* robot_map = nullptr, uint8_t robot_voxel_meaning = eBVM_OCCUPIED, const UpdateOptions& options = {});

		/// Voxel-wise collision with a bit map of the same dimensions (CUDA DefaultCollider): this voxel >= the threshold
		/// and the other voxel has a bit except FREE. Side effect (as in CUDA): every colliding voxel of *this* gets one
		/// update with model.occupied_update (insert(eBVM_COLLISION)).
		/// \param threshold probability in [0, 1] converted with probability_from_float() (CUDA default 1.0 -> 127)
		/// \return number of colliding voxels
		uint32_t collide_with(const BitVoxelMap& other, float threshold = 1.0f, const SensorModel& model = {});
		/// Both voxels >= the threshold. Same side effect on *this*.
		uint32_t collide_with(const ProbVoxelMap& other, float threshold = 1.0f, const SensorModel& model = {});

		/// Occupancy of `voxel_count` voxels from `first_voxel` (-128 = unknown).
		[[nodiscard]] std::vector<int8_t> download_occupancy(uint64_t first_voxel = 0, uint64_t voxel_count = UINT64_MAX) const;

		/// Replaces the occupancy of all voxels (voxel_count() values, -128 = unknown, the layout of download_occupancy()).
		/// 	hrows VulkanError for another number of values; the map is unchanged then
		void upload_occupancy(std::span<const int8_t> values);

		/// Device buffer with one 32-bit word per voxel (sign extended int8) for other Vulkan code; read-only for callers.
		[[nodiscard]] const DeviceBuffer& voxel_buffer() const { return voxels_; }

		/// CUDA ProbabilisticVoxel::floatToProbability: clamp to [0, 1] (NaN -> 1), then -127 + p * 254, truncated.
		[[nodiscard]] static int8_t probability_from_float(float value);
		/// CUDA ProbabilisticVoxel::probabilityToFloat.
		[[nodiscard]] static float probability_to_float(int8_t value);
		/// Linear voxel index (CUDA getVoxelIndexUnsigned): z * dim_x * dim_y + y * dim_x + x.
		[[nodiscard]] static uint64_t voxel_index(Dim3ui dimensions, Coord3ui c)
		{
			return (static_cast<uint64_t>(c.z) * dimensions.y + c.y) * dimensions.x + c.x;
		}

	private:
		struct Kernels;
		struct EventSource;
		struct EventParams;

		SensorResult apply_events(const EventSource& source, const EventParams& params, const UpdateOptions& options);
		SensorResult apply_event_chunk(const EventSource& source, const EventParams& params, const UpdateOptions& options);
		SensorResult apply_sequential(const EventSource& source, const EventParams& params, uint32_t max_events);
		uint32_t collide(const DeviceBuffer& other, bool other_is_bit, float threshold, const SensorModel& model);
		Primitives& primitives();

		Context* context_;
		Dim3ui dimensions_;
		float voxel_side_length_;
		DeviceBuffer voxels_;
		DeviceBuffer scratch_counters_;       ///< two words: outside / cut by the robot, or one collision counter
		std::unique_ptr<DeviceBuffer> accumulator_; ///< Accumulate mode: sum of the updates per voxel (zero between calls)
		std::unique_ptr<DeviceBuffer> touched_;     ///< Accumulate mode: one bit per voxel (zero between calls)
		std::unique_ptr<Primitives> primitives_;    ///< created on first Sequential use
		std::shared_ptr<Kernels> kernels_;
	};
}

#endif
