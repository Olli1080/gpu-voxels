// Independent CPU reference of the ProbVoxelMap semantics (slice S3).
//
// Written from the CUDA code (voxel/ProbabilisticVoxel.hpp: updateOccupancy, insert, floatToProbability;
// voxelmap/kernels/VoxelMapOperations.{h,hpp}: RayCaster::rayCast, kernelInsertSensorData, kernelInsertGlobalPointCloud,
// kernelCollideVoxelMapsDebug; voxel/DefaultCollider.hpp) and deliberately shares NO code or data layout with the Vulkan
// implementation: voxels live in a sparse map of ints, updates are generated as an explicit event list and folded on the CPU.
//
// A voxel update ("event") is (voxel index, increment). The three ways the GPU combines the events of one call:
//   apply_sequential   - events in order, saturating after each one (single-threaded CUDA semantics)
//   apply_accumulated  - per voxel the sum of all events, applied once: clamp(old + sum), unknown counts as 0
//   atomic mode        - some serial order of the events; tests use serializations() to enumerate all of them
// Behavioural decisions (see Agent/knowledge_base/architecture/vulkan_prob_voxel_map.md): outside = negative, too large,
// NaN voxel coordinates (as in S1); sensor measurements are dropped when the hit voxel OR the sensor voxel is outside.
#ifndef GVL_TEST_PROB_VOXEL_MAP_REFERENCE_H_INCLUDED
#define GVL_TEST_PROB_VOXEL_MAP_REFERENCE_H_INCLUDED

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

#include "bit_voxel_map_reference.h"

namespace gvl_reference
{
	class ProbVoxelMapReference
	{
	public:
		static constexpr int kUnknown = -128;
		static constexpr int kMin = -127;
		static constexpr int kMax = 127;

		struct Event
		{
			uint64_t voxel;
			int increment;
		};

		struct Generated
		{
			std::vector<Event> events;
			uint32_t outside = 0;
			uint32_t cut_by_robot = 0;
		};

		ProbVoxelMapReference(uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float side)
			: dim_{ dim_x, dim_y, dim_z }, side_(side)
		{
		}

		/// ProbabilisticVoxel::updateOccupancy
		static int apply(int old_value, int increment)
		{
			return old_value == kUnknown ? increment : std::clamp(old_value + increment, kMin, kMax);
		}

		/// ProbabilisticVoxel::floatToProbability (NaN ends up as 1.0)
		static int probability_from_float(float value)
		{
			float tmp = value < 1.0f ? value : 1.0f; // std::min(1, value) keeps 1 for NaN
			tmp = tmp > 0.0f ? tmp : 0.0f;
			tmp = (tmp * (127.0f - (-127.0f))) + (-127.0f);
			return static_cast<int>(tmp); // truncation, like the cast to int8_t
		}

		void clear() { voxels_.clear(); }

		[[nodiscard]] int get(uint64_t index) const
		{
			const auto it = voxels_.find(index);
			return it == voxels_.end() ? kUnknown : it->second;
		}

		[[nodiscard]] uint64_t voxel_count() const { return static_cast<uint64_t>(dim_[0]) * dim_[1] * dim_[2]; }

		void set(uint64_t index, int value) { voxels_[index] = value; }

		// ---- event generation ----

		/// insert_point_cloud: one event per inside point. P needs x, y, z (float).
		template<typename P>
		Generated point_events(const std::vector<P>& points, int increment) const
		{
			Generated out;
			for (const P& p : points)
			{
				uint32_t c[3];
				if (!voxel_of(p.x, p.y, p.z, c))
				{
					++out.outside;
					continue;
				}
				out.events.push_back({ index(c[0], c[1], c[2]), increment });
			}
			return out;
		}

		/// insert_coordinate_list: C needs x, y, z (uint32_t).
		template<typename C>
		Generated coordinate_events(const std::vector<C>& coordinates, int increment) const
		{
			Generated out;
			for (const C& c : coordinates)
			{
				if (c.x >= dim_[0] || c.y >= dim_[1] || c.z >= dim_[2])
				{
					++out.outside;
					continue;
				}
				out.events.push_back({ index(c.x, c.y, c.z), increment });
			}
			return out;
		}

		/// insert_sensor_data. `robot` may be null; the measurement is cut when the robot voxel has bit `robot_bit`.
		template<typename P>
		Generated sensor_events(const std::vector<P>& points, const float origin[3], bool raycast, const BitVoxelMapReference* robot,
			uint8_t robot_bit, int free_increment, int hit_increment) const
		{
			Generated out;
			uint32_t o[3];
			const bool origin_inside = voxel_of(origin[0], origin[1], origin[2], o);
			for (const P& p : points)
			{
				uint32_t hit[3];
				if (!voxel_of(p.x, p.y, p.z, hit) || !origin_inside)
				{
					++out.outside;
					continue;
				}
				if (robot && robot->get(index(hit[0], hit[1], hit[2])).test(robot_bit))
				{
					++out.cut_by_robot;
					continue;
				}
				if (raycast)
					for (const auto& v : walk({ o[0], o[1], o[2] }, { hit[0], hit[1], hit[2] }))
						out.events.push_back({ index(static_cast<uint32_t>(v[0]), static_cast<uint32_t>(v[1]), static_cast<uint32_t>(v[2])), free_increment });
				out.events.push_back({ index(hit[0], hit[1], hit[2]), hit_increment });
			}
			return out;
		}

		/// The voxels visited by RayCaster::rayCast from `from` to `to`, in order (translated from the CUDA code).
		[[nodiscard]] static std::vector<std::array<int64_t, 3>> walk(std::array<uint32_t, 3> from, std::array<uint32_t, 3> to)
		{
			std::array<int64_t, 3> diff{}, step{}, pos{};
			for (int a = 0; a < 3; ++a)
			{
				diff[a] = to[a] > from[a] ? static_cast<int64_t>(to[a]) - from[a] : static_cast<int64_t>(from[a]) - to[a];
				step[a] = to[a] > from[a] ? 1 : -1;
				pos[a] = from[a];
			}
			int64_t remaining = 1 + diff[0] + diff[1] + diff[2];
			int64_t error_xy = diff[0] - diff[1];
			int64_t error_xz = diff[0] - diff[2];
			int64_t error_yz = diff[1] - diff[2];
			const std::array<int64_t, 3> twice{ 2 * diff[0], 2 * diff[1], 2 * diff[2] };
			std::vector<std::array<int64_t, 3>> visited;
			for (; remaining > 0; --remaining)
			{
				visited.push_back(pos);
				if (error_xy > 0 && error_xz > 0)
				{
					pos[0] += step[0];
					error_xy -= twice[1];
					error_xz -= twice[2];
				}
				else if (error_yz > 0)
				{
					pos[1] += step[1];
					error_xy += twice[0];
					error_yz -= twice[2];
				}
				else
				{
					pos[2] += step[2];
					error_xz += twice[0];
					error_yz += twice[1];
				}
			}
			return visited;
		}

		// ---- applying events ----

		void apply_sequential(const std::vector<Event>& events)
		{
			for (const Event& e : events)
				voxels_[e.voxel] = apply(get(e.voxel), e.increment);
		}

		void apply_accumulated(const std::vector<Event>& events)
		{
			std::unordered_map<uint64_t, long long> sums;
			for (const Event& e : events)
				sums[e.voxel] += e.increment;
			for (const auto& [voxel, sum] : sums)
			{
				const long long base = get(voxel) == kUnknown ? 0 : get(voxel);
				voxels_[voxel] = static_cast<int>(std::clamp<long long>(base + sum, kMin, kMax));
			}
		}

		void assign_unknown(const std::vector<Event>& events)
		{
			for (const Event& e : events)
				voxels_[e.voxel] = kUnknown;
		}

		/// All results one voxel can end up with if the increments are applied in some serial order.
		[[nodiscard]] static std::set<int> serializations(int start, std::vector<int> increments)
		{
			std::set<int> results;
			std::sort(increments.begin(), increments.end());
			do
			{
				int value = start;
				for (int inc : increments)
					value = apply(value, inc);
				results.insert(value);
			} while (std::next_permutation(increments.begin(), increments.end()));
			return results;
		}

		// ---- collision ----

		/// Prob x Bit (DefaultCollider::collide(Prob, Bit)): this >= threshold and the bit voxel has a bit except bit 0.
		/// Colliding voxels receive insert(eBVM_COLLISION) = one update with `mark_increment`. \return collisions
		uint32_t collide_with(const BitVoxelMapReference& other, float threshold, int mark_increment)
		{
			const int level = probability_from_float(threshold);
			uint32_t collisions = 0;
			for (uint64_t i = 0; i < voxel_count(); ++i)
				if (get(i) >= level && BitVoxelMapReference::occupied(other.get(i)))
				{
					voxels_[i] = apply(get(i), mark_increment);
					++collisions;
				}
			return collisions;
		}

		/// Prob x Prob: both >= threshold.
		uint32_t collide_with(const ProbVoxelMapReference& other, float threshold, int mark_increment)
		{
			const int level = probability_from_float(threshold);
			uint32_t collisions = 0;
			for (uint64_t i = 0; i < voxel_count(); ++i)
				if (get(i) >= level && other.get(i) >= level)
				{
					voxels_[i] = apply(get(i), mark_increment);
					++collisions;
				}
			return collisions;
		}

		/// Bit x Prob (DefaultCollider::collide(Bit, Prob)): marks the bit map; call on the bit map reference.
		static uint32_t collide_bit_with_prob(BitVoxelMapReference& bits, const ProbVoxelMapReference& prob, float threshold)
		{
			const int level = probability_from_float(threshold);
			uint32_t collisions = 0;
			for (uint64_t i = 0; i < prob.voxel_count(); ++i)
			{
				const auto voxel = bits.get(i);
				if (prob.get(i) >= level && BitVoxelMapReference::occupied(voxel))
				{
					bits.set_bit(i, 2);
					++collisions;
				}
			}
			return collisions;
		}

	private:
		bool voxel_of(float x, float y, float z, uint32_t out[3]) const
		{
			const float f[3] = { std::floor(x / side_), std::floor(y / side_), std::floor(z / side_) };
			for (int a = 0; a < 3; ++a)
				if (!(f[a] >= 0.0f && f[a] < static_cast<float>(dim_[a])))
					return false;
			for (int a = 0; a < 3; ++a)
				out[a] = static_cast<uint32_t>(f[a]);
			return true;
		}

		[[nodiscard]] uint64_t index(uint32_t x, uint32_t y, uint32_t z) const
		{
			return (static_cast<uint64_t>(z) * dim_[1] + y) * dim_[0] + x;
		}

		uint32_t dim_[3];
		float side_;
		std::unordered_map<uint64_t, int> voxels_;
	};
}

#endif
