// Independent CPU reference of the DistanceVoxelMap semantics (slice S5).
//
// Written from the CUDA code (voxel/DistanceVoxel.{h,hpp}: insert, squaredObstacleDistance, extract_byte_distance;
// voxelmap/src/DistanceVoxelMap.cu: mergeOccupied, getObstacleDistance; voxelmap/src/VoxelMapOperations.cu: obstacle
// insertion) and shares NO algorithm with the Vulkan implementation: the distance field is a plain brute force over the
// list of obstacles (what CUDA's exactDistances3D does), O(voxels x obstacles).
//
// Behavioural decisions (see Agent/knowledge_base/architecture/vulkan_distance_map.md): outside = negative, too large, NaN
// voxel coordinates (as in S1); the stored nearest obstacle is only specified up to ties, so tests check that it IS an
// obstacle and that its distance is the minimum.
#ifndef GVL_TEST_DISTANCE_MAP_REFERENCE_H_INCLUDED
#define GVL_TEST_DISTANCE_MAP_REFERENCE_H_INCLUDED

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

namespace gvl_reference
{
	class DistanceMapReference
	{
	public:
		struct Coord { uint32_t x, y, z; };

		DistanceMapReference(uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float side)
			: dim_{ dim_x, dim_y, dim_z }, side_(side)
		{
		}

		void clear()
		{
			obstacles_.clear();
			dirty_ = true;
		}

		/// P needs x, y, z (float). \return number of points outside the map
		template<typename P>
		uint32_t insert_points(const std::vector<P>& points)
		{
			uint32_t outside = 0;
			for (const P& p : points)
			{
				const float f[3] = { std::floor(p.x / side_), std::floor(p.y / side_), std::floor(p.z / side_) };
				bool inside = true;
				for (int a = 0; a < 3; ++a)
					inside = inside && f[a] >= 0.0f && f[a] < static_cast<float>(dim_[a]);
				if (!inside)
				{
					++outside;
					continue;
				}
				obstacles_.insert(index(static_cast<uint32_t>(f[0]), static_cast<uint32_t>(f[1]), static_cast<uint32_t>(f[2])));
				dirty_ = true;
			}
			return outside;
		}

		/// C needs x, y, z (uint32_t). \return number of coordinates outside the map
		template<typename C>
		uint32_t insert_coordinates(const std::vector<C>& coordinates)
		{
			uint32_t outside = 0;
			for (const C& c : coordinates)
			{
				if (c.x >= dim_[0] || c.y >= dim_[1] || c.z >= dim_[2])
				{
					++outside;
					continue;
				}
				obstacles_.insert(index(c.x, c.y, c.z));
				dirty_ = true;
			}
			return outside;
		}

		void insert_obstacle(uint32_t x, uint32_t y, uint32_t z)
		{
			obstacles_.insert(index(x, y, z));
			dirty_ = true;
		}

		[[nodiscard]] bool is_obstacle(uint32_t x, uint32_t y, uint32_t z) const
		{
			return x < dim_[0] && y < dim_[1] && z < dim_[2] && obstacles_.count(index(x, y, z)) != 0;
		}

		[[nodiscard]] std::size_t obstacle_count() const { return obstacles_.size(); }

		/// Squared distance of voxel (x, y, z) to the nearest obstacle by brute force, -1 without obstacles.
		[[nodiscard]] int64_t min_squared_distance(uint32_t x, uint32_t y, uint32_t z) const
		{
			if (dirty_)
				rebuild();
			int64_t best = -1;
			for (const Coord& o : flat_)
			{
				const int64_t dx = static_cast<int64_t>(x) - o.x, dy = static_cast<int64_t>(y) - o.y, dz = static_cast<int64_t>(z) - o.z;
				const int64_t d = dx * dx + dy * dy + dz * dz;
				if (best < 0 || d < best)
					best = d;
			}
			return best;
		}

		/// CUDA extract_byte_distance with the brute force distance instead of the stored obstacle.
		[[nodiscard]] int extract(uint32_t x, uint32_t y, uint32_t z, int radius) const
		{
			const int64_t pos[3] = { x, y, z };
			for (int a = 0; a < 3; ++a)
				if (pos[a] > static_cast<int64_t>(dim_[a]) - radius)
					return static_cast<int>(std::max<int64_t>(static_cast<int64_t>(dim_[a]) - pos[a], 0));
			for (int a = 0; a < 3; ++a)
				if (pos[a] < radius)
					return static_cast<int>(std::max<int64_t>(pos[a], 0));
			const int64_t d = min_squared_distance(x, y, z);
			if (d < 0)
				return 127; // uninitialised voxel: squared distance INT_MAX
			return static_cast<int>(std::min<int64_t>(static_cast<int64_t>(std::floor(std::sqrt(static_cast<double>(d)))), 127));
		}

		[[nodiscard]] uint64_t voxel_count() const { return static_cast<uint64_t>(dim_[0]) * dim_[1] * dim_[2]; }
		[[nodiscard]] uint32_t dim(int axis) const { return dim_[axis]; }

	private:
		[[nodiscard]] uint64_t index(uint32_t x, uint32_t y, uint32_t z) const
		{
			return (static_cast<uint64_t>(z) * dim_[1] + y) * dim_[0] + x;
		}

		void rebuild() const
		{
			dirty_ = false;
			flat_.clear();
			for (uint64_t id : obstacles_)
				flat_.push_back({ static_cast<uint32_t>(id % dim_[0]), static_cast<uint32_t>((id / dim_[0]) % dim_[1]),
					static_cast<uint32_t>(id / dim_[0] / dim_[1]) });
		}

		uint32_t dim_[3];
		float side_;
		std::set<uint64_t> obstacles_;
		mutable std::vector<Coord> flat_;
		mutable bool dirty_ = true;
	};
}

#endif
