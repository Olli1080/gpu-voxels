// Host model of the collisions of a BitVoxelList with a bit map or probabilistic map (slice S7-5), shared by the collision tests and the example
// smoke tests. It applies the rules of the CUDA kernels (kernelCollideWithVoxelMap with the DefaultCollider, the types form and the bit mask
// form of VoxelListOperations.hpp) to the downloaded states of the list and the map. The three-dimensional offset (an entry that leaves the map
// does not collide) is the documented difference to CUDA's linear offset.
#ifndef GVL_TEST_LIST_COLLISION_MODEL_H_INCLUDED
#define GVL_TEST_LIST_COLLISION_MODEL_H_INCLUDED

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>

namespace gvl_test::collision_model
{
	using namespace gpu_voxels::vulkan;

	inline BitVector256 bits_of(std::initializer_list<int> bits)
	{
		BitVector256 v{};
		for (const int b : bits)
			v[b / 32] |= 1u << (b % 32);
		return v;
	}

	inline BitVector256 operator|(const BitVector256& a, const BitVector256& b)
	{
		BitVector256 r{};
		for (int w = 0; w < 8; ++w)
			r[w] = a[w] | b[w];
		return r;
	}

	/// "Occupied" for a bit vector: any bit besides bit 0 (BitVector::anyNotEmpty).
	inline bool not_empty(const BitVector256& v)
	{
		if ((v[0] & ~1u) != 0)
			return true;
		for (int w = 1; w < 8; ++w)
			if (v[w] != 0)
				return true;
		return false;
	}

	inline bool shares_bit(const BitVector256& a, const BitVector256& b)
	{
		for (int w = 0; w < 8; ++w)
			if ((a[w] & b[w]) != 0)
				return true;
		return false;
	}

	struct Entry
	{
		Coord3ui c;
		BitVector256 bits;
	};

	/// The two kinds of maps as the model sees them.
	struct MapView
	{
		Dim3ui dims;
		bool probabilistic;
		std::vector<uint32_t> bit_words;       // BitVoxelMap::download_words
		std::vector<int8_t> occupancy;         // ProbVoxelMap::download_occupancy

		[[nodiscard]] uint64_t index(Coord3ui c) const { return (uint64_t(c.z) * dims.y + c.y) * dims.x + c.x; }
		[[nodiscard]] bool bit_occupied(Coord3ui c) const
		{
			BitVector256 v{};
			for (int w = 0; w < 8; ++w)
				v[w] = bit_words[index(c) * 8 + w];
			return not_empty(v);
		}
		[[nodiscard]] int occupancy_at(Coord3ui c) const { return occupancy[index(c)]; }
	};

	inline MapView view_of(const BitVoxelMap& map) { return { map.dimensions(), false, map.download_words(), {} }; }
	inline MapView view_of(const ProbVoxelMap& map) { return { map.dimensions(), true, {}, map.download_occupancy() }; }

	inline std::vector<Entry> entries_of(const BitVoxelList& list)
	{
		const auto coordinates = list.download_coordinates();
		const auto bits = list.download_bit_vectors();
		std::vector<Entry> entries;
		for (std::size_t i = 0; i < coordinates.size(); ++i)
			entries.push_back({ coordinates[i], bits[i] });
		return entries;
	}

	enum class Mode { Default, Types, Mask };

	struct Expected
	{
		uint32_t count = 0;
		BitVector256 types{};
		std::vector<BitVector256> bits_after;
	};

	/// CUDA floatToProbability for the thresholds used in the tests (none lies near a rounding boundary except the exact ones): -127 + p * 254, truncated.
	inline int level_of(float threshold)
	{
		const double p = std::isnan(threshold) ? 1.0 : std::clamp(static_cast<double>(threshold), 0.0, 1.0);
		return static_cast<int>(-127.0 + p * 254.0);
	}

	/// `linear_offset` = CUDA's rule (the offset is added to the linear index: it wraps across rows and slices, and only the ends of the voxel array
	/// are checked); the default is the three-dimensional rule of the Vulkan classes.
	inline Expected model(const std::vector<Entry>& entries, const MapView& map, Mode mode, Offset3i offset, float threshold, const BitVector256& mask,
		bool linear_offset = false)
	{
		Expected e;
		for (const Entry& entry : entries)
		{
			BitVector256 bits = entry.bits;
			int64_t x = int64_t(entry.c.x) + offset.x, y = int64_t(entry.c.y) + offset.y, z = int64_t(entry.c.z) + offset.z;
			bool inside = x >= 0 && y >= 0 && z >= 0 && x < map.dims.x && y < map.dims.y && z < map.dims.z;
			if (linear_offset)
			{
				const int64_t index = int64_t(map.index(entry.c)) + (int64_t(offset.z) * map.dims.y + offset.y) * map.dims.x + offset.x;
				inside = index >= 0 && index < int64_t(map.dims.voxel_count());
				x = index % map.dims.x;
				y = (index / map.dims.x) % map.dims.y;
				z = index / map.dims.x / map.dims.y;
			}
			bool collides = false;
			if (inside)
			{
				const Coord3ui c{ static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z) };
				bool map_occupied;
				if (!map.probabilistic)
					map_occupied = map.bit_occupied(c);
				else if (mode == Mode::Default)
					map_occupied = map.occupancy_at(c) >= level_of(threshold);
				else
					map_occupied = static_cast<float>(map.occupancy_at(c)) >= threshold;
				collides = map_occupied;
				if (mode == Mode::Default)
					collides = collides && not_empty(entry.bits);
				if (mode == Mode::Mask)
					collides = collides && shares_bit(mask, entry.bits);
			}
			if (collides)
			{
				++e.count;
				if (mode == Mode::Types)
					e.types = e.types | entry.bits;
				bits[0] |= 1u << 2; // eBVM_COLLISION
			}
			e.bits_after.push_back(bits);
		}
		return e;
	}
}

#endif
