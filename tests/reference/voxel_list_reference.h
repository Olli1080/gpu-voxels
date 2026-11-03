// Independent CPU reference of the voxel list semantics (slice S4).
//
// Written from the CUDA code (voxellist/TemplateVoxelList.hpp: insert*, make_unique, remove_out_of_bounds, subtract,
// collideVoxellists; voxellist/CountingVoxelList.cu: collideWith, remove_underpopulated; voxellist/kernels/
// VoxelListOperations.hpp: kernelCollideWithVoxelMap; voxelmap/TemplateVoxelMap.hpp: merge) and deliberately shares NO
// code or data layout with the Vulkan implementation: entries live in a std::map ordered by voxel id, bit vectors are
// std::bitset<256>, counts are int8_t that wrap through ordinary integer arithmetic.
//
// Behavioural decisions (see Agent/knowledge_base/architecture/vulkan_voxel_lists.md):
//  * voxel = floor(p / side) in single precision; negative, too large and NaN coordinates are OUTSIDE and dropped
//    (CUDA would saturate negative values to voxel 0, see known issue K21);
//  * entries with equal ids are merged: bit vectors OR-ed, counts added with int8 wrap-around;
//  * counting entries start with count 1 per inserted point / coordinate, the meaning is ignored;
//  * list vs map collision (DefaultCollider): the MAP voxel and the LIST voxel must both be occupied (any bit except
//    bit 0); the list voxel is marked with bit 2. S4 had this wrong (only the map voxel); corrected in S7-5 after reading
//    TemplateVoxelList::collisionCheckWithCollider / kernelCollideWithVoxelMap(Collider) / DefaultCollider::collide.
//    List vs list collision counts entries whose id occurs in the other list.
#ifndef GVL_TEST_VOXEL_LIST_REFERENCE_H_INCLUDED
#define GVL_TEST_VOXEL_LIST_REFERENCE_H_INCLUDED

#include <bitset>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include "bit_voxel_map_reference.h"

namespace gvl_reference
{
	class VoxelListReference
	{
	public:
		struct Entry
		{
			uint32_t x = 0, y = 0, z = 0;
			std::bitset<256> bits;  // bit lists
			int8_t count = 0;       // counting lists
		};

		VoxelListReference(uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float side, bool counting)
			: dim_{ dim_x, dim_y, dim_z }, side_(side), counting_(counting)
		{
		}

		void clear() { entries_.clear(); }

		/// P needs members x, y, z (float). \return number of points outside the reference map
		template<typename P>
		uint32_t insert_points(const std::vector<P>& points, uint8_t meaning)
		{
			uint32_t outside = 0;
			for (const P& p : points)
				outside += insert_point(p.x, p.y, p.z, meaning) ? 0 : 1;
			return outside;
		}

		/// C needs members x, y, z (uint32_t). \return number of coordinates outside the reference map
		template<typename C>
		uint32_t insert_coordinates(const std::vector<C>& coordinates, uint8_t meaning)
		{
			uint32_t outside = 0;
			for (const C& c : coordinates)
			{
				if (c.x >= dim_[0] || c.y >= dim_[1] || c.z >= dim_[2])
				{
					++outside;
					continue;
				}
				add(c.x, c.y, c.z, meaning);
			}
			return outside;
		}

		/// Sub cloud k is inserted with meanings[k]. \return number of points outside the reference map
		template<typename P>
		uint32_t insert_meta_clouds(const std::vector<std::vector<P>>& clouds, const std::vector<uint8_t>& meanings)
		{
			uint32_t outside = 0;
			for (std::size_t k = 0; k < clouds.size(); ++k)
				outside += insert_points(clouds[k], meanings[k]);
			return outside;
		}

		/// Entries of this list whose id occurs in `other`; with a threshold only counting entries with count >= threshold.
		[[nodiscard]] uint32_t collide_with(const VoxelListReference& other, std::optional<int> threshold = std::nullopt) const
		{
			uint32_t collisions = 0;
			for (const auto& [id, entry] : entries_)
				if (other.entries_.count(id) != 0 && (!threshold || entry.count >= *threshold))
					++collisions;
			return collisions;
		}

		/// Bit lists: entries whose own voxel and map voxel are occupied get bit 2. \return number of such entries
		uint32_t collide_with_map(const BitVoxelMapReference& map)
		{
			uint32_t collisions = 0;
			for (auto& [id, entry] : entries_)
				if (BitVoxelMapReference::occupied(entry.bits) && BitVoxelMapReference::occupied(map.get(id)))
				{
					entry.bits.set(2);
					++collisions;
				}
			return collisions;
		}

		void subtract(const VoxelListReference& other)
		{
			for (auto it = entries_.begin(); it != entries_.end();)
				it = other.entries_.count(it->first) != 0 ? entries_.erase(it) : std::next(it);
		}

		void remove_underpopulated(int threshold)
		{
			for (auto it = entries_.begin(); it != entries_.end();)
				it = it->second.count < threshold ? entries_.erase(it) : std::next(it);
		}

		/// Bit lists: adds every occupied voxel of the map (bits OR-ed into existing entries). \return number of voxels
		uint32_t insert_occupied_voxels(const BitVoxelMapReference& map)
		{
			uint32_t added = 0;
			for (const auto& [id, voxel] : map.voxels())
			{
				if (!BitVoxelMapReference::occupied(voxel))
					continue;
				Entry& entry = entry_for(static_cast<uint32_t>(id));
				entry.bits |= voxel;
				++added;
			}
			return added;
		}

		[[nodiscard]] const std::map<uint32_t, Entry>& entries() const { return entries_; }
		[[nodiscard]] std::size_t size() const { return entries_.size(); }

		[[nodiscard]] std::vector<uint32_t> ids() const
		{
			std::vector<uint32_t> out;
			for (const auto& [id, entry] : entries_)
				out.push_back(id);
			return out;
		}

		/// x, y, z of every entry, three words each.
		[[nodiscard]] std::vector<uint32_t> coordinate_words() const
		{
			std::vector<uint32_t> out;
			for (const auto& [id, entry] : entries_)
				out.insert(out.end(), { entry.x, entry.y, entry.z });
			return out;
		}

		/// Bit lists: 8 words per entry (word w holds bits 32w..32w+31); counting lists: the count sign extended.
		[[nodiscard]] std::vector<uint32_t> voxel_words() const
		{
			std::vector<uint32_t> out;
			for (const auto& [id, entry] : entries_)
			{
				if (counting_)
				{
					out.push_back(static_cast<uint32_t>(static_cast<int32_t>(entry.count)));
					continue;
				}
				for (uint32_t w = 0; w < 8; ++w)
				{
					uint32_t word = 0;
					for (uint32_t b = 0; b < 32; ++b)
						if (entry.bits.test(w * 32 + b))
							word |= 1u << b;
					out.push_back(word);
				}
			}
			return out;
		}

	private:
		bool insert_point(float px, float py, float pz, uint8_t meaning)
		{
			const float fx = std::floor(px / side_);
			const float fy = std::floor(py / side_);
			const float fz = std::floor(pz / side_);
			const bool inside = fx >= 0.0f && fx < static_cast<float>(dim_[0]) && fy >= 0.0f &&
				fy < static_cast<float>(dim_[1]) && fz >= 0.0f && fz < static_cast<float>(dim_[2]);
			if (inside)
				add(static_cast<uint32_t>(fx), static_cast<uint32_t>(fy), static_cast<uint32_t>(fz), meaning);
			return inside;
		}

		void add(uint32_t x, uint32_t y, uint32_t z, uint8_t meaning)
		{
			Entry& entry = entry_for(static_cast<uint32_t>((static_cast<uint64_t>(z) * dim_[1] + y) * dim_[0] + x));
			if (counting_)
				entry.count = static_cast<int8_t>(entry.count + 1);
			else
				entry.bits.set(meaning);
		}

		Entry& entry_for(uint32_t id)
		{
			auto [it, inserted] = entries_.try_emplace(id);
			if (inserted)
			{
				it->second.x = id % dim_[0];
				it->second.y = (id / dim_[0]) % dim_[1];
				it->second.z = id / dim_[0] / dim_[1];
			}
			return it->second;
		}

		uint32_t dim_[3];
		float side_;
		bool counting_;
		std::map<uint32_t, Entry> entries_;
	};
}

#endif
