// Independent CPU reference of the BitVoxelMap semantics (slice S1).
//
// Written from the CUDA code (voxelmap/kernels/VoxelMapOperations.hpp: kernelInsertGlobalPointCloud,
// kernelInsertCoordinateTuples, kernelClearVoxelMap, kernelCollideVoxelMapsDebug; voxel/DefaultCollider.hpp;
// helpers/BitVector.cuhpp: noneButEmpty) and deliberately shares NO code or data layout with the Vulkan
// implementation: voxels are std::bitset<256> in a sparse map, not 32-bit words in a dense buffer.
//
// Behavioural decisions (see Agent/knowledge_base/architecture/vulkan_bit_voxel_map.md):
//  * voxel = floor(p / side) evaluated in single precision, exactly like mapToVoxels();
//  * points with a negative, too large or NaN coordinate are OUTSIDE (CUDA would saturate negatives to voxel 0);
//  * a voxel is occupied when any bit except bit 0 (eBVM_FREE) is set; collision = both occupied;
//  * colliding voxels of the first map receive bit 2 (eBVM_COLLISION).
#ifndef GVL_TEST_BIT_VOXEL_MAP_REFERENCE_H_INCLUDED
#define GVL_TEST_BIT_VOXEL_MAP_REFERENCE_H_INCLUDED

#include <bitset>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace gvl_reference
{
	class BitVoxelMapReference
	{
	public:
		using Voxel = std::bitset<256>;

		BitVoxelMapReference(uint32_t dim_x, uint32_t dim_y, uint32_t dim_z, float side)
			: dim_{ dim_x, dim_y, dim_z }, side_(side)
		{
		}

		void clear() { voxels_.clear(); }

		void clear_meaning(uint8_t meaning)
		{
			for (auto it = voxels_.begin(); it != voxels_.end();)
			{
				it->second.reset(meaning);
				it = it->second.none() ? voxels_.erase(it) : std::next(it);
			}
		}

		/// P needs members x, y, z (float). \return number of points outside the map
		template<typename P>
		uint32_t insert_points(const std::vector<P>& points, uint8_t meaning)
		{
			uint32_t outside = 0;
			for (const P& p : points)
			{
				const float fx = std::floor(p.x / side_);
				const float fy = std::floor(p.y / side_);
				const float fz = std::floor(p.z / side_);
				const bool inside = fx >= 0.0f && fx < static_cast<float>(dim_[0]) && fy >= 0.0f &&
					fy < static_cast<float>(dim_[1]) && fz >= 0.0f && fz < static_cast<float>(dim_[2]);
				if (!inside)
				{
					++outside;
					continue;
				}
				voxels_[index(static_cast<uint32_t>(fx), static_cast<uint32_t>(fy), static_cast<uint32_t>(fz))].set(meaning);
			}
			return outside;
		}

		/// C needs members x, y, z (uint32_t). \return number of coordinates outside the map
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
				voxels_[index(c.x, c.y, c.z)].set(meaning);
			}
			return outside;
		}

		/// Sub cloud k is inserted with meanings[k]. \return number of points outside the map
		template<typename P>
		uint32_t insert_meta_clouds(const std::vector<std::vector<P>>& clouds, const std::vector<uint8_t>& meanings)
		{
			uint32_t outside = 0;
			for (std::size_t k = 0; k < clouds.size(); ++k)
				outside += insert_points(clouds[k], meanings[k]);
			return outside;
		}

		struct SelfCollision
		{
			bool colliding = false;
			Voxel meanings;
			uint32_t outside = 0;
		};

		/// Sequential reference of insertMetaPointCloudWithSelfCollisionCheck (CUDA kernelInsertMetaPointCloudSelfCollCheck):
		/// sub clouds in order, points in order; masked = voxel & mask[k]; if masked has a bit except bit 0:
		/// colliding |= masked, colliding.set(meaning[k]), voxel.set(2); then voxel.set(meaning[k]).
		/// Meanings/masks empty => defaults (4 + k; everything except bit 2 and the own bit).
		template<typename P>
		SelfCollision insert_meta_clouds_self_collision(const std::vector<std::vector<P>>& clouds,
			std::vector<uint8_t> meanings = {}, std::vector<Voxel> masks = {})
		{
			if (meanings.empty())
				for (std::size_t k = 0; k < clouds.size(); ++k)
					meanings.push_back(static_cast<uint8_t>(4 + k));
			if (masks.empty())
				for (std::size_t k = 0; k < clouds.size(); ++k)
				{
					Voxel mask;
					mask.set();
					mask.reset(2);
					mask.reset(meanings[k]);
					masks.push_back(mask);
				}
			SelfCollision result;
			for (std::size_t k = 0; k < clouds.size(); ++k)
			{
				for (const P& p : clouds[k])
				{
					const float fx = std::floor(p.x / side_);
					const float fy = std::floor(p.y / side_);
					const float fz = std::floor(p.z / side_);
					const bool inside = fx >= 0.0f && fx < static_cast<float>(dim_[0]) && fy >= 0.0f &&
						fy < static_cast<float>(dim_[1]) && fz >= 0.0f && fz < static_cast<float>(dim_[2]);
					if (!inside)
					{
						++result.outside;
						continue;
					}
					Voxel& voxel = voxels_[index(static_cast<uint32_t>(fx), static_cast<uint32_t>(fy), static_cast<uint32_t>(fz))];
					const Voxel masked = voxel & masks[k];
					if (occupied(masked))
					{
						result.meanings |= masked;
						result.meanings.set(meanings[k]);
						voxel.set(2);
						result.colliding = true;
					}
					voxel.set(meanings[k]);
				}
			}
			return result;
		}

		/// \return number of colliding voxels; sets bit 2 in the colliding voxels of *this*
		uint32_t collide_with(const BitVoxelMapReference& other)
		{
			uint32_t collisions = 0;
			for (auto& [idx, voxel] : voxels_)
			{
				const auto it = other.voxels_.find(idx);
				if (it != other.voxels_.end() && occupied(voxel) && occupied(it->second))
				{
					voxel.set(2);
					++collisions;
				}
			}
			return collisions;
		}

		struct TypeCollision
		{
			uint32_t count = 0;
			Voxel types;
		};

		/// CUDA collideWithTypes (SVCollider, window 0): the colliding bits of two voxels are (a & b) without bits 0..3;
		/// the voxels collide if that is non-zero. Colliding voxels of *this* get bit 2; `types` is the OR of all colliding bits.
		TypeCollision collide_with_types(const BitVoxelMapReference& other)
		{
			TypeCollision result;
			const Voxel not_low_bits = ~Voxel(0xF);
			for (auto& [idx, voxel] : voxels_)
			{
				const auto it = other.voxels_.find(idx);
				if (it == other.voxels_.end())
					continue;
				const Voxel shared = voxel & it->second & not_low_bits;
				if (shared.any())
				{
					voxel.set(2);
					++result.count;
					result.types |= shared;
				}
			}
			return result;
		}

		/// Sets one bit of one voxel.
		void set_bit(uint64_t voxel_index, uint8_t bit) { voxels_[voxel_index].set(bit); }

		[[nodiscard]] Voxel get(uint64_t voxel_index) const
		{
			const auto it = voxels_.find(voxel_index);
			return it == voxels_.end() ? Voxel() : it->second;
		}

		[[nodiscard]] const std::unordered_map<uint64_t, Voxel>& voxels() const { return voxels_; }
		[[nodiscard]] uint64_t voxel_count() const { return static_cast<uint64_t>(dim_[0]) * dim_[1] * dim_[2]; }

		/// 32-bit words of `count` voxels from `first` (word w holds bits 32w..32w+31 of the voxel).
		[[nodiscard]] std::vector<uint32_t> to_words(uint64_t first, uint64_t count) const
		{
			std::vector<uint32_t> words(count * 8, 0);
			for (const auto& [idx, voxel] : voxels_)
			{
				if (idx < first || idx >= first + count)
					continue;
				for (uint32_t bit = 0; bit < 256; ++bit)
					if (voxel.test(bit))
						words[(idx - first) * 8 + bit / 32] |= 1u << (bit % 32);
			}
			return words;
		}

		[[nodiscard]] static bool occupied(const Voxel& v) { return (v >> 1).any(); }

	private:
		[[nodiscard]] uint64_t index(uint32_t x, uint32_t y, uint32_t z) const
		{
			return (static_cast<uint64_t>(z) * dim_[1] + y) * dim_[0] + x;
		}

		uint32_t dim_[3];
		float side_;
		std::unordered_map<uint64_t, Voxel> voxels_;
	};
}

#endif
