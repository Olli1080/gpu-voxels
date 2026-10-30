// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Morton (Z-order) codes of the octree, host side. Mirrors shaders/morton.glsl bit for bit.
 *
 * Layout (same as the CUDA octree, `octree/Morton.h`): per triple of bits x is bit 0, y bit 1, z bit 2; a 60 bit code
 * holds 20 bits per axis; coordinates are masked to 20 bits (CUDA leaves larger values undefined). The octree itself uses
 * coordinates < 2^14 (levels 0..14, 42 bit ids). The id of a node at `level` is the Morton prefix `id >> 3*level`;
 * the index of the ancestor at `level` among its siblings is `(id >> 3*level) & 7`.
 */
#ifndef GPU_VOXELS_VULKAN_MORTON_H_INCLUDED
#define GPU_VOXELS_VULKAN_MORTON_H_INCLUDED

#include <bit>
#include <cstdint>

namespace gpu_voxels::vulkan::morton
{
	/// Bits per axis of a 60 bit code.
	constexpr uint32_t kBitsPerAxis = 20;

	/// Spreads the low 10 bits of x to every third bit (bit i -> bit 3i).
	constexpr uint32_t spread10(uint32_t x)
	{
		x &= 1023u;
		x = (x | (x << 16)) & 0x030000FFu;
		x = (x | (x << 8)) & 0x0300F00Fu;
		x = (x | (x << 4)) & 0x030C30C3u;
		x = (x | (x << 2)) & 0x09249249u;
		return x;
	}

	/// Inverse of spread10: collects every third bit (bit 3i -> bit i).
	constexpr uint32_t compact10(uint32_t x)
	{
		x &= 0x09249249u;
		x = (x ^ (x >> 2)) & 0x030C30C3u;
		x = (x ^ (x >> 4)) & 0x0300F00Fu;
		x = (x ^ (x >> 8)) & 0xFF0000FFu;
		x = (x ^ (x >> 16)) & 0x000003FFu;
		return x;
	}

	/// 30 bit code of three 10 bit coordinates.
	constexpr uint32_t code30(uint32_t x, uint32_t y, uint32_t z) { return spread10(x) | (spread10(y) << 1) | (spread10(z) << 2); }

	/// 60 bit code of three 20 bit coordinates (higher bits are ignored: spread10 uses 10 bits only).
	constexpr uint64_t code60(uint32_t x, uint32_t y, uint32_t z)
	{
		return (uint64_t(code30(x >> 10, y >> 10, z >> 10)) << 30) | uint64_t(code30(x, y, z));
	}

	struct Coords { uint32_t x, y, z; };

	/// Inverse of code60 (bits above 60 are ignored).
	constexpr Coords decode60(uint64_t code)
	{
		const uint32_t lo = uint32_t(code);      // compact10 only looks at bits 0..29: higher bits are ignored
		const uint32_t hi = uint32_t(code >> 30);
		return { (compact10(hi) << 10) | compact10(lo), (compact10(hi >> 1) << 10) | compact10(lo >> 1),
			(compact10(hi >> 2) << 10) | compact10(lo >> 2) };
	}

	/// Node id of the ancestor of voxel `id` at `level + 1` (CUDA getZOrderPrefix); the ancestor at `level` has id `id >> 3*level`.
	constexpr uint64_t z_order_prefix(uint64_t id, uint32_t level) { return id >> (3u * (level + 1u)); }

	/// Index (0..7) of the ancestor of voxel `id` at `level` among its siblings (CUDA getZOrderNodeId).
	constexpr uint32_t z_order_node_id(uint64_t id, uint32_t level) { return uint32_t((id >> (3u * level)) & 7u); }

	/// Number of levels one has to climb until the paths of two voxels meet (0 when equal); CUDA getCommonLevel (device
	/// version, count of leading zeros) without its host/device inconsistency.
	constexpr uint32_t common_level(uint64_t a, uint64_t b) { return (uint32_t(std::bit_width(a ^ b)) + 2u) / 3u; }
}

#endif
