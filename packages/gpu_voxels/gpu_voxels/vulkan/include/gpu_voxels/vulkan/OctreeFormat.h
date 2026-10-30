// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Host side view of the octree node memory formats; mirrors shaders/octree_nodes.glsl.
 *
 * A level is one chunk of blocks of 8 sibling nodes. Inner nodes (levels 1..14) hold status, flags, (probabilistic: occupancy)
 * and the 48-bit device address of their child block (0 = none); leaves (level 0) hold the status (and occupancy).
 * Child addresses: low 48 bits stored, sign extended on read (Intel returns canonical addresses with the upper 16 bits set).
 *   deterministic inner node, 8 bytes:   word0 = status | flags << 8 | (child & 0xFFFF) << 16, word1 = child >> 16
 *   probabilistic inner node, 12 bytes:  word0 = status | flags << 8 | occupancy << 16, word1 / word2 = child (low 32 / high 16 bit)
 *   deterministic leaf 1 byte (status), probabilistic leaf 2 bytes (status | occupancy << 8)
 */
#ifndef GPU_VOXELS_VULKAN_OCTREE_FORMAT_H_INCLUDED
#define GPU_VOXELS_VULKAN_OCTREE_FORMAT_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace gpu_voxels::vulkan
{
	enum class OctreeKind : uint32_t
	{
		Deterministic = 0, ///< status only (CUDA Environment::InnerNode / LeafNode)
		Probabilistic = 1  ///< status + int8 log-odds occupancy (CUDA InnerNodeProb / LeafNodeProb)
	};
}

namespace gpu_voxels::vulkan::octree_format
{
	/// Node status bits (CUDA Nodes.h).
	constexpr uint8_t kFree = 1, kUnknown = 2, kOccupied = 4, kPart = 8, kLastLevel = 16, kCollision = 32, kStatic = 64, kDynamic = 128;
	constexpr uint8_t kOccupancyMask = kFree | kUnknown | kOccupied;
	constexpr int8_t kProbUnknown = -128;
	constexpr int8_t kProbMax = 127;

	constexpr uint32_t kRootLevel = 14;
	constexpr uint32_t kLevelCount = 15;

	/// Child pointers keep the low 48 bits of a device address and are sign extended from bit 47 when read ("canonical" addresses).
	constexpr uint64_t sign_extend_48(uint64_t raw) { return uint64_t((int64_t(raw << 16)) >> 16); }
	/// True if a buffer of `size` bytes starting at `address` can be addressed by 48 bit node pointers.
	constexpr bool addressable(uint64_t address, uint64_t size)
	{
		const uint64_t last = address + size - 1;
		return last >= address && sign_extend_48(address) == address && sign_extend_48(last) == last;
	}

	struct HostNode
	{
		uint8_t status = 0;
		uint8_t flags = 0;
		int8_t occupancy = 0;
		uint64_t child = 0;
	};

	constexpr size_t inner_block_bytes(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? 96 : 64; }
	constexpr size_t leaf_block_bytes(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? 16 : 8; }
	constexpr size_t block_bytes(OctreeKind kind, uint32_t level) { return level == 0 ? leaf_block_bytes(kind) : inner_block_bytes(kind); }

	inline uint32_t word(const uint8_t* block, size_t index)
	{
		uint32_t w;
		std::memcpy(&w, block + 4 * index, 4);
		return w;
	}

	inline void set_word(uint8_t* block, size_t index, uint32_t value) { std::memcpy(block + 4 * index, &value, 4); }

	inline HostNode decode_inner(OctreeKind kind, const uint8_t* block, uint32_t slot)
	{
		HostNode n;
		if (kind == OctreeKind::Probabilistic)
		{
			const uint32_t w0 = word(block, 3 * slot);
			n.status = uint8_t(w0);
			n.flags = uint8_t(w0 >> 8);
			n.occupancy = int8_t(uint8_t(w0 >> 16));
			n.child = sign_extend_48(uint64_t(word(block, 3 * slot + 1)) | (uint64_t(word(block, 3 * slot + 2)) << 32));
		}
		else
		{
			const uint32_t w0 = word(block, 2 * slot);
			n.status = uint8_t(w0);
			n.flags = uint8_t(w0 >> 8);
			n.child = sign_extend_48(uint64_t(w0 >> 16) | (uint64_t(word(block, 2 * slot + 1)) << 16));
		}
		return n;
	}

	inline void encode_inner(OctreeKind kind, uint8_t* block, uint32_t slot, const HostNode& n)
	{
		if (kind == OctreeKind::Probabilistic)
		{
			set_word(block, 3 * slot, uint32_t(n.status) | (uint32_t(n.flags) << 8) | (uint32_t(uint8_t(n.occupancy)) << 16));
			set_word(block, 3 * slot + 1, uint32_t(n.child));
			set_word(block, 3 * slot + 2, uint32_t((n.child >> 32) & 0xFFFF));
		}
		else
		{
			set_word(block, 2 * slot, uint32_t(n.status) | (uint32_t(n.flags) << 8) | (uint32_t(n.child) << 16));
			set_word(block, 2 * slot + 1, uint32_t(n.child >> 16));
		}
	}

	inline HostNode decode_leaf(OctreeKind kind, const uint8_t* block, uint32_t slot)
	{
		HostNode n;
		if (kind == OctreeKind::Probabilistic)
		{
			const uint32_t v = (word(block, slot >> 1) >> ((slot & 1) * 16)) & 0xFFFF;
			n.status = uint8_t(v);
			n.occupancy = int8_t(uint8_t(v >> 8));
		}
		else
			n.status = uint8_t(word(block, slot >> 2) >> ((slot & 3) * 8));
		return n;
	}

	/// Writes leaf `slot` of a leaf block (the other leaves of the block are kept).
	inline void encode_leaf(OctreeKind kind, uint8_t* block, uint32_t slot, const HostNode& n)
	{
		if (kind == OctreeKind::Probabilistic)
		{
			const uint32_t shift = (slot & 1) * 16;
			uint32_t w = word(block, slot >> 1) & ~(0xFFFFu << shift);
			set_word(block, slot >> 1, w | ((uint32_t(n.status) | (uint32_t(uint8_t(n.occupancy)) << 8)) << shift));
		}
		else
		{
			const uint32_t shift = (slot & 3) * 8;
			uint32_t w = word(block, slot >> 2) & ~(0xFFu << shift);
			set_word(block, slot >> 2, w | (uint32_t(n.status) << shift));
		}
	}

	/// Node without children and without occupied voxels, as left by build + propagate.
	inline HostNode filler(OctreeKind kind, uint32_t level)
	{
		HostNode n;
		n.status = uint8_t(kUnknown | kStatic | (level == 1 ? kLastLevel : 0));
		n.occupancy = kind == OctreeKind::Probabilistic ? kProbUnknown : int8_t(0);
		return n;
	}

	/// Status / occupancy of an inner node of `level` computed from its 8 children (CUDA getNewStatus, probabilistic: max
	/// occupancy and the PART rule). The child pointer is not set.
	inline HostNode parent_of(OctreeKind kind, const HostNode (&kids)[8], uint32_t level)
	{
		HostNode p;
		if (kind == OctreeKind::Probabilistic)
		{
			int pmax = kids[0].occupancy, pmin = kids[0].occupancy;
			bool all_non_part = true;
			for (const HostNode& k : kids)
			{
				pmax = pmax > k.occupancy ? pmax : k.occupancy;
				pmin = pmin < k.occupancy ? pmin : k.occupancy;
				all_non_part = all_non_part && !(k.status & kPart);
			}
			p.occupancy = int8_t(pmax);
			p.status = kUnknown | kStatic;
			if (!(pmax == pmin && all_non_part))
				p.status |= kPart;
		}
		else
		{
			uint8_t o = 0, a = 0xFF, sd = 0;
			for (const HostNode& k : kids)
			{
				o |= k.status & kOccupancyMask;
				a &= k.status & kOccupancyMask;
				sd |= k.status;
			}
			const bool single_bit = a == kFree || a == kUnknown || a == kOccupied;
			p.status = (single_bit && o == a) ? a : uint8_t(o | kPart);
			p.status |= sd & (kStatic | kDynamic);
		}
		if (level == 1)
			p.status |= kLastLevel;
		return p;
	}
}

#endif
