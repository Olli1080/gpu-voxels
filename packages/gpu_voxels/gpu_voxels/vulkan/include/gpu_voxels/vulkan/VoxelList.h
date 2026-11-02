// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Sparse voxel lists on the Vulkan backend (slice S4 of the migration plan).
 *
 * A list holds entries (voxel id, voxel coordinates, voxel payload) sorted by id and unique in the id, like the CUDA
 * classes voxellist::BitVoxelList / CountingVoxelList (TemplateVoxelList). The id is the linear voxel index of the
 * reference map (z * dx * dy + y * dx + x). Every insertion appends the new entries, then drops entries outside the
 * reference map (remove_out_of_bounds) and merges entries with equal ids (make_unique): bit vectors are OR-ed, counting
 * voxels are added (wrapping like int8_t).
 *
 * Documented deviations from CUDA are listed in Agent/knowledge_base/architecture/vulkan_voxel_lists.md.
 */
#ifndef GPU_VOXELS_VULKAN_VOXEL_LIST_H_INCLUDED
#define GPU_VOXELS_VULKAN_VOXEL_LIST_H_INCLUDED

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/Primitives.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	class BitVoxelMap;
	class CountingVoxelList;

	/// Behaviour shared by all list types. Not instantiable; use BitVoxelList or CountingVoxelList.
	class VoxelListBase
	{
	public:
		/// Voxel payload of a list.
		enum class Kind : uint32_t
		{
			Bit = 0,      ///< BitVoxel<256>: 8 words, bit i = meaning i
			Counting = 1  ///< CountingVoxel: one int8 count (stored sign extended in one word)
		};

		/// Maximum number of entries (limit of the sort primitive, Primitives::kMaxSortElements). Insertions that would exceed it throw.
		static constexpr uint32_t kMaxEntries = Primitives::kMaxSortElements;

		~VoxelListBase();
		VoxelListBase(const VoxelListBase&) = delete;
		VoxelListBase& operator=(const VoxelListBase&) = delete;
		VoxelListBase(VoxelListBase&&) noexcept;
		VoxelListBase& operator=(VoxelListBase&&) noexcept;

		[[nodiscard]] Kind kind() const { return kind_; }
		[[nodiscard]] Context& context() const { return *context_; }
		/// Dimensions of the reference map (ids / coordinates refer to it).
		[[nodiscard]] Dim3ui reference_dimensions() const { return dimensions_; }
		[[nodiscard]] float voxel_side_length() const { return voxel_side_length_; }
		/// Number of entries.
		[[nodiscard]] uint32_t size() const;
		[[nodiscard]] bool empty() const { return size() == 0; }
		/// Bytes of the three entry arrays (ids, coordinates, payload) for the current size.
		[[nodiscard]] uint64_t memory_usage_bytes() const;

		/// Removes all entries.
		void clear();

		/// Sorts the entries by id and merges entries with equal ids (done by every insert_*; public for completeness).
		void make_unique();

		/// Inserts each point as voxel floor(p / voxel_side_length) with the given meaning (counting lists: count 1,
		/// the meaning is ignored). Points outside the reference map (negative, too large, NaN, inf) are dropped.
		/// \return number of dropped points
		uint32_t insert_point_cloud(std::span<const Point3f> points, uint8_t meaning);
		/// Inserts voxel coordinates. \return number of coordinates outside the reference map (dropped).
		uint32_t insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning);
		/// All sub clouds with the same meaning. \return number of dropped points.
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning);
		/// Sub cloud k uses meanings[k] (one meaning per sub cloud). \return number of dropped points.
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings);

		// ---- host access (blocking copies) ----
		[[nodiscard]] std::vector<uint32_t> download_ids() const;
		[[nodiscard]] std::vector<Coord3ui> download_coordinates() const;
		/// Payload words: 8 words per entry for Bit lists (bit i of an entry is bit i % 32 of word i / 32), one word per
		/// entry (the sign extended count) for Counting lists.
		[[nodiscard]] std::vector<uint32_t> download_voxel_words() const;

		/// Replaces all entries. The entries must be valid: ids strictly ascending, every id the linear index of its coordinates, coordinates inside
		/// the reference map, `payload` of words-per-entry words each (bit lists 8, counting lists one sign extended int8 per entry).
		/// \throws VulkanError for anything else; the list is unchanged then
		void assign_entries(std::span<const uint32_t> ids, std::span<const Coord3ui> coordinates, std::span<const uint32_t> payload);
		/// The reason why assign_entries would refuse these entries for a list of `kind` with these reference dimensions, empty if they are valid.
		[[nodiscard]] static std::optional<std::string> entries_problem(Dim3ui reference_dimensions, Kind kind, std::span<const uint32_t> ids,
			std::span<const Coord3ui> coordinates, std::span<const uint32_t> payload);

		// ---- device access for other Vulkan code ----
		/// Buffers hold at least `size()` entries (ids: 1 word, coordinates: 3 words, payload: words per entry).
		[[nodiscard]] const DeviceBuffer& ids_buffer() const;
		[[nodiscard]] const DeviceBuffer& coordinates_buffer() const;
		[[nodiscard]] const DeviceBuffer& voxel_buffer() const;

	protected:
		VoxelListBase(Context& context, Dim3ui reference_dimensions, float voxel_side_length, Kind kind);

		/// Removes the entries whose id is contained in `other` (same reference dimensions required).
		void subtract_entries_of(const VoxelListBase& other);
		/// Entries of this list whose id is in `other`; if `threshold` is set only counting voxels >= *threshold count.
		/// Marks nothing. \return number of such entries
		uint32_t count_common_entries(const VoxelListBase& other, const int8_t* threshold) const;
		/// Removes the entries of a Counting list with count < threshold.
		void remove_counts_below(int8_t threshold);
		/// Bit lists: marks entries whose voxel in `map` is occupied with eBVM_COLLISION. \return number of collisions
		uint32_t collide_with_bit_map(const BitVoxelMap& map);
		/// Bit lists: appends all occupied voxels of the map (payload copied), then make_unique().
		uint32_t append_occupied_voxels(const BitVoxelMap& map);
		/// Appends the entries of `other` moved by `offset` (host side: the entries of `other` are downloaded), then make_unique().
		/// Payload: bit list from bit list: copied; bit list from counting list: no bits; counting list from counting list: the counts;
		/// counting list from bit list: count 0 (all as CUDA). `meaning` (bit lists only) replaces the payload of the new entries by that one bit.
		/// \return number of entries of `other` whose moved voxel lies outside the reference map (dropped)
		/// \throws VulkanError for another context or other reference dimensions or voxel side length, or `&other == this`
		uint32_t merge_entries(const VoxelListBase& other, Offset3i offset, std::optional<uint8_t> meaning);

	private:
		struct Impl;

		Context* context_;
		Dim3ui dimensions_;
		float voxel_side_length_;
		Kind kind_;
		std::unique_ptr<Impl> impl_;
	};

	/// Sparse BitVoxel<256> list (CUDA voxellist::BitVoxelList<256>).
	class BitVoxelList : public VoxelListBase
	{
	public:
		/// \param reference_dimensions dimensions of the map the ids refer to; the number of voxels must be below 2^32
		///        and every axis at most 2^24
		BitVoxelList(Context& context, Dim3ui reference_dimensions, float voxel_side_length);

		/// Number of entries whose id also occurs in `other` (CUDA collideWith(list), no offset).
		[[nodiscard]] uint32_t collide_with(const BitVoxelList& other) const;
		/// Number of entries whose voxel in `map` is occupied (any bit except eBVM_FREE). Side effect (as in CUDA):
		/// these entries get the eBVM_COLLISION bit. The list voxel itself is not inspected.
		/// \throws VulkanError if the map dimensions differ from the reference dimensions
		uint32_t collide_with(const BitVoxelMap& map);
		/// Removes the entries whose id occurs in `other` (CUDA subtract).
		void subtract(const BitVoxelList& other);
		/// Appends every occupied voxel of `map` (payload copied; OR-ed into entries that already exist). No CUDA
		/// counterpart; replaces the map scan that CUDA does in extractCubes for the viewer.
		/// \return number of voxels appended
		uint32_t insert_occupied_voxels(const BitVoxelMap& map);

		/// CUDA merge of a bit list / counting list / bit map into this list (new entries are OR-ed into existing ones).
		/// \return number of entries dropped because their moved voxel lies outside (0 for a map)
		uint32_t merge(const BitVoxelList& other, Offset3i offset = {}, std::optional<uint8_t> new_meaning = std::nullopt);
		uint32_t merge(const CountingVoxelList& other, Offset3i offset = {}, std::optional<uint8_t> new_meaning = std::nullopt);
		/// The occupied voxels of the map (same dimensions); `new_meaning` replaces their bits. CUDA applies an offset to the ids after the
		/// copy, which is not a shift of the voxels, so there is none here.
		uint32_t merge(const BitVoxelMap& other, std::optional<uint8_t> new_meaning = std::nullopt);

		/// One 256-bit vector per entry.
		[[nodiscard]] std::vector<BitVector256> download_bit_vectors() const;
	};

	/// Sparse counting list (CUDA voxellist::CountingVoxelList; int8 counts).
	class CountingVoxelList : public VoxelListBase
	{
	public:
		CountingVoxelList(Context& context, Dim3ui reference_dimensions, float voxel_side_length);

		/// Number of entries with count >= threshold whose id occurs in `other` (CUDA collideWith(BitVoxelList)).
		[[nodiscard]] uint32_t collide_with(const BitVoxelList& other, int8_t threshold) const;
		/// Removes the entries with count < threshold.
		void remove_underpopulated(int8_t threshold);
		/// Removes the entries whose id occurs in `other`.
		void subtract(const CountingVoxelList& other);
		/// Removes the entries whose id occurs in the bit list (CUDA subtractFromCountingVoxelList).
		void subtract_from_counting_list(const BitVoxelList& other);

		/// CUDA merge: entries of another counting list keep their counts, entries of a bit list get count 0; counts of equal ids are added
		/// (wrapping like int8_t). \return number of entries dropped (moved outside the reference map)
		uint32_t merge(const CountingVoxelList& other, Offset3i offset = {});
		uint32_t merge(const BitVoxelList& other, Offset3i offset = {});

		[[nodiscard]] std::vector<int8_t> download_counts() const;
	};
}

#endif
