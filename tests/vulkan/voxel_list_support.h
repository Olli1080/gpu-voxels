// Helpers shared by the voxel list tests (slice S4).
#ifndef GVL_TEST_VOXEL_LIST_SUPPORT_H_INCLUDED
#define GVL_TEST_VOXEL_LIST_SUPPORT_H_INCLUDED

#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/VoxelList.h>

#include "../reference/voxel_list_reference.h"

namespace gvl_test::lists
{
	using namespace gpu_voxels::vulkan;
	using gvl_reference::BitVoxelMapReference;
	using gvl_reference::VoxelListReference;

	inline constexpr Dim3ui kDims{ 37, 29, 41 }; // deliberately not powers of two
	inline constexpr float kSide = 0.1f;

	inline VoxelListReference bit_reference(Dim3ui d = kDims, float side = kSide) { return VoxelListReference(d.x, d.y, d.z, side, false); }
	inline VoxelListReference counting_reference(Dim3ui d = kDims, float side = kSide) { return VoxelListReference(d.x, d.y, d.z, side, true); }
	inline BitVoxelMapReference map_reference(Dim3ui d = kDims, float side = kSide) { return BitVoxelMapReference(d.x, d.y, d.z, side); }

	/// Random point strictly inside a voxel (fraction 0.1..0.9 of the voxel edge: no boundary cases).
	inline Point3f random_inside_point(std::mt19937& rng, Dim3ui d, float side)
	{
		std::uniform_int_distribution<uint32_t> vx(0, d.x - 1), vy(0, d.y - 1), vz(0, d.z - 1);
		std::uniform_real_distribution<float> frac(0.1f, 0.9f);
		return { (static_cast<float>(vx(rng)) + frac(rng)) * side, (static_cast<float>(vy(rng)) + frac(rng)) * side,
			(static_cast<float>(vz(rng)) + frac(rng)) * side };
	}

	inline std::vector<Point3f> random_points(std::mt19937& rng, std::size_t n, Dim3ui d = kDims, float side = kSide)
	{
		std::vector<Point3f> points(n);
		for (auto& p : points)
			p = random_inside_point(rng, d, side);
		return points;
	}

	inline uint8_t random_meaning(std::mt19937& rng)
	{
		static constexpr uint8_t meanings[] = { 1, 2, 4, 31, 32, 63, 64, 100, 128, 255 };
		return meanings[rng() % std::size(meanings)];
	}

	inline std::vector<uint32_t> flatten(const std::vector<Coord3ui>& coordinates)
	{
		std::vector<uint32_t> out;
		for (const auto& c : coordinates)
			out.insert(out.end(), { c.x, c.y, c.z });
		return out;
	}

	inline ::testing::AssertionResult equal_to_reference(const VoxelListBase& list, const VoxelListReference& ref)
	{
		if (list.size() != ref.size())
			return ::testing::AssertionFailure() << "size " << list.size() << " vs reference " << ref.size();
		const auto ids = list.download_ids();
		const auto expected_ids = ref.ids();
		for (std::size_t i = 0; i < ids.size(); ++i)
			if (ids[i] != expected_ids[i])
				return ::testing::AssertionFailure() << "id " << i << ": vulkan " << ids[i] << " reference " << expected_ids[i];
		const auto coords = flatten(list.download_coordinates());
		const auto expected_coords = ref.coordinate_words();
		for (std::size_t i = 0; i < coords.size(); ++i)
			if (coords[i] != expected_coords[i])
				return ::testing::AssertionFailure() << "coordinate word " << i << ": vulkan " << coords[i] << " reference " << expected_coords[i];
		const auto words = list.download_voxel_words();
		const auto expected_words = ref.voxel_words();
		if (words.size() != expected_words.size())
			return ::testing::AssertionFailure() << "voxel word count " << words.size() << " vs " << expected_words.size();
		for (std::size_t i = 0; i < words.size(); ++i)
			if (words[i] != expected_words[i])
				return ::testing::AssertionFailure() << "voxel word " << i << ": vulkan 0x" << std::hex << words[i] << " reference 0x"
					<< expected_words[i];
		return ::testing::AssertionSuccess();
	}
}

#endif
