// Shared by the façade tests: the whole content of a map, list or octree as one comparable vector.
#ifndef GVL_TEST_FACADE_SUPPORT_H_INCLUDED
#define GVL_TEST_FACADE_SUPPORT_H_INCLUDED

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include <gpu_voxels/vulkan/GpuVoxels.h>

namespace gvl_test
{
	using namespace gpu_voxels::vulkan;

	inline std::vector<uint64_t> snapshot(const BitVoxelMap& m)
	{
		const auto words = m.download_words();
		return { words.begin(), words.end() };
	}
	inline std::vector<uint64_t> snapshot(const ProbVoxelMap& m)
	{
		std::vector<uint64_t> out;
		for (const int8_t v : m.download_occupancy())
			out.push_back(static_cast<uint8_t>(v));
		return out;
	}
	inline std::vector<uint64_t> snapshot(const DistanceVoxelMap& m)
	{
		const auto words = m.download_words();
		return { words.begin(), words.end() };
	}
	inline std::vector<uint64_t> snapshot(const VoxelListBase& m)
	{
		std::vector<uint64_t> out;
		for (const uint32_t id : m.download_ids())
			out.push_back(id);
		out.push_back(UINT64_MAX);
		for (const uint32_t w : m.download_voxel_words())
			out.push_back(w);
		return out;
	}
	inline std::vector<uint64_t> snapshot(const Octree& m)
	{
		std::vector<uint64_t> out;
		for (const OctreeNode& n : m.dump())
		{
			out.push_back(n.id);
			out.push_back((uint64_t(n.level) << 32) | (uint64_t(n.status) << 16) | (uint64_t(static_cast<uint8_t>(n.occupancy)) << 8) | (n.has_children ? 1u : 0u));
		}
		return out;
	}

	inline std::vector<uint64_t> snapshot(GpuVoxelsMap& m)
	{
		if (auto* p = m.as<BitVoxelMap>()) return snapshot(*p);
		if (auto* p = m.as<ProbVoxelMap>()) return snapshot(*p);
		if (auto* p = m.as<DistanceVoxelMap>()) return snapshot(*p);
		if (auto* p = m.as<BitVoxelList>()) return snapshot(static_cast<const VoxelListBase&>(*p));
		if (auto* p = m.as<CountingVoxelList>()) return snapshot(static_cast<const VoxelListBase&>(*p));
		if (auto* p = m.as<Octree>()) return snapshot(*p);
		ADD_FAILURE() << "map without a class";
		return {};
	}
}

#endif
