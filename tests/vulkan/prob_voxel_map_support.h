// Helpers shared by the probabilistic voxel map tests (slice S3).
#ifndef GVL_TEST_PROB_VOXEL_MAP_SUPPORT_H_INCLUDED
#define GVL_TEST_PROB_VOXEL_MAP_SUPPORT_H_INCLUDED

#include <gtest/gtest.h>

#include <vector>

#include <gpu_voxels/vulkan/ProbVoxelMap.h>

#include "../reference/prob_voxel_map_reference.h"
#include "voxel_list_support.h" // random_points(), random_inside_point()

namespace gvl_test::prob
{
	using namespace gpu_voxels::vulkan;
	using gvl_reference::ProbVoxelMapReference;
	using namespace gvl_test::lists;

	inline constexpr Dim3ui kProbDims{ 23, 19, 17 }; // deliberately not powers of two

	inline ProbVoxelMapReference prob_reference(Dim3ui d = kProbDims, float side = kSide)
	{
		return ProbVoxelMapReference(d.x, d.y, d.z, side);
	}

	inline ::testing::AssertionResult equal_to_reference(const ProbVoxelMap& map, const ProbVoxelMapReference& ref)
	{
		const auto gpu = map.download_occupancy();
		const Dim3ui d = map.dimensions();
		for (std::size_t i = 0; i < gpu.size(); ++i)
			if (gpu[i] != ref.get(i))
				return ::testing::AssertionFailure() << "voxel " << i << " (" << i % d.x << "," << (i / d.x) % d.y << ","
					<< i / d.x / d.y << "): vulkan " << int(gpu[i]) << " reference " << ref.get(i);
		return ::testing::AssertionSuccess();
	}

	inline const char* mode_name(UpdateMode mode)
	{
		switch (mode)
		{
		case UpdateMode::Atomic: return "Atomic";
		case UpdateMode::Accumulate: return "Accumulate";
		case UpdateMode::Sequential: return "Sequential";
		}
		return "?";
	}

	/// Applies the generated events to the reference the way `mode` promises (Atomic is only used when no voxel saturates,
	/// where all serial orders agree).
	inline void apply_to_reference(ProbVoxelMapReference& ref, const std::vector<ProbVoxelMapReference::Event>& events, UpdateMode mode)
	{
		if (mode == UpdateMode::Accumulate)
			ref.apply_accumulated(events);
		else
			ref.apply_sequential(events);
	}
}

#endif
