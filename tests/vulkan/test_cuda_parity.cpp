// CUDA parity checks: the Vulkan results are compared with results that the CUDA library produced on an NVIDIA GPU (golden files recorded by the
// programs in tests/cuda_parity; they run the same inputs from tests/cuda_parity/parity_inputs.h). A difference is a real finding: the test reports
// every differing line. Method and results: Agent/Analysis/2026-11-03_cuda_parity.md
#include "vulkan_fixture.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>

#include "../cuda_parity/parity_inputs.h"
#include "list_collision_model.h"

using namespace gpu_voxels::vulkan;
using namespace gvl_test::collision_model;
using gvl_test::VulkanTest;

namespace
{
	/// The lines of a golden file that start with one of the prefixes (the file holds the lines of all scenarios).
	std::vector<std::string> read_lines(const std::string& name, const std::vector<std::string>& prefixes)
	{
		std::ifstream in(std::filesystem::path(GVL_TEST_PARITY_DIR) / name);
		std::vector<std::string> lines;
		for (std::string text; std::getline(in, text);)
		{
			if (!text.empty() && text.back() == '\r')
				text.pop_back();
			for (const std::string& prefix : prefixes)
				if (text.rfind(prefix, 0) == 0)
				{
					lines.push_back(text);
					break;
				}
		}
		return lines;
	}

	void insert(GpuVoxels& gvl, const char* map, const std::vector<gvl_parity::PointSet>& sets)
	{
		for (const auto& set : sets)
		{
			std::vector<Point3f> points;
			for (const auto& p : set.points)
				points.push_back({ p[0], p[1], p[2] });
			ASSERT_TRUE(gvl.insert_point_cloud_into_map(points, map, static_cast<uint8_t>(set.meaning)));
		}
	}

	std::vector<int> set_bits(const BitVector256& v)
	{
		std::vector<int> bits;
		for (int b = 0; b < 256; ++b)
			if ((v[b / 32] >> (b % 32)) & 1u)
				bits.push_back(b);
		return bits;
	}
}

// Scenario 1: BitVoxelList x BitVoxelMap / ProbVoxelMap, all three forms, thresholds, offsets and masks (tests/cuda_parity/cuda_collisions_dump.cu).
// The CUDA kernels were changed to check the border per axis (VoxelMapOperations.h isInsideAfterOffset), the golden file collisions_cuda.txt is a run of the
// changed CUDA library; collisions_cuda_unfixed_border.txt is a run of the original one (linear offset). Required: Vulkan == CUDA in every line, Vulkan ==
// host model with the 3D offset, and the original CUDA == host model with the linear offset (so the change of CUDA is exactly the border rule).
TEST_F(VulkanTest, CudaParityListMapCollisions)
{
	const std::vector<std::string> golden = read_lines("collisions_cuda.txt", { "default ", "types ", "mask ", "list entries" });
	const std::vector<std::string> unfixed = read_lines("collisions_cuda_unfixed_border.txt", { "default ", "types ", "mask ", "list entries" }); // CUDA as upstream: linear offset
	ASSERT_GT(golden.size(), 400u) << "golden file missing: " << GVL_TEST_PARITY_DIR;

	GpuVoxels gvl(ctx());
	gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
	ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELMAP, "bitmap"), nullptr);
	ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "probmap"), nullptr);
	insert(gvl, "bitmap", gvl_parity::bit_map_sets());
	insert(gvl, "probmap", gvl_parity::prob_map_sets());
	const BitVoxelMap& bit_map = *gvl.get_map("bitmap")->as<BitVoxelMap>();
	const ProbVoxelMap& prob_map = *gvl.get_map("probmap")->as<ProbVoxelMap>();
	const MapView bit_view = view_of(bit_map), prob_view = view_of(prob_map);
	const auto mask_bits = gvl_parity::masks();

	std::vector<std::string> vulkan, model_3d, model_linear;
	uint64_t entries = 0;
	for (const gvl_parity::Call& c : gvl_parity::calls())
	{
		ASSERT_TRUE(gvl.del_map("list") || vulkan.empty());
		ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "list"), nullptr);
		insert(gvl, "list", gvl_parity::list_sets());
		BitVoxelList& list = *gvl.get_map("list")->as<BitVoxelList>();
		entries = list.size();
		const Offset3i offset{ c.offset[0], c.offset[1], c.offset[2] };
		const std::vector<Entry> before = entries_of(list);

		BitVector256 mask{};
		for (const int b : mask_bits[c.mask])
			mask[b / 32] |= 1u << (b % 32);
		const Mode mode = c.form == gvl_parity::Form::Default ? Mode::Default : c.form == gvl_parity::Form::Types ? Mode::Types : Mode::Mask;
		const bool types_form = c.form == gvl_parity::Form::Types;
		for (const bool linear : { false, true })
		{
			const Expected e = model(before, c.prob ? prob_view : bit_view, mode, offset, c.threshold, mask, linear);
			(linear ? model_linear : model_3d).push_back(gvl_parity::line(c, e.count, types_form ? set_bits(e.types) : std::vector<int>{}));
		}

		uint64_t count = 0;
		std::vector<int> types;
		if (c.form == gvl_parity::Form::Default)
			count = c.prob ? list.collide_with(prob_map, c.threshold, offset) : list.collide_with(bit_map, offset);
		else if (types_form)
		{
			const auto result = c.prob ? list.collide_with_types(prob_map, c.threshold, offset) : list.collide_with_types(bit_map, c.threshold, offset);
			count = result.count;
			types = set_bits(result.types);
		}
		else
			count = c.prob ? list.collide_with_type_mask(prob_map, mask, c.threshold, offset) : list.collide_with_type_mask(bit_map, mask, c.threshold, offset);
		vulkan.push_back(gvl_parity::line(c, count, types));
	}
	for (auto* lines : { &vulkan, &model_3d, &model_linear })
		lines->push_back("list entries " + std::to_string(entries));

	if (const char* dump = std::getenv("GVL_PARITY_DUMP"))
	{
		std::ofstream out(dump);
		for (const std::string& text : vulkan)
			out << text << '\n';
	}

	ASSERT_EQ(vulkan.size(), golden.size());
	ASSERT_EQ(vulkan.size(), unfixed.size());
	std::size_t vulkan_vs_3d = 0, vulkan_vs_cuda = 0, unfixed_vs_linear = 0, border_lines = 0;
	for (std::size_t i = 0; i < vulkan.size(); ++i)
	{
		if (vulkan[i] != model_3d[i] && vulkan_vs_3d++ < 20)
			ADD_FAILURE() << "Vulkan differs from its model (3D offset):\nVulkan: " << vulkan[i] << "\nmodel:  " << model_3d[i];
		if (vulkan[i] != golden[i] && vulkan_vs_cuda++ < 20)
			ADD_FAILURE() << "Vulkan differs from CUDA:\nCUDA:   " << golden[i] << "\nVulkan: " << vulkan[i];
		if (unfixed[i] != model_linear[i] && unfixed_vs_linear++ < 20)
			ADD_FAILURE() << "CUDA before the border fix differs from the linear offset model:\nCUDA:  " << unfixed[i] << "\nmodel: " << model_linear[i];
		border_lines += unfixed[i] != golden[i] ? 1 : 0;
	}
	std::cout << "[parity] " << vulkan.size() << " lines; the border fix of CUDA changed " << border_lines << "\n";
	EXPECT_EQ(vulkan_vs_3d, 0u);
	EXPECT_EQ(vulkan_vs_cuda, 0u);
	EXPECT_EQ(unfixed_vs_linear, 0u);
	EXPECT_GT(border_lines, 100u) << "the scenario must exercise the border";
}

// Scenario 2: GpuVoxelsMap::merge of a bit list into a bit list with an offset (entries that leave the map are dropped; the CUDA lines are hashes of the
// whole resulting list: coordinates and set bits in list order).
TEST_F(VulkanTest, CudaParityListMerge)
{
	const std::vector<std::string> golden = read_lines("collisions_cuda.txt", { "merge " });
	ASSERT_GT(golden.size(), 20u) << "golden file missing: " << GVL_TEST_PARITY_DIR;

	GpuVoxels gvl(ctx());
	gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
	std::vector<std::string> mine;
	for (const gvl_parity::MergeCall& c : gvl_parity::merge_calls())
	{
		for (const char* name : { "target", "source" })
		{
			gvl.del_map(name);
			ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, name), nullptr);
		}
		insert(gvl, "target", gvl_parity::merge_target_sets());
		insert(gvl, "source", gvl_parity::list_sets());
		const GpuVoxelsMapSharedPtr target = gvl.get_map("target");
		target->merge(*gvl.get_map("source"), { c.offset[0], c.offset[1], c.offset[2] },
			c.meaning < 0 ? std::nullopt : std::optional<uint8_t>(static_cast<uint8_t>(c.meaning)));
		const BitVoxelList& list = *target->as<BitVoxelList>();
		const auto coordinates = list.download_coordinates();
		const auto vectors = list.download_bit_vectors();
		std::vector<gvl_parity::ListEntry> entries;
		for (std::size_t i = 0; i < coordinates.size(); ++i)
			entries.push_back({ coordinates[i].x, coordinates[i].y, coordinates[i].z, set_bits(vectors[i]) });
		mine.push_back(gvl_parity::merge_line(c, entries));
	}
	ASSERT_EQ(mine.size(), golden.size());
	std::size_t different = 0;
	for (std::size_t i = 0; i < mine.size(); ++i)
		if (mine[i] != golden[i] && different++ < 20)
			ADD_FAILURE() << "Vulkan differs from CUDA:\nCUDA:   " << golden[i] << "\nVulkan: " << mine[i];
	EXPECT_EQ(different, 0u);
}
