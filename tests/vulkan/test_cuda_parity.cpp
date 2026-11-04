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

// Scenario 4: ProbVoxelMap insert_point_cloud with swept volume meanings (one call each, UpdateMode::Accumulate): the whole map after every call, as the
// count of unknown / positive / negative voxels and a hash of all values. (CUDA insertSensorData does not launch in this build: K41, not compared.)
TEST_F(VulkanTest, CudaParityProbMap)
{
	const std::vector<std::string> golden = read_lines("collisions_cuda.txt", { "prob insert " });
	ASSERT_GT(golden.size(), 10u) << "golden file missing: " << GVL_TEST_PARITY_DIR;

	GpuVoxels gvl(ctx());
	gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
	ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "prob4"), nullptr);

	const auto values = [&]()
	{
		std::vector<int> out;
		for (const int8_t v : gvl.get_map("prob4")->as<ProbVoxelMap>()->download_occupancy())
			out.push_back(v);
		return out;
	};
	std::vector<std::string> mine;
	int step = 0;
	for (const gvl_parity::PointSet& set : gvl_parity::prob_insert_sets())
	{
		insert(gvl, "prob4", { set });
		mine.push_back(gvl_parity::prob_line("insert " + std::to_string(step++) + " meaning=" + std::to_string(set.meaning), values()));
	}
	ASSERT_EQ(mine.size(), golden.size());
	std::size_t different = 0;
	for (std::size_t i = 0; i < mine.size(); ++i)
		if (mine[i] != golden[i] && different++ < 20)
			ADD_FAILURE() << "Vulkan differs from CUDA:\nCUDA:   " << golden[i] << "\nVulkan: " << mine[i];
	EXPECT_EQ(different, 0u);
}

// Scenario 5: ProbVoxelMap::insertSensorData (CUDA) against insert_sensor_data (Vulkan), one measurement per call (several rays in one CUDA call lose updates,
// K42), a fresh map per group, the whole map after every call (tests/cuda_parity/parity_inputs.h sensor_groups()).
namespace
{
	/// `saturate`: feed the group "negative" with the coordinates CUDA effectively uses (gvl_parity::cuda_saturated); otherwise Vulkan gets the raw input.
	std::vector<std::string> sensor_vulkan_lines(GpuVoxels& gvl, UpdateMode mode, bool saturate)
	{
		std::vector<std::string> lines;
		gvl.del_map("robot5");
		gvl.add_map(MT_BITVECTOR_VOXELMAP, "robot5");
		insert(gvl, "robot5", gvl_parity::sensor_robot_sets());
		const BitVoxelMap& robot = *gvl.get_map("robot5")->as<BitVoxelMap>();
		for (const gvl_parity::SensorGroup& group : gvl_parity::sensor_groups())
		{
			gvl.del_map("sens5");
			gvl.add_map(MT_PROBAB_VOXELMAP, "sens5");
			ProbVoxelMap& map = *gvl.get_map("sens5")->as<ProbVoxelMap>();
			UpdateOptions options;
			options.mode = mode;
			for (std::size_t i = 0; i < group.calls.size(); ++i)
			{
				const gvl_parity::SensorCall& c = group.calls[i];
				const gvl_parity::SensorCall used = saturate && group.name == "negative" ? gvl_parity::cuda_saturated(c) : c;
				const std::vector<Point3f> hit{ { used.hit[0], used.hit[1], used.hit[2] } };
				const bool cut = c.robot_bit >= 0;
				map.insert_sensor_data(hit, { used.pose[0], used.pose[1], used.pose[2] }, c.raycast, cut ? &robot : nullptr, static_cast<uint8_t>(cut ? c.robot_bit : 0), options);
				std::vector<int> values;
				for (const int8_t v : map.download_occupancy())
					values.push_back(v);
				lines.push_back(gvl_parity::prob_line(gvl_parity::sensor_label(group.name, i, c), values));
			}
		}
		return lines;
	}
}

TEST_F(VulkanTest, CudaParitySensorData)
{
	const std::vector<std::string> golden = read_lines("collisions_cuda.txt", { "prob sensor " });
	const std::vector<std::string> unfixed = read_lines("negative_coordinates_cuda_unfixed.txt", { "prob sensor negative " }); // CUDA before K44 was fixed
	ASSERT_GT(golden.size(), 150u) << "golden file missing: " << GVL_TEST_PARITY_DIR;
	ASSERT_EQ(unfixed.size(), 18u);

	GpuVoxels gvl(ctx());
	gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
	for (const UpdateMode mode : { UpdateMode::Accumulate, UpdateMode::Atomic, UpdateMode::Sequential })
	{
		// 1. Vulkan equals CUDA in every line (rays, special, outside, negative, robot); K44: CUDA drops negative coordinates now, as Vulkan does
		const std::vector<std::string> mine = sensor_vulkan_lines(gvl, mode, false);
		ASSERT_EQ(mine.size(), golden.size());
		std::size_t different = 0;
		for (std::size_t i = 0; i < mine.size(); ++i)
			if (mine[i] != golden[i] && different++ < 40)
				ADD_FAILURE() << "mode " << static_cast<int>(mode) << ": Vulkan differs from CUDA:\nCUDA:   " << golden[i] << "\nVulkan: " << mine[i];
		EXPECT_EQ(different, 0u) << "mode " << static_cast<int>(mode);

		// 2. The original CUDA (negative coordinates and a NaN pose became voxel 0) equals Vulkan fed with those saturated coordinates
		const std::vector<std::string> saturated = sensor_vulkan_lines(gvl, mode, true);
		std::vector<std::string> negative;
		for (const std::string& line : saturated)
			if (line.rfind("prob sensor negative ", 0) == 0)
				negative.push_back(line);
		ASSERT_EQ(negative.size(), unfixed.size());
		for (std::size_t i = 0; i < negative.size(); ++i)
			EXPECT_EQ(negative[i], unfixed[i]) << "mode " << static_cast<int>(mode);
	}
}

// K44 for insert_point_cloud: negative, NaN and infinite coordinates, one point per call (gvl_parity::pointcloud_edge_points). The original CUDA put the
// negative and NaN points into voxel 0 of that axis (negative_coordinates_cuda_unfixed.txt); the fixed CUDA drops them like Vulkan (collisions_cuda.txt).
TEST_F(VulkanTest, CudaParityPointCloudNegativeCoordinates)
{
	const std::vector<std::string> golden = read_lines("collisions_cuda.txt", { "prob pcneg " });
	const std::vector<std::string> unfixed = read_lines("negative_coordinates_cuda_unfixed.txt", { "prob pcneg " });
	const auto points = gvl_parity::pointcloud_edge_points();
	ASSERT_EQ(golden.size(), points.size());
	ASSERT_EQ(unfixed.size(), points.size());

	for (const bool saturate : { false, true })
	{
		GpuVoxels gvl(ctx());
		gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
		ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "pcneg"), nullptr);
		for (std::size_t i = 0; i < points.size(); ++i)
		{
			const auto p = saturate ? gvl_parity::cuda_saturated_point(points[i]) : points[i];
			gvl.insert_point_cloud_into_map(std::vector<Point3f>{ { p[0], p[1], p[2] } }, "pcneg", 1); // false for a point outside of the map: not an error here
			std::vector<int> values;
			for (const int8_t v : gvl.get_map("pcneg")->as<ProbVoxelMap>()->download_occupancy())
				values.push_back(v);
			const std::string line = gvl_parity::prob_line("pcneg " + std::to_string(i), values);
			EXPECT_EQ(line, saturate ? unfixed[i] : golden[i]) << (saturate ? "unfixed CUDA vs Vulkan with saturated points" : "fixed CUDA vs Vulkan");
		}
	}

	// the same points into a fresh voxel list each (CUDA lists took them as well: kernelInsertGlobalPointCloud of the MapVoxelID lists)
	const std::vector<std::string> list_golden = read_lines("collisions_cuda.txt", { "listneg " });
	const std::vector<std::string> list_unfixed = read_lines("negative_coordinates_cuda_unfixed.txt", { "listneg " });
	ASSERT_EQ(list_golden.size(), points.size());
	ASSERT_EQ(list_unfixed.size(), points.size());
	for (const bool saturate : { false, true })
	{
		GpuVoxels gvl(ctx());
		gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
		for (std::size_t i = 0; i < points.size(); ++i)
		{
			gvl.del_map("listneg");
			ASSERT_NE(gvl.add_map(MT_BITVECTOR_VOXELLIST, "listneg"), nullptr);
			const auto p = saturate ? gvl_parity::cuda_saturated_point(points[i]) : points[i];
			gvl.insert_point_cloud_into_map(std::vector<Point3f>{ { p[0], p[1], p[2] } }, "listneg", 1);
			const BitVoxelList& list = *gvl.get_map("listneg")->as<BitVoxelList>();
			const auto coordinates = list.download_coordinates();
			const auto vectors = list.download_bit_vectors();
			std::vector<gvl_parity::ListEntry> entries;
			for (std::size_t e = 0; e < coordinates.size(); ++e)
				entries.push_back({ coordinates[e].x, coordinates[e].y, coordinates[e].z, set_bits(vectors[e]) });
			EXPECT_EQ(gvl_parity::list_edge_line(i, entries), saturate ? list_unfixed[i] : list_golden[i]) << (saturate ? "unfixed CUDA list" : "fixed CUDA list");
		}
	}

	// CUDA's Morton coded lists (OctreeVoxelID, created directly: GpuVoxels::addMap does not implement the type) had the same defect and are fixed the same way.
	// Vulkan has no Morton list; their entries (coordinates and bits) must equal those of the map-id list line by line, before and after the fix.
	const std::vector<std::string> morton_golden = read_lines("collisions_cuda.txt", { "mortonneg " });
	const std::vector<std::string> morton_unfixed = read_lines("negative_coordinates_cuda_unfixed.txt", { "mortonneg " });
	ASSERT_EQ(morton_golden.size(), points.size());
	ASSERT_EQ(morton_unfixed.size(), points.size());
	const auto as_list = [](std::string line) { return "listneg" + line.substr(std::string("mortonneg").size()); };
	for (std::size_t i = 0; i < points.size(); ++i)
	{
		EXPECT_EQ(as_list(morton_golden[i]), list_golden[i]);
		EXPECT_EQ(as_list(morton_unfixed[i]), list_unfixed[i]);
	}
}

// Scenario 3: voxel map x voxel map collisions with an offset (CUDA BitVoxelMap / ProbVoxelMap::collideWith(map, threshold, offset); K41 made them
// unusable until the kernel host functions were fixed). Vulkan has no map x map offset, so the offset lines are compared with a host model of the 3D rule
// that CUDA now implements (the voxel at c of the other map meets the voxel at c + offset of this map, a voxel that leaves the map does not collide); the
// lines with offset 0 are also run on the Vulkan classes. The model must reproduce the unfixed CUDA run (linear offset) with the linear rule as well.
namespace
{
	struct MapMapResult
	{
		uint64_t count = 0;
		uint64_t hash = 0;
	};

	/// CUDA rule of kernelCollideVoxelMapsDebug + DefaultCollider; `linear` = the original (offset added to the linear index, only the ends of the array checked).
	MapMapResult map_map_model(const gvl_parity::MapCall& c, const MapView& self, const MapView& other, bool linear)
	{
		const int level = level_of(c.threshold);
		const auto occupied = [&](const MapView& m, bool prob, Coord3ui at) { return prob ? m.occupancy_at(at) >= level : m.bit_occupied(at); };
		const uint64_t voxels = self.dims.voxel_count();
		std::vector<char> marked(voxels, 0);
		MapMapResult r;
		for (uint32_t z = 0; z < self.dims.z; ++z)
			for (uint32_t y = 0; y < self.dims.y; ++y)
				for (uint32_t x = 0; x < self.dims.x; ++x)
				{
					const int64_t linear_index = int64_t(other.index({ x, y, z })) + (int64_t(c.offset[2]) * self.dims.y + c.offset[1]) * self.dims.x + c.offset[0];
					const int64_t tx = int64_t(x) + c.offset[0], ty = int64_t(y) + c.offset[1], tz = int64_t(z) + c.offset[2];
					const bool inside = linear ? (linear_index >= 0 && linear_index < int64_t(voxels))
					                           : (tx >= 0 && ty >= 0 && tz >= 0 && tx < self.dims.x && ty < self.dims.y && tz < self.dims.z);
					if (!inside)
						continue;
					const Coord3ui target = linear ? Coord3ui{ uint32_t(linear_index % self.dims.x), uint32_t(linear_index / self.dims.x % self.dims.y), uint32_t(linear_index / self.dims.x / self.dims.y) }
					                               : Coord3ui{ uint32_t(tx), uint32_t(ty), uint32_t(tz) };
					if (occupied(self, c.this_prob, target) && occupied(other, c.other_prob, { x, y, z }))
					{
						++r.count;
						marked[self.index(target)] = 1;
					}
				}
		if (!c.this_prob)
		{
			std::vector<std::vector<int>> out(voxels);
			for (uint64_t i = 0; i < voxels; ++i)
			{
				BitVector256 v{};
				for (int w = 0; w < 8; ++w)
					v[w] = self.bit_words[i * 8 + w];
				if (marked[i])
					v[0] |= 1u << 2; // eBVM_COLLISION
				out[i] = set_bits(v);
			}
			r.hash = gvl_parity::hash_bit_map(out);
		}
		else
		{
			std::vector<int> out(voxels);
			for (uint64_t i = 0; i < voxels; ++i)
				out[i] = marked[i] ? std::min(127, self.occupancy[i] + 72) : self.occupancy[i]; // insert(eBVM_COLLISION): one occupied update (+72), clamped
			r.hash = gvl_parity::hash_occupancy(out);
		}
		return r;
	}

	std::string map_map_text(const gvl_parity::MapCall& c, const MapMapResult& r) { return gvl_parity::map_line(c, r.count, r.hash); }

	/// The same line from the Vulkan classes (offset 0 only).
	std::string map_map_vulkan(GpuVoxels& gvl, const gvl_parity::MapCall& c)
	{
		MapMapResult r;
		if (!c.this_prob)
		{
			BitVoxelMap& a = *gvl.get_map("ma")->as<BitVoxelMap>();
			r.count = c.other_prob ? a.collide_with(*gvl.get_map("mb")->as<ProbVoxelMap>(), c.threshold) : a.collide_with(*gvl.get_map("mb")->as<BitVoxelMap>());
			const auto words = a.download_words();
			std::vector<std::vector<int>> voxels;
			for (std::size_t i = 0; i < words.size() / 8; ++i)
			{
				BitVector256 v{};
				for (int w = 0; w < 8; ++w)
					v[w] = words[i * 8 + w];
				voxels.push_back(set_bits(v));
			}
			r.hash = gvl_parity::hash_bit_map(voxels);
		}
		else
		{
			ProbVoxelMap& a = *gvl.get_map("ma")->as<ProbVoxelMap>();
			r.count = c.other_prob ? a.collide_with(*gvl.get_map("mb")->as<ProbVoxelMap>(), c.threshold) : a.collide_with(*gvl.get_map("mb")->as<BitVoxelMap>(), c.threshold);
			std::vector<int> values;
			for (const int8_t v : a.download_occupancy())
				values.push_back(v);
			r.hash = gvl_parity::hash_occupancy(values);
		}
		return map_map_text(c, r);
	}
}

TEST_F(VulkanTest, CudaParityMapMapCollisions)
{
	const std::vector<std::string> golden = read_lines("collisions_cuda.txt", { "mapmap " });
	// The recording of the original CUDA only has the lines of a bit map as the map the call is made on ("bit x ..."): the original kernel read
	// outside the allocation for every voxel that the linear offset moved out of the array (garbage next to the allocation decides there), which a model
	// cannot reproduce for probabilistic maps (zero bytes count as occupancy 0), and the original ProbVoxelMap::collideWith(BitVectorVoxelMap*) read the
	// bit map as probabilistic voxels (K43).
	const std::vector<std::string> unfixed = read_lines("collisions_cuda_unfixed_border.txt", { "mapmap " }); // CUDA as upstream: linear offset
	ASSERT_EQ(golden.size(), gvl_parity::map_calls().size()) << "golden file missing: " << GVL_TEST_PARITY_DIR;

	GpuVoxels gvl(ctx());
	gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
	std::vector<std::string> model_3d, model_linear, golden_with_unfixed, vulkan_zero, golden_zero;
	std::size_t index = 0;
	for (const gvl_parity::MapCall& c : gvl_parity::map_calls())
	{
		for (const char* name : { "ma", "mb" })
			gvl.del_map(name);
		ASSERT_NE(gvl.add_map(c.this_prob ? MT_PROBAB_VOXELMAP : MT_BITVECTOR_VOXELMAP, "ma"), nullptr);
		ASSERT_NE(gvl.add_map(c.other_prob ? MT_PROBAB_VOXELMAP : MT_BITVECTOR_VOXELMAP, "mb"), nullptr);
		insert(gvl, "ma", gvl_parity::map_sets(c.this_prob, 0));
		insert(gvl, "mb", gvl_parity::map_sets(c.other_prob, 1));
		const GpuVoxelsMapSharedPtr a = gvl.get_map("ma"), b = gvl.get_map("mb");
		const MapView self = c.this_prob ? view_of(*a->as<ProbVoxelMap>()) : view_of(*a->as<BitVoxelMap>());
		const MapView other = c.other_prob ? view_of(*b->as<ProbVoxelMap>()) : view_of(*b->as<BitVoxelMap>());
		model_3d.push_back(map_map_text(c, map_map_model(c, self, other, false)));
		if (!c.this_prob)
		{
			model_linear.push_back(map_map_text(c, map_map_model(c, self, other, true)));
			golden_with_unfixed.push_back(golden[index]);
		}
		if (c.offset == std::array<int, 3>{ 0, 0, 0 })
		{
			vulkan_zero.push_back(map_map_vulkan(gvl, c));
			golden_zero.push_back(golden[index]);
		}
		++index;
	}

	std::size_t vs_model = 0, vs_unfixed = 0, vs_vulkan = 0, border_lines = 0, nonzero = 0;
	for (std::size_t i = 0; i < golden.size(); ++i)
	{
		if (golden[i] != model_3d[i] && vs_model++ < 20)
			ADD_FAILURE() << "CUDA (3D offset) differs from the model:\nCUDA:  " << golden[i] << "\nmodel: " << model_3d[i];
		nonzero += golden[i].find(" count=0 ") == std::string::npos ? 1 : 0;
	}
	ASSERT_EQ(unfixed.size(), model_linear.size());
	for (std::size_t i = 0; i < unfixed.size(); ++i)
	{
		if (unfixed[i] != model_linear[i] && vs_unfixed++ < 20)
			ADD_FAILURE() << "CUDA before the border fix differs from the linear offset model:\nCUDA:  " << unfixed[i] << "\nmodel: " << model_linear[i];
		border_lines += unfixed[i] != golden_with_unfixed[i] ? 1 : 0;
	}
	ASSERT_EQ(vulkan_zero.size(), golden_zero.size());
	for (std::size_t i = 0; i < golden_zero.size(); ++i)
		if (vulkan_zero[i] != golden_zero[i] && vs_vulkan++ < 20)
			ADD_FAILURE() << "Vulkan differs from CUDA:\nCUDA:   " << golden_zero[i] << "\nVulkan: " << vulkan_zero[i];
	std::cout << "[parity] " << golden.size() << " map x map lines (" << vulkan_zero.size() << " also on the Vulkan classes); the border fix of CUDA changed "
	          << border_lines << ", " << nonzero << " lines have collisions\n";
	EXPECT_EQ(vs_model, 0u);
	EXPECT_EQ(vs_unfixed, 0u);
	EXPECT_EQ(vs_vulkan, 0u);
	EXPECT_GT(border_lines, 40u) << "the scenario must exercise the border";
	EXPECT_GT(nonzero, 100u) << "the scenario must produce collisions";
}

// A voxel that is hit twice (or four times) in one insert_point_cloud call: Vulkan applies every update (the sum, clamped), CUDA applies exactly one
// (the voxel update of the kernel is not an atomic read-modify-write, the other updates of the call are lost). Documented difference, K42: the CUDA
// golden lines show 72 for two +72 updates and -10 for four -10 updates, Vulkan 127 and -40.
TEST_F(VulkanTest, CudaParityProbMapDuplicatePointsInOneCall)
{
	const std::vector<std::string> golden = read_lines("collisions_cuda.txt", { "prob dup " });
	ASSERT_EQ(golden.size(), 2u) << "golden file missing: " << GVL_TEST_PARITY_DIR;

	GpuVoxels gvl(ctx());
	gvl.initialize({ gvl_parity::kDim, gvl_parity::kDim, gvl_parity::kDim }, 1.0f);
	std::vector<std::string> mine;
	for (const gvl_parity::PointSet& set : gvl_parity::prob_duplicate_sets())
	{
		gvl.del_map("prob4");
		ASSERT_NE(gvl.add_map(MT_PROBAB_VOXELMAP, "prob4"), nullptr);
		insert(gvl, "prob4", { set });
		std::vector<int> values;
		for (const int8_t v : gvl.get_map("prob4")->as<ProbVoxelMap>()->download_occupancy())
			values.push_back(v);
		mine.push_back(gvl_parity::prob_line("dup meaning=" + std::to_string(set.meaning) + " points=" + std::to_string(set.points.size()), values));
	}
	EXPECT_NE(golden[0].find("values=72:1"), std::string::npos) << golden[0];
	EXPECT_NE(golden[1].find("values=-10:1"), std::string::npos) << golden[1];
	EXPECT_NE(mine[0].find("values=127:1"), std::string::npos) << mine[0];
	EXPECT_NE(mine[1].find("values=-40:1"), std::string::npos) << mine[1];
}
