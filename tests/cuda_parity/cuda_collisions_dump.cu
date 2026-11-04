// CUDA parity, scenarios 1 to 5 (list x map collisions, list merges, map x map collisions, prob. map inserts, sensor data): runs the calls of parity_inputs.h on the CUDA GpuVoxels and prints one line per call.
// The output is stored in tests/cuda_parity/golden/collisions_cuda.txt and compared by tests/vulkan/test_cuda_parity.cpp.
// Usage: cuda_collisions_dump > collisions_cuda.txt
#include <cstdio>
#include <iostream>
#include <vector>

#include <gpu_voxels/GpuVoxels.h>
#include <gpu_voxels/helpers/PointCloud.h>
#include <gpu_voxels/logging/logging_gpu_voxels.h>

#include "parity_inputs.h"

using namespace gpu_voxels;
using namespace gvl_parity;

namespace
{
	void insert(GpuVoxelsSharedPtr& gvl, const char* map, const std::vector<PointSet>& sets)
	{
		for (const PointSet& set : sets)
		{
			std::vector<Vector3f> points;
			for (const auto& p : set.points)
				points.emplace_back(p[0], p[1], p[2]);
			if (!gvl->insertPointCloudIntoMap(points, map, BitVoxelMeaning(set.meaning)))
				std::fprintf(stderr, "insert into %s failed\n", map);
		}
	}

	std::vector<int> set_bits(const BitVectorVoxel& voxel)
	{
		std::vector<int> bits;
		for (int b = 0; b < 256; ++b)
			if (voxel.bitVector().getBit(b))
				bits.push_back(b);
		return bits;
	}

	/// The entries of a bit list in list order (sorted by id).
	std::vector<ListEntry> list_entries(voxellist::BitVectorVoxelList& list)
	{
		const size_t n = list.getDimensions().x();
		thrust::host_vector<Vector3ui> coords;
		list.copyCoordsToHost(coords);
		std::vector<BitVectorVoxel> voxels(n);
		if (n > 0)
			HANDLE_CUDA_ERROR(cudaMemcpy(voxels.data(), list.getDeviceDataPtr(), n * sizeof(BitVectorVoxel), cudaMemcpyDeviceToHost));
		std::vector<ListEntry> entries;
		for (size_t i = 0; i < n; ++i)
			entries.push_back({ coords[i].x(), coords[i].y(), coords[i].z(), set_bits(voxels[i]) });
		return entries;
	}
}

int main(int argc, char* argv[])
{
	icl_core::logging::initialize(argc, argv);
	GpuVoxelsSharedPtr gvl = GpuVoxels::getInstance();
	gvl->initialize(kDim, kDim, kDim, 1.0f);
	gvl->addMap(MT_BITVECTOR_VOXELMAP, "bitmap");
	gvl->addMap(MT_PROBAB_VOXELMAP, "probmap");
	insert(gvl, "bitmap", bit_map_sets());
	insert(gvl, "probmap", prob_map_sets());
	const auto* bit_map = gvl->getMap("bitmap")->as<voxelmap::BitVectorVoxelMap>();
	const auto* prob_map = gvl->getMap("probmap")->as<voxelmap::ProbVoxelMap>();
	const auto mask_bits = masks();

	for (const Call& c : calls())
	{
		// every call starts with a fresh list: a collision marks the entries (and the next call would read the mark)
		gvl->delMap("list");
		gvl->addMap(MT_BITVECTOR_VOXELLIST, "list");
		insert(gvl, "list", list_sets());
		auto* list = gvl->getMap("list")->as<voxellist::BitVectorVoxelList>();
		const Vector3i offset(c.offset[0], c.offset[1], c.offset[2]);

		size_t count = 0;
		BitVectorVoxel types;
		if (c.form == Form::Default)
			count = c.prob ? list->collideWith(prob_map, c.threshold, offset) : list->collideWith(bit_map, c.threshold, offset);
		else if (c.form == Form::Types)
			count = c.prob ? list->collideWithTypes(prob_map, types, c.threshold, offset) : list->collideWithTypes(bit_map, types, c.threshold, offset);
		else
		{
			BitVectorVoxel mask;
			for (const int b : mask_bits[c.mask])
				mask.bitVector().setBit(b);
			count = c.prob ? list->collideWithTypeMask(prob_map, mask, c.threshold, offset) : list->collideWithTypeMask(bit_map, mask, c.threshold, offset);
		}
		std::printf("%s\n", line(c, count, set_bits(types)).c_str());
	}
	std::printf("list entries %zu\n", static_cast<size_t>(gvl->getMap("list")->getDimensions().x()));

	// scenario 2: merge of a bit list into a bit list with an offset (fresh lists for every call)
	for (const MergeCall& c : merge_calls())
	{
		for (const char* name : { "target", "source" })
		{
			gvl->delMap(name);
			gvl->addMap(MT_BITVECTOR_VOXELLIST, name);
		}
		insert(gvl, "target", merge_target_sets());
		insert(gvl, "source", list_sets());
		const BitVoxelMeaning meaning = BitVoxelMeaning(c.meaning < 0 ? 0 : c.meaning);
		auto target = gvl->getMap("target");
		if (!target->merge(gvl->getMap("source"), Vector3i(c.offset[0], c.offset[1], c.offset[2]), c.meaning < 0 ? nullptr : &meaning))
			std::fprintf(stderr, "merge failed\n");
		std::printf("%s\n", merge_line(c, list_entries(*target->as<voxellist::BitVectorVoxelList>())).c_str());
	}

	// scenario 4: probabilistic map, insertPointCloud with swept volume meanings (one call each)
	gvl->addMap(MT_PROBAB_VOXELMAP, "prob4");
	{
		const auto values = [&]()
		{
			auto* map = gvl->getMap("prob4")->as<voxelmap::ProbVoxelMap>();
			const size_t n = static_cast<size_t>(kDim) * kDim * kDim;
			std::vector<ProbabilisticVoxel> host(n);
			HANDLE_CUDA_ERROR(cudaMemcpy(host.data(), map->getDeviceDataPtr(), n * sizeof(ProbabilisticVoxel), cudaMemcpyDeviceToHost));
			std::vector<int> out;
			for (const ProbabilisticVoxel& v : host)
				out.push_back(static_cast<int>(v.getOccupancy()));
			return out;
		};
		int step = 0;
		for (const PointSet& set : prob_insert_sets())
		{
			insert(gvl, "prob4", std::vector<PointSet>{ set });
			std::printf("%s\n", prob_line("insert " + std::to_string(step++) + " meaning=" + std::to_string(set.meaning), values()).c_str());
		}
		// the same voxel several times in one call, on a fresh map each
		for (const PointSet& set : prob_duplicate_sets())
		{
			gvl->delMap("prob4");
			gvl->addMap(MT_PROBAB_VOXELMAP, "prob4");
			insert(gvl, "prob4", std::vector<PointSet>{ set });
			std::printf("%s\n", prob_line("dup meaning=" + std::to_string(set.meaning) + " points=" + std::to_string(set.points.size()), values()).c_str());
		}
	}

	// scenario 3: voxel map x voxel map collisions with an offset (fresh maps for every call; the marked map is read back)
	for (const MapCall& c : map_calls())
	{
		for (const char* name : { "ma", "mb" })
			gvl->delMap(name);
		gvl->addMap(c.this_prob ? MT_PROBAB_VOXELMAP : MT_BITVECTOR_VOXELMAP, "ma");
		gvl->addMap(c.other_prob ? MT_PROBAB_VOXELMAP : MT_BITVECTOR_VOXELMAP, "mb");
		insert(gvl, "ma", map_sets(c.this_prob, 0));
		insert(gvl, "mb", map_sets(c.other_prob, 1));
		const Vector3i offset(c.offset[0], c.offset[1], c.offset[2]);
		const size_t n = static_cast<size_t>(kDim) * kDim * kDim;
		size_t count = 0;
		uint64_t hash = 0;
		if (!c.this_prob)
		{
			auto* a = gvl->getMap("ma")->as<voxelmap::BitVectorVoxelMap>();
			count = c.other_prob ? a->collideWith(gvl->getMap("mb")->as<voxelmap::ProbVoxelMap>(), c.threshold, offset)
			                     : a->collideWith(gvl->getMap("mb")->as<voxelmap::BitVectorVoxelMap>(), c.threshold, offset);
			std::vector<BitVectorVoxel> host(n);
			HANDLE_CUDA_ERROR(cudaMemcpy(host.data(), a->getDeviceDataPtr(), n * sizeof(BitVectorVoxel), cudaMemcpyDeviceToHost));
			std::vector<std::vector<int>> voxels;
			for (const BitVectorVoxel& v : host)
				voxels.push_back(set_bits(v));
			hash = hash_bit_map(voxels);
		}
		else
		{
			auto* a = gvl->getMap("ma")->as<voxelmap::ProbVoxelMap>();
			count = c.other_prob ? a->collideWith(gvl->getMap("mb")->as<voxelmap::ProbVoxelMap>(), c.threshold, offset)
			                     : a->collideWith(gvl->getMap("mb")->as<voxelmap::BitVectorVoxelMap>(), c.threshold, offset);
			std::vector<ProbabilisticVoxel> host(n);
			HANDLE_CUDA_ERROR(cudaMemcpy(host.data(), a->getDeviceDataPtr(), n * sizeof(ProbabilisticVoxel), cudaMemcpyDeviceToHost));
			std::vector<int> values;
			for (const ProbabilisticVoxel& v : host)
				values.push_back(static_cast<int>(v.getOccupancy()));
			hash = hash_occupancy(values);
		}
		std::printf("%s\n", map_line(c, count, hash).c_str());
	}
	// scenario 5: ProbVoxelMap::insertSensorData, one measurement per call, a fresh map for every group; the whole map is printed after every call
	gvl->addMap(MT_BITVECTOR_VOXELMAP, "robot5");
	insert(gvl, "robot5", sensor_robot_sets());
	for (const SensorGroup& group : sensor_groups())
	{
		gvl->delMap("sens5");
		gvl->addMap(MT_PROBAB_VOXELMAP, "sens5");
		auto* map = gvl->getMap("sens5")->as<voxelmap::ProbVoxelMap>();
		auto* robot = gvl->getMap("robot5")->as<voxelmap::BitVectorVoxelMap>();
		for (std::size_t i = 0; i < group.calls.size(); ++i)
		{
			const SensorCall& c = group.calls[i];
			PointCloud cloud(std::vector<Vector3f>{ Vector3f(c.hit[0], c.hit[1], c.hit[2]) });
			const bool cut = c.robot_bit >= 0;
			map->insertSensorData<BIT_VECTOR_LENGTH>(cloud, Vector3f(c.pose[0], c.pose[1], c.pose[2]), c.raycast, cut, BitVoxelMeaning(cut ? c.robot_bit : 0),
				cut ? robot->getDeviceDataPtr() : nullptr);
			const size_t n = static_cast<size_t>(kDim) * kDim * kDim;
			std::vector<ProbabilisticVoxel> host(n);
			HANDLE_CUDA_ERROR(cudaMemcpy(host.data(), map->getDeviceDataPtr(), n * sizeof(ProbabilisticVoxel), cudaMemcpyDeviceToHost));
			std::vector<int> values;
			for (const ProbabilisticVoxel& v : host)
				values.push_back(static_cast<int>(v.getOccupancy()));
			std::printf("%s\n", prob_line(sensor_label(group.name, i, c), values).c_str());
		}
	}
	gvl.reset();
	return 0;
}
