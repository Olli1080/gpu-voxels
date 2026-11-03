// CUDA parity, scenario 1 (list x map collisions): runs the calls of parity_inputs.h on the CUDA GpuVoxels and prints one line per call.
// The output is stored in tests/cuda_parity/golden/collisions_cuda.txt and compared by tests/vulkan/test_cuda_parity.cpp.
// Usage: cuda_collisions_dump > collisions_cuda.txt
#include <cstdio>
#include <iostream>
#include <vector>

#include <gpu_voxels/GpuVoxels.h>
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
	gvl.reset();
	return 0;
}
