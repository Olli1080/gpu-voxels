#pragma once

#include <gpu_visualization/CubelistContext.h>
#include "gpu_visualization/VisualizerContext.h"

namespace gpu_voxels
{
	namespace visualization
	{
		void calculateNumberOfCubeTypes(VisualizerContext& current_ctx, CubelistContext& context);
		thrust::device_vector<uint32_t> test(VisualizerContext& current_ctx, VoxelmapContext& context);
		void fillGLBufferWithCubelistWOUpdate(VisualizerContext& current_ctx, CubelistContext& context, uint32_t index);
	}
}