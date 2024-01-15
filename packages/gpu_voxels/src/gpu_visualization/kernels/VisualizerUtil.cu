#include "VisualizerUtil.h"

#include "VoxelMapVisualizerOperations.h"

//using namespace gpu_voxels;
//using namespace gpu_voxels::visualization;

namespace gpu_voxels
{
	namespace visualization
	{
		void calculateNumberOfCubeTypes(VisualizerContext& current_ctx, CubelistContext& context)
		{
			thrust::device_vector<uint32_t> num_voxels_per_type(context.voxel_types(), 0);
			//thrust::fill(context.m_d_num_voxels_per_type.begin(), context.m_d_num_voxels_per_type.end(), 0);
			// Launch kernel to copy data into the OpenGL buffer. <<<context.getNumberOfCubes(),1>>><<<num_threads_per_block,num_blocks>>>
			calculate_cubes_per_type_list<<<context.num_blocks(), context.threads_per_block()>>>(
				context.getCubesDevicePointer(),/**/
				context.getNumberOfCubes(),/**/
				thrust::raw_pointer_cast(num_voxels_per_type.data()),
				thrust::raw_pointer_cast(current_ctx.m_d_draw_types.data()),/**/
				thrust::raw_pointer_cast(current_ctx.m_d_prefixes.data()));/**/
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			context.set_num_voxels_per_type(num_voxels_per_type);
		}

		thrust::device_vector<uint32_t> test(VisualizerContext& current_ctx, VoxelmapContext& context)
		{
			thrust::device_vector<uint32_t> indices(context.voxel_types(), 0);

			float4* vbo_ptr; //float4 because the translation (x,y,z) and cube size (w) will be stored in there
			size_t num_bytes; // size of the buffer
			HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, context.cuda_ressource(), nullptr));
			HANDLE_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vbo_ptr), &num_bytes, *context.cuda_ressource()));

			// Launch kernel to copy data into the OpenGL buffer.
			// fill_vbo_without_precounting<<< dim3(1,1,1), dim3(1,1,1)>>>(/**/
			// CHECK_CUDA_ERROR();
			switch (context.m_voxelMap->getMapType())
			{
			case MT_BITVECTOR_VOXELMAP:
			{
				if constexpr (BIT_VECTOR_LENGTH > MAX_DRAW_TYPES)
					LOGGING_ERROR_C(Visualization, Visualizer,
						"Only " << MAX_DRAW_TYPES << " different draw types supported. But bit vector has " << BIT_VECTOR_LENGTH << " different types." << endl);

				fill_vbo_without_precounting<<<context.num_blocks(), context.threads_per_block()>>>(
					/**/
					static_cast<BitVectorVoxel*>(context.m_voxelMap->getVoidDeviceDataPtr()),/**/
					context.m_voxelMap->getDimensions(),/**/
					current_ctx.m_dim_svoxel,/**/
					current_ctx.m_view_start_voxel_pos,/**/
					current_ctx.m_view_end_voxel_pos,/**/
					context.occupancy_threshold(),/**/
					vbo_ptr,/**/
					thrust::raw_pointer_cast(context.m_d_vbo_offsets.data()),/**/
					thrust::raw_pointer_cast(context.m_d_vbo_segment_voxel_capacities.data()),/**/
					thrust::raw_pointer_cast(indices.data()),/**/
					thrust::raw_pointer_cast(current_ctx.m_d_draw_types.data()),/**/
					thrust::raw_pointer_cast(current_ctx.m_d_prefixes.data()));/**/
				CHECK_CUDA_ERROR();
				break;
			}
			case MT_PROBAB_VOXELMAP:
			{
				fill_vbo_without_precounting<<<context.num_blocks(), context.threads_per_block()>>>(
					/**/
					static_cast<ProbabilisticVoxel*>(context.m_voxelMap->getVoidDeviceDataPtr()),/**/
					context.m_voxelMap->getDimensions(),/**/
					current_ctx.m_dim_svoxel,/**/
					current_ctx.m_view_start_voxel_pos,/**/
					current_ctx.m_view_end_voxel_pos,/**/
					context.occupancy_threshold(),/**/
					vbo_ptr,/**/
					thrust::raw_pointer_cast(context.m_d_vbo_offsets.data()),/**/
					thrust::raw_pointer_cast(context.m_d_vbo_segment_voxel_capacities.data()),/**/
					thrust::raw_pointer_cast(indices.data()),/**/
					thrust::raw_pointer_cast(current_ctx.m_d_draw_types.data()),/**/
					thrust::raw_pointer_cast(current_ctx.m_d_prefixes.data()));/**/
				CHECK_CUDA_ERROR();
				break;
			}
			case MT_DISTANCE_VOXELMAP:
			{
				fill_vbo_without_precounting<<<context.num_blocks(), context.threads_per_block()>>>(
					/**/
					static_cast<DistanceVoxel*>(context.m_voxelMap->getVoidDeviceDataPtr()),/**/
					context.m_voxelMap->getDimensions(),/**/
					current_ctx.m_dim_svoxel,/**/
					current_ctx.m_view_start_voxel_pos,/**/
					current_ctx.m_view_end_voxel_pos,/**/
					static_cast<visualizer_distance_drawmodes>(current_ctx.m_distance_drawmode),/**/
					vbo_ptr,/*TODO: if there is a way to pass GL_RGBA color info to OpenGL, generate those colors here too? would need to register and map additional cuda resource*/
					thrust::raw_pointer_cast(context.m_d_vbo_offsets.data()),/**/
					thrust::raw_pointer_cast(context.m_d_vbo_segment_voxel_capacities.data()),/**/
					thrust::raw_pointer_cast(indices.data()),/**/
					thrust::raw_pointer_cast(current_ctx.m_d_draw_types.data()),/**/
					thrust::raw_pointer_cast(current_ctx.m_d_prefixes.data()));/**/
				CHECK_CUDA_ERROR();
				break;
			}
			default:
			{
				LOGGING_ERROR_C(Visualization, Visualizer,
					"No implementation to fill a voxel map of this type!" << endl);
				exit(EXIT_FAILURE);
			}
			}
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, context.cuda_ressource(), nullptr));

			return indices;
		}

		void fillGLBufferWithCubelistWOUpdate(VisualizerContext& current_ctx, CubelistContext& context, uint32_t index)
		{
			thrust::device_vector<uint32_t> indices(context.voxel_types(), 0);

			float4* vbo_ptr;
			size_t num_bytes;
			HANDLE_CUDA_ERROR(cudaGraphicsMapResources(1, context.cuda_ressource(), nullptr));
			HANDLE_CUDA_ERROR(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&vbo_ptr), &num_bytes, *context.cuda_ressource()));

			// Launch kernel to copy data into the OpenGL buffer.
			fill_vbo_with_cubelist_host(context.num_blocks(), context.threads_per_block(),
				/**/
				context.getCubesDevicePointer(),/**/
				context.getNumberOfCubes(),/**/
				vbo_ptr,/**/
				thrust::raw_pointer_cast(context.m_d_vbo_offsets.data()),/**/
				thrust::raw_pointer_cast(indices.data()),/**/
				thrust::raw_pointer_cast(current_ctx.m_d_draw_types.data()),/**/
				thrust::raw_pointer_cast(current_ctx.m_d_prefixes.data()));/**/
			CHECK_CUDA_ERROR();

			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
			HANDLE_CUDA_ERROR(cudaGraphicsUnmapResources(1, context.cuda_ressource(), nullptr));
		}

	}
}
