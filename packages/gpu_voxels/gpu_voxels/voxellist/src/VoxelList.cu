// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
*
* \author  Felix Mauch <mauch@fzi.de>
* \date    2015-05-04
*
*/
//----------------------------------------------------------------------

#include "VoxelList.h"
//#include <gpu_voxels/voxelmap/VoxelMap.hpp>

namespace gpu_voxels {

	//template class BitVector<BIT_VECTOR_LENGTH>;
	//template class BitVoxel<BIT_VECTOR_LENGTH>;


	namespace voxellist {

		// ############################### BitVoxelList ######################################
		// Explicit instantiation of template class to link against from other files where this template is used
		class CountingVoxelList;

		/*template size_t BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>::collideWithTypeMask(
			const TemplateVoxelMap<BitVoxel<BIT_VECTOR_LENGTH>>*,
			const BitVectorVoxel&, float, const Vector3i&);
		template size_t BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>::collideWithTypeMask(
			const TemplateVoxelMap<ProbabilisticVoxel>*,
			const BitVectorVoxel&, float, const Vector3i&);*/

		// ##################################################################################

		// ############################### TemplateVoxelList ######################################
		// Explicitly instantiate template methods to enable GCC to link agains NVCC compiled objects
		


		//template size_t TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::collideCountingPerMeaning(const GpuVoxelsMapSharedPtr, std::vector<size_t>&, const Vector3i&);

		//virtual bool subtract(const GpuVoxelsMapSharedPtr other, const Vector3i &voxel_offset = Vector3i::Zero());

		// ############################### ProbVoxelMap (inherits from TemplateVoxelMap) ######################################
		// Explicitly instantiate template methods to enable GCC to link agains NVCC compiled objects
		//template void ProbVoxelMap::insertSensorData<BIT_VECTOR_LENGTH>(const Vector3f*, const bool, const bool,
		//                                                                const uint32_t, BitVoxel<BIT_VECTOR_LENGTH>*);


		// ##################################################################################

	} // end of namespace voxellist
} // end of namespace