#pragma once

#include "TemplateVoxelList.h"

#include <thrust/device_vector.h>

#include <gpu_voxels/helpers/cuda_vectors.hpp>

namespace gpu_voxels {
	namespace voxellist {

		template<class Voxel, class VoxelIDType>
		struct TemplateVoxelList<Voxel, VoxelIDType>::CUDA_private
		{
			friend class CUDA_public;
			//! results of collision check on device
			thrust::device_vector<bool> m_dev_collision_check_results;

			//! result array for collision check with counter on device
			thrust::device_vector<uint16_t> m_dev_collision_check_results_counter;

			/* ======== Variables with content on device ======== */
			/* Follow the Thrust paradigm: Struct of Vectors */
			/* need to be public in order to be accessed by TemplateVoxelLists with other template arguments*/
			thrust::device_vector<VoxelIDType> m_dev_id_list;  // contains the voxel addresses / morton codes (This can not be a Voxel*, as Thrust can not sort pointers)
			thrust::device_vector<Vector3ui> m_dev_coord_list; // contains the voxel metric coordinates
			thrust::device_vector<Voxel> m_dev_list;           // contains the actual data: bitvector or probability
		};

		template<class Voxel, class VoxelIDType>
		class TemplateVoxelList<Voxel, VoxelIDType>::CUDA_public
		{
		public:

			CUDA_public(TemplateVoxelList& parent)
				: parent(parent), data(*parent.cuda_priv_impl)
			{}

			~CUDA_public() = default;

			typedef typename thrust::device_vector<VoxelIDType>::iterator  keyIterator;

			typedef typename thrust::device_vector<Vector3ui>::iterator  coordIterator;
			typedef typename thrust::device_vector<Voxel>::iterator  voxelIterator;

			typedef thrust::tuple<coordIterator, voxelIterator> valuesIteratorTuple;
			typedef thrust::zip_iterator<valuesIteratorTuple> zipValuesIterator;

			typedef thrust::tuple<keyIterator, voxelIterator> keyVoxelIteratorTuple;
			typedef thrust::zip_iterator<keyVoxelIteratorTuple> keyVoxelZipIterator;

			typedef thrust::tuple<keyIterator, coordIterator, voxelIterator> keyCoordVoxelIteratorTriple;
			typedef thrust::zip_iterator<keyCoordVoxelIteratorTriple> keyCoordVoxelZipIterator;


			//! get thrust triple to the beginning of all data vectors
			keyCoordVoxelZipIterator getBeginTripleZipIterator()
			{
				return thrust::make_zip_iterator(thrust::make_tuple(data.m_dev_id_list.begin(), data.m_dev_coord_list.begin(), data.m_dev_list.begin()));
			}

			//! get thrust triple to the end of all data vectors
			keyCoordVoxelZipIterator getEndTripleZipIterator()
			{
				return thrust::make_zip_iterator(thrust::make_tuple(data.m_dev_id_list.end(), data.m_dev_coord_list.end(), data.m_dev_list.end()));
			}

			//! get access to data vectors on device
			typename thrust::device_vector<Voxel>::iterator getDeviceDataVectorBeginning()
			{
				return data.m_dev_list.begin();
			}

			typename thrust::device_vector<VoxelIDType>::iterator getDeviceIdVectorBeginning()
			{
				return data.m_dev_id_list.begin();
			}

			typename thrust::device_vector<VoxelIDType>::iterator getDeviceIdVectorEnd()
			{
				return data.m_dev_id_list.end();
			}

			typename thrust::device_vector<VoxelIDType>::iterator getDeviceIdVectorBeginning() const
			{
				return data.m_dev_id_list.begin();
			}

			typename thrust::device_vector<VoxelIDType>::iterator getDeviceIdVectorEnd() const
			{
				return data.m_dev_id_list.end();
			}

			typename thrust::device_vector<Vector3ui>::iterator getDeviceCoordVectorBeginning()
			{
				return data.m_dev_coord_list.begin();
			}

		private:

			TemplateVoxelList& parent;
			CUDA_private& data;
		};
	}
}