#include "TemplateVoxelList.hpp"

namespace gpu_voxels
{
	namespace voxellist
	{
		template<>
		bool TemplateVoxelList<CountingVoxel, MapVoxelID>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning)
		{
			switch (other->getMapType())
			{
			case MT_BITVECTOR_VOXELLIST:
			{
				auto* m = other->as<BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>>();
				std::scoped_lock lock(this->m_mutex, m->m_mutex);

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = static_cast<uint32_t>(m_dev_list.size());
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();

				return true;
			}

			case MT_COUNTING_VOXELLIST:
			{
				auto* m = other->as<CountingVoxelList>();
				std::scoped_lock lock(this->m_mutex, m->m_mutex);

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = static_cast<uint32_t>(m_dev_list.size());
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.begin(), m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.end(), m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_list.begin() + offset_new_entries, m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();

				return true;
			}
			default:
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
				return false;
			}
			}
		}

		template<>
		bool TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::merge(const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning)
		{
			std::scoped_lock lock(this->m_mutex, other->m_mutex);

			switch (other->getMapType())
			{
			case MT_BITVECTOR_VOXELLIST:
			{
				auto* m = other->as<BitVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>>();

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = static_cast<uint32_t>(m_dev_list.size());
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.begin(), m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_list.end(), m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_list.begin() + offset_new_entries, m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				// if a new meaning was given, iterate over the voxellist and overwrite the meaning
				if (new_meaning)
				{
					BitVectorVoxel fillVoxel;
					fillVoxel.bitVector().setBit(*new_meaning);
					thrust::fill(m_dev_list.begin() + offset_new_entries, m_dev_list.end(), fillVoxel);
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();

				return true;
			}
			case MT_COUNTING_VOXELLIST:
			{
				auto* m = other->as<CountingVoxelList>();

				const uint32_t num_new_voxels = m->getDimensions().x();
				const uint32_t offset_new_entries = static_cast<uint32_t>(m_dev_list.size());
				// resize capacity
				this->resize(offset_new_entries + num_new_voxels);

				// We append the given list to our own list of points.
				thrust::copy(
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.begin(), m->m_dev_id_list.begin())),
					thrust::make_zip_iterator(thrust::make_tuple(m->m_dev_coord_list.end(), m->m_dev_id_list.end())),
					thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				BitVectorVoxel fillVoxel;
				if (new_meaning)
				{
					fillVoxel.bitVector().setBit(*new_meaning);
				}
				// iterate over the voxellist and overwrite the meaning
				thrust::fill(m_dev_list.begin() + offset_new_entries, m_dev_list.end(), fillVoxel);
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				make_unique();

				return true;
			}
			case MT_BITVECTOR_VOXELMAP:
			{
				auto m = other->as<voxelmap::BitVectorVoxelMap>();

				const size_t offset_new_entries = m_dev_list.size();

				// Resize list to add space for new voxels
				const size_t num_new_entries = thrust::count_if(
					thrust::device_ptr<BitVectorVoxel>(m->getDeviceDataPtr()),
					thrust::device_ptr<BitVectorVoxel>(m->getDeviceDataPtr() + m->getVoxelMapSize()),
					is_occupied<BitVectorVoxel>(0.0f)); // Threshold doesn't matter for BitVectors
				this->resize(offset_new_entries + num_new_entries);

				// Fill MapVoxelIDs of occupied voxels into end of m_dev_id_list
				thrust::copy_if(
					thrust::counting_iterator<MapVoxelID>(0),                     // src.begin
					thrust::counting_iterator<MapVoxelID>(m->getVoxelMapSize()),  // src.end
					thrust::device_ptr<BitVectorVoxel>(m->getDeviceDataPtr()),    // stencil.begin (predicate is used here)
					m_dev_id_list.begin() + offset_new_entries,                   // dest.begin
					is_occupied<BitVectorVoxel>(0.0f)); // Threshold doesn't matter for BitVectors
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// Fill m_dev_coord_list and m_dev_list by transforming added values in m_dev_id_list
				thrust::transform(
					m_dev_id_list.begin() + offset_new_entries,
					m_dev_id_list.end(),
					m_dev_coord_list.begin() + offset_new_entries,
					voxelid_to_voxelcoord<BitVectorVoxel>(m->getDeviceDataPtr(), m->getDimensions()));
				thrust::transform(
					m_dev_id_list.begin() + offset_new_entries,
					m_dev_id_list.end(),
					m_dev_list.begin() + offset_new_entries,
					voxelid_to_voxel<BitVectorVoxel>(m->getDeviceDataPtr()));
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// If an offset was given, we have to alter the newly added voxels.
				if (voxel_offset != Vector3i::Zero())
				{
					thrust::transform(
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.end(), m_dev_id_list.end())),
						thrust::make_zip_iterator(thrust::make_tuple(m_dev_coord_list.begin() + offset_new_entries, m_dev_id_list.begin() + offset_new_entries)),
						applyOffsetOperator<MapVoxelID>(m_ref_map_dim, voxel_offset));
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				// if a new meaning was given, iterate over the voxellist and overwrite the meaning
				if (new_meaning)
				{
					BitVectorVoxel fillVoxel;
					fillVoxel.bitVector().setBit(*new_meaning);
					thrust::fill(m_dev_list.begin() + offset_new_entries, m_dev_list.end(), fillVoxel);
					HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
				}

				make_unique();
				return true;
			}
			default:
			{
				LOGGING_ERROR_C(VoxellistLog, TemplateVoxelList, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
				return false;
			}
			}
		}

		//template TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::keyCoordVoxelZipIterator TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::getBeginTripleZipIterator();
		//template TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::keyCoordVoxelZipIterator TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::getEndTripleZipIterator();

		template class TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>;
		template class TemplateVoxelList<CountingVoxel, MapVoxelID>;
		template class TemplateVoxelList<ProbabilisticVoxel, MapVoxelID>;
		//template class TemplateVoxelList<DistanceVoxel, MapVoxelID>;

		template size_t TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::collideVoxellists(const TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>* other, const Vector3i& offset, thrust::device_vector<bool>& collision_stencil) const;
		template size_t TemplateVoxelList<CountingVoxel, MapVoxelID>::collideVoxellists(const TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>* other, const Vector3i& offset, thrust::device_vector<bool>& collision_stencil) const;
		template size_t TemplateVoxelList<ProbabilisticVoxel, MapVoxelID>::collideVoxellists(const TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>* other, const Vector3i& offset, thrust::device_vector<bool>& collision_stencil) const;

		template bool TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::equals(const TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>&) const;
		//template bool TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::subtract(const TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>*, const Vector3f&);
		//template void TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::screendump(bool) const;
		//template bool TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>::subtract(const TemplateVoxelList<ProbabilisticVoxel, MapVoxelID>*, const Vector3f&);

		//template size_t TemplateVoxelList<DistanceVoxel, MapVoxelID>::collideVoxellists(const TemplateVoxelList<BitVoxel<BIT_VECTOR_LENGTH>, MapVoxelID>* other, const Vector3i& offset, thrust::device_vector<bool>& collision_stencil) const;
	}
}