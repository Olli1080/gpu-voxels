// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-07-07
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_GVL_NTREE_HPP_INCLUDED
#define GPU_VOXELS_OCTREE_GVL_NTREE_HPP_INCLUDED

#include <gpu_voxels/helpers/PointCloud.h>

#include <gpu_voxels/octree/GvlNTree.h>
#include <gpu_voxels/octree/Octree.h>
#include <gpu_voxels/voxel/BitVoxel.h>

namespace gpu_voxels {
	namespace NTree {

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::GvlNTree(const float voxel_side_length, const MapType map_type) :
			base(NUM_BLOCKS, NUM_THREADS_PER_BLOCK, static_cast<uint32_t>(voxel_side_length * 1000.f))
		{
			this->m_map_type = map_type;

			m_d_free_space_voxel2 = nullptr;
			m_d_object_voxel2 = nullptr;

			// setup sensor parameter
			m_sensor.object_data.m_initial_probability = INITIAL_OCCUPIED_PROBABILITY;
			m_sensor.object_data.m_update_probability = OCCUPIED_UPDATE_PROBABILITY;
			m_sensor.object_data.m_invalid_measure = 0;
			m_sensor.object_data.m_remove_max_range_data = true;
			m_sensor.object_data.m_sensor_range = 7;
			m_sensor.object_data.m_use_invalid_measures = false;
			m_sensor.object_data.m_process_data = true;

			m_sensor.free_space_data = m_sensor.object_data; // copy data which doesn't matter

			// probabilities for free space aren't used for preprocessing of sensor data
			m_sensor.free_space_data.m_cut_x_boarder = KINECT_CUT_FREE_SPACE_X;
			m_sensor.free_space_data.m_cut_y_boarder = KINECT_CUT_FREE_SPACE_Y;
			m_sensor.free_space_data.m_invalid_measure = 0;
			m_sensor.free_space_data.m_remove_max_range_data = false;
			m_sensor.free_space_data.m_sensor_range = 7;
			m_sensor.free_space_data.m_use_invalid_measures = true;
			m_sensor.free_space_data.m_process_data = true; // parameter.compute_free_space;
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::~GvlNTree() = default;

		// ------ BEGIN Global API functions ------
		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertPointCloud(
			const std::vector<Vector3f>& point_cloud, BitVoxelMeaning voxelType)
		{
			std::lock_guard guard(this->m_mutex);
			if (voxelType != eBVM_OCCUPIED)
				LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
			else
			{
				// Copy points to gpu and transform to voxel coordinates
				thrust::host_vector<Vector3f> h_points(point_cloud.begin(), point_cloud.end());
				thrust::device_vector<Vector3ui> d_voxels;
				this->toVoxelCoordinates(h_points, d_voxels);

				insertVoxelData(d_voxels);
			}
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertPointCloud(const PointCloud& pointcloud, const BitVoxelMeaning voxel_meaning)
		{
			std::lock_guard guard(this->m_mutex);

			if (voxel_meaning != eBVM_OCCUPIED)
				LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
			else
			{
				thrust::device_vector<Vector3ui> d_voxels(pointcloud.getPointCloudSize());

				kernel_toVoxels<<<this->numBlocks, this->numThreadsPerBlock>>>(pointcloud.getPointsDevice().data().get(), pointcloud.getPointCloudSize(), D_PTR(d_voxels), static_cast<float>(this->m_resolution) / 1000.0f);
				CHECK_CUDA_ERROR();
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				insertVoxelData(d_voxels);
			}
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertCoordinateList(const std::vector<Vector3ui>& coordinates, const BitVoxelMeaning voxel_meaning)
		{
			std::lock_guard guard(this->m_mutex);
			if (voxel_meaning != eBVM_OCCUPIED)
				LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
			else
			{
				// Copy points to gpu and transform to voxel coordinates
				thrust::device_vector<Vector3ui> d_voxels = { coordinates.begin(), coordinates.end() };

				insertVoxelData(d_voxels);
			}
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertCoordinateList(const thrust::device_vector<Vector3ui>& d_coordinates, const BitVoxelMeaning voxel_meaning)
		{
			std::lock_guard guard(this->m_mutex);
			if (voxel_meaning != eBVM_OCCUPIED)
				LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
			else
			{
				insertVoxelData(d_coordinates);
			}
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertPointCloudWithFreespaceCalculation(
			const std::vector<Vector3f>& point_cloud_in_sensor_coords, const Matrix4f& sensor_pose,
			uint32_t free_space_resolution, uint32_t occupied_space_resolution)
		{
			std::lock_guard guard(this->m_mutex);
			m_sensor.free_space_data.m_voxel_side_length = free_space_resolution;
			m_sensor.object_data.m_voxel_side_length = occupied_space_resolution;

			m_sensor.pose = sensor_pose;

			m_sensor.data_width = static_cast<uint32_t>(point_cloud_in_sensor_coords.size());
			m_sensor.data_height = 1;

			// processSensorData() will allocate space for d_free_space_voxel and d_object_voxel if they are nullptr
			m_sensor.processSensorData(point_cloud_in_sensor_coords.data(), m_d_free_space_voxel2, m_d_object_voxel2);
			// convert sensor origin in discrete coordinates of the NTree
			gpu_voxels::Vector3ui sensor_origin = (sensor_pose.block<3, 1>(0, 3) * 1000.f / this->m_resolution).template cast<uint32_t>();

			this->insertVoxel(*m_d_free_space_voxel2, *m_d_object_voxel2, sensor_origin,
				free_space_resolution, occupied_space_resolution);

			//CAUTION: Check for needs_rebuild after inserting new data!
		}

		//Collision Interface Implementation
		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
			const voxelmap::BitVectorVoxelMap* map, float coll_threshold, const Vector3i& offset)
		{
			return collideWithResolution(map, coll_threshold, 0, offset);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
			const voxelmap::ProbVoxelMap* map, float coll_threshold, const Vector3i& offset)
		{
			return collideWithResolution(map, coll_threshold, 0, offset);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
			const voxellist::BitVectorVoxelList* map, float coll_threshold, const Vector3i& offset)
		{
			return collideWithResolution(map, coll_threshold, 0, offset);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
			const GvlNTreeDet* map, float coll_threshold, const Vector3i& offset)
		{
			return collideWithResolution(map, coll_threshold, 0, offset);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
			const GvlNTreeProb* map, float coll_threshold, const Vector3i& offset)
		{
			return collideWithResolution(map, coll_threshold, 0, offset);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWith(
			const voxellist::BitVectorMortonVoxelList* map, float coll_threshold, const Vector3i& offset)
		{
			return collideWithResolution(map, coll_threshold, 0, offset);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
			const voxelmap::BitVectorVoxelMap* map, float coll_threshold, const uint32_t resolution_level, const Vector3i& offset)
		{
			if (resolution_level >= level_count)
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" << endl);
				return (std::numeric_limits<size_t>::max)();
			}
			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			return this->template intersect_sparse<true, false, false, BitVectorVoxel>(*map, nullptr, 0, offset, nullptr);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
			const voxelmap::ProbVoxelMap* map, float coll_threshold, const uint32_t resolution_level, const Vector3i& offset)
		{
			if (resolution_level >= level_count)
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" << endl);
				return (std::numeric_limits<size_t>::max)();
			}
			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			return this->template intersect_sparse<true, false, false, ProbabilisticVoxel>(*map, nullptr, 0, offset, nullptr);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
			const voxellist::BitVectorVoxelList* map, float coll_threshold, const uint32_t resolution_level, const Vector3i& offset)
		{
			if (resolution_level >= level_count)
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" << endl);
				return (std::numeric_limits<size_t>::max)();
			}
			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			return this->template intersect_sparse<true, false, false, BitVectorVoxel>(*map, nullptr, 0, offset, nullptr);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
			const GvlNTreeDet* map, float coll_threshold, const uint32_t resolution_level, const Vector3i& offset)
		{
			if (resolution_level >= level_count)
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" << endl);
				return (std::numeric_limits<size_t>::max)();
			}
			if (offset != Vector3i::Zero())
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);

			constexpr bool save_collisions = true;

			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			return this->template intersect_load_balance<>(static_cast<const NTreeDet*>(map), resolution_level, DefaultCollider(), save_collisions);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
			const GvlNTreeProb* map, float coll_threshold, const uint32_t resolution_level, const Vector3i& offset)
		{
			if (resolution_level >= level_count)
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" << endl);
				return (std::numeric_limits<size_t>::max)();
			}

			if (offset != Vector3i::Zero())
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);

			constexpr bool save_collisions = true;

			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			return this->template intersect_load_balance<>(static_cast<const NTreeProb*>(map), resolution_level, DefaultCollider(), save_collisions);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithResolution(
			const voxellist::BitVectorMortonVoxelList* map, float coll_threshold, const uint32_t resolution_level, const Vector3i& offset)
		{
			if (resolution_level >= level_count)
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "resolution_level(" << resolution_level << ") greater than octree height!" << endl);
				return (std::numeric_limits<size_t>::max)();
			}
			
			if (offset != Vector3i::Zero())
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OFFSET_ON_WRONG_DATA_STRUCTURE << endl);

			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			// This previously used "intersect<BIT_VECTOR_LENGTH, true, false>" which itself utilized a more effective kernel.
			return this->template intersect_morton<true, false, false, BitVectorVoxel>(*map);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithTypes(
			const voxelmap::BitVectorVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold, const Vector3i& offset)
		{
			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			return this->template intersect_sparse<true, true, false, BitVectorVoxel>(*map, &types_in_collision, 0, offset, nullptr);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithTypes(
			const voxellist::BitVectorVoxelList* map, BitVectorVoxel& types_in_collision, float coll_threshold, const Vector3i& offset)
		{
			std::scoped_lock lock(this->m_mutex, map->m_mutex);
			return this->template intersect_sparse<true, true, false, BitVectorVoxel>(*map, &types_in_collision, 0, offset, nullptr);
		}


		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithTypesConsideringUnknownCells(
			const GpuVoxelsMapSharedPtr map, BitVectorVoxel& types_in_collision, size_t& num_colls_with_unknown_cells, const Vector3i& offset)
		{
			std::scoped_lock lock(this->m_mutex, map->m_mutex);

			size_t num_collisions = (std::numeric_limits<size_t>::max)();
			num_colls_with_unknown_cells = (std::numeric_limits<size_t>::max)();
			const MapType type = map->getMapType();
			voxel_count tmp_cols_w_unknown;

			if (type == MT_BITVECTOR_VOXELMAP)
			{
				auto* _voxelmap = dynamic_cast<const voxelmap::BitVectorVoxelMap*>(map.get());
				if (_voxelmap == nullptr)
					LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelMap' failed!" << endl);
				num_collisions = this->template intersect_sparse<true, true, true, BitVectorVoxel>(*_voxelmap, &types_in_collision, 0, offset, &tmp_cols_w_unknown);
				num_colls_with_unknown_cells = tmp_cols_w_unknown;
			}
			else if (type == MT_BITVECTOR_VOXELLIST)
			{
				auto* _voxellist = dynamic_cast<const voxellist::BitVectorVoxelList*>(map.get());
				if (_voxellist == nullptr)
					LOGGING_ERROR_C(OctreeLog, NTree, "dynamic_cast to 'BitVectorVoxelList' failed!" << endl);

				num_collisions = this->template intersect_sparse<true, true, true, BitVectorVoxel>(*_voxellist, &types_in_collision, 0, offset, &tmp_cols_w_unknown);
				num_colls_with_unknown_cells = tmp_cols_w_unknown;
			}
			else
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);

			return num_collisions;
		}


		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::collideWithBitcheck(
			const GpuVoxelsMapSharedPtr map, const uint8_t margin, const Vector3i& offset)
		{
			switch (map->getMapType())
			{
			case MT_BITVECTOR_VOXELMAP:
			{
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
				break;
			}
			case MT_BITVECTOR_OCTREE:
			{
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
				break;
			}
			default:
			{
				LOGGING_ERROR_C(VoxelmapLog, TemplateVoxelMap, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
				break;
			}
			}
			return (std::numeric_limits<size_t>::max)();
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloudWithSelfCollisionCheck(
			const MetaPointCloud* robot_links,
			const std::vector<BitVoxelMeaning>& voxel_meanings,
			const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks,
			BitVector<BIT_VECTOR_LENGTH>* colliding_meanings)
		{
			LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_SUPPORTED << endl);
			return true;
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloud(
			const MetaPointCloud& meta_point_cloud, BitVoxelMeaning voxelType)
		{
			std::lock_guard guard(this->m_mutex);
			if (voxelType != eBVM_OCCUPIED)
				LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);

			// Get address from device
			Vector3f* d_points = nullptr;
			MetaPointCloudStruct tmp_struct;
			HANDLE_CUDA_ERROR(cudaMemcpy(&tmp_struct, meta_point_cloud.getDeviceConstPointer().get(), sizeof(MetaPointCloudStruct), cudaMemcpyDeviceToHost));
			HANDLE_CUDA_ERROR(cudaMemcpy(&d_points, tmp_struct.clouds_base_addresses, sizeof(Vector3f*), cudaMemcpyDeviceToHost));

			const size_t num_points = meta_point_cloud.getAccumulatedPointcloudSize();
			thrust::device_vector<Vector3ui> d_voxels(num_points);
			kernel_toVoxels<<<this->numBlocks, this->numThreadsPerBlock>>>(d_points, num_points, D_PTR(d_voxels), this->m_resolution / 1000.0f);
			CHECK_CUDA_ERROR();
			HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

			insertVoxelData(d_voxels);
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertMetaPointCloud(
			const MetaPointCloud& meta_point_cloud, const std::vector<BitVoxelMeaning>& voxel_meanings)
		{
			/* Basically this is a dummy implementation since the method can't be left abstract.
			   However, I'm not sure whether this functionality makes sense here, so I didn't
			   implement it.
			 */
			LOGGING_WARNING_C(OctreeLog, NTree, "This functionality is not implemented, yet. The pointcloud will be inserted with the first BitVoxelMeaning." << endl);
			insertMetaPointCloud(meta_point_cloud, voxel_meanings.front());
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::merge(
			const GpuVoxelsMapSharedPtr other, const Vector3f& metric_offset, const BitVoxelMeaning* new_meaning)
		{
			LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
			return false;
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::merge(
			const GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset, const BitVoxelMeaning* new_meaning)
		{
			LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_OPERATION_NOT_YET_SUPPORTED << endl);
			return false;
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		std::size_t GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getMemoryUsage() const
		{
			std::lock_guard guard(this->m_mutex);
			return this->getMemUsage();
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::clearMap()
		{
			std::lock_guard guard(this->m_mutex);
			this->clear();
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning)
		{
			if (voxel_meaning != eBVM_OCCUPIED)
				LOGGING_ERROR_C(OctreeLog, NTree, GPU_VOXELS_MAP_ONLY_SUPPORTS_BVM_OCCUPIED << endl);
			else
				clearMap();
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::writeToDisk(const std::string path)
		{
			std::lock_guard guard(this->m_mutex);
			std::ofstream out(path.c_str());
			if (!out.is_open())
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "Write to file " << path << " failed!" << endl);
				return false;
			}
			this->serialize(out);
			out.close();
			return true;
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::readFromDisk(const std::string path)
		{
			std::lock_guard guard(this->m_mutex);
			std::ifstream in(path.c_str());
			if (!in.is_open())
			{
				LOGGING_ERROR_C(OctreeLog, NTree, "Read from file " << path << " failed!" << endl);
				return false;
			}
			this->deserialize(in);
			in.close();
			return true;
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::needsRebuild() const
		{
			std::lock_guard guard(this->m_mutex);
			return this->NTree<branching_factor, level_count, InnerNode, LeafNode>::needsRebuild();
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		bool GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::rebuild()
		{
			std::lock_guard guard(this->m_mutex);
			this->NTree<branching_factor, level_count, InnerNode, LeafNode>::rebuild();
			return true;
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		Vector3ui GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getDimensions() const
		{
			auto s = static_cast<uint32_t>(getVoxelSideLength<branching_factor>(level_count - 1));
			return { s, s, s };
		}

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		Vector3f GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::getMetricDimensions() const
		{
			Vector3ui dim_in_voxel = getDimensions();
			return Vector3f(dim_in_voxel.x(), dim_in_voxel.z(), dim_in_voxel.z()) * static_cast<float>(base::m_resolution) / 1000.0f;
		}

		// ------ END Global API functions ------

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		void GvlNTree<branching_factor, level_count, InnerNode, LeafNode>::insertVoxelData(
			const thrust::device_vector<Vector3ui>& d_voxels)
		{
			const uint32_t num_points = static_cast<uint32_t>(d_voxels.size());
			if (num_points <= 0)
				return;

			if (this->m_has_data)
			{
				// Have to insert voxels and adjust occupancy since there are already some voxels in the NTree
				// Transform voxel coordinates to morton code
				thrust::device_vector<OctreeVoxelID> d_voxels_morton(num_points);
				kernel_toMortonCode<<<this->numBlocks, this->numThreadsPerBlock>>>(D_PTR(d_voxels), num_points, D_PTR(d_voxels_morton));
				CHECK_CUDA_ERROR();
				HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

				// Sort and remove duplicates
				// TODO: remove thrust::unique() and adapt NTree::insert() to handle duplicates in the input data
				thrust::sort(d_voxels_morton.begin(), d_voxels_morton.end());
				const thrust::device_vector<OctreeVoxelID>::iterator new_end = thrust::unique(d_voxels_morton.begin(),
				                                                                              d_voxels_morton.end());
				size_t num_voxel_unique = new_end - d_voxels_morton.begin();

				// Insert voxels
				typename base::BasicData tmp;
				getHardInsertResetData(tmp);
				thrust::constant_iterator<typename base::BasicData> reset_data(tmp);
				getOccupiedData(tmp);
				thrust::constant_iterator<typename base::BasicData> set_basic_data(tmp);
				this->template insertVoxel<true, typename base::BasicData>(D_PTR(d_voxels_morton), set_basic_data, reset_data, num_voxel_unique, 0);

				// Recover tree invariant
				this->propagate(static_cast<uint32_t>(num_voxel_unique));
			}
			else
			{
				// Use plain this->build since NTree is empty
				this->build(d_voxels, false);
			}
		}

	}  // end of ns
}  // end of ns

#endif