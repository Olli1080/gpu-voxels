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
 * \date    2014-07-04
 *
 */
 //----------------------------------------------------------------------/*
#ifndef  GPU_VOXELS_OCTREE_GVL_NTREE_H_INCLUDED
#define  GPU_VOXELS_OCTREE_GVL_NTREE_H_INCLUDED

#include <gpu_voxels/helpers/ThrustForward.h>

#include <gpu_voxels/GpuVoxelsMap.h>
#include <gpu_voxels/octree/NTree.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/octree/Sensor.h>

namespace gpu_voxels {
	namespace NTree {

		template<std::size_t branching_factor, std::size_t level_count, typename InnerNode, typename LeafNode>
		class GvlNTree : public GpuVoxelsMap, public NTree<branching_factor, level_count, InnerNode, LeafNode>,
			public CollidableWithBitVectorVoxelMap, public CollidableWithBitVectorVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>, public CollidableWithBitVectorOctree,
			public CollidableWithBitVectorMortonVoxelList, public CollidableWithProbVoxelMap, public CollidableWithProbOctree,
			public CollidableWithResolutionBitVectorVoxelMap, public CollidableWithResolutionBitVectorVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>, public CollidableWithResolutionBitVectorOctree,
			public CollidableWithResolutionBitVectorMortonVoxelList, public CollidableWithResolutionProbVoxelMap, public CollidableWithResolutionProbOctree,
			public CollidableWithTypesBitVectorVoxelMap, public CollidableWithTypesBitVectorVoxelList<BIT_VECTOR_LENGTH, MapVoxelID>
		{
		public:
			typedef NTree<branching_factor, level_count, InnerNode, LeafNode> base;

			GvlNTree(float voxel_side_length, MapType map_type);
			~GvlNTree() override;

			// ------ START Global API functions ------

			void insertPointCloud(const std::vector<Vector3f>& point_cloud, BitVoxelMeaning voxel_meaning) override;

			void insertPointCloud(const PointCloud& pointcloud, BitVoxelMeaning voxel_meaning) override;

			void insertCoordinateList(const std::vector<Vector3ui>& coordinates, BitVoxelMeaning voxel_meaning) override;

			void insertCoordinateList(const ThrustDeviceVector<Vector3ui>& d_coordinates, BitVoxelMeaning voxel_meaning) override;

			void insertMetaPointCloud(const MetaPointCloud& meta_point_cloud, BitVoxelMeaning voxelType) override;

			void insertMetaPointCloud(const MetaPointCloud& meta_point_cloud, const std::vector<BitVoxelMeaning>& voxel_meanings) override;

			void clearMap() override;

			void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning) override;

			virtual size_t collideWithBitcheck(GpuVoxelsMapSharedPtr other, uint8_t margin = 0, const Vector3i& offset = Vector3i::Zero());

			bool insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud* robot_links,
				const std::vector<BitVoxelMeaning>& voxel_meanings = {},
				const std::vector<BitVector<BIT_VECTOR_LENGTH>>& collision_masks = {},
				BitVector<BIT_VECTOR_LENGTH>* colliding_meanings = nullptr) override;

			bool merge(GpuVoxelsMapSharedPtr other, const Vector3f& metric_offset = Vector3f::Zero(), const BitVoxelMeaning* new_meaning = nullptr) override;
			bool merge(GpuVoxelsMapSharedPtr other, const Vector3i& voxel_offset = Vector3i::Zero(), const BitVoxelMeaning* new_meaning = nullptr) override;

			std::size_t getMemoryUsage() const override;

			bool writeToDisk(std::string path) override;

			bool readFromDisk(std::string path) override;

			bool needsRebuild() const override;

			bool rebuild() override;

			Vector3ui getDimensions() const override;

			Vector3f getMetricDimensions() const override;

			// ------ END Global API functions ------

			//Collision Interface
			size_t collideWith(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWith(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWith(const voxellist::BitVectorVoxelList* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWith(const GvlNTreeDet* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWith(const GvlNTreeProb* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWith(const voxellist::BitVectorMortonVoxelList* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;

			size_t collideWithResolution(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.0, uint32_t resolution_level = 0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithResolution(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.0, uint32_t resolution_level = 0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithResolution(const voxellist::BitVectorVoxelList* map, float coll_threshold = 1.0, uint32_t resolution_level = 0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithResolution(const GvlNTreeDet* map, float coll_threshold = 1.0, uint32_t resolution_level = 0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithResolution(const GvlNTreeProb* map, float coll_threshold = 1.0, uint32_t resolution_level = 0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithResolution(const voxellist::BitVectorMortonVoxelList* map, float coll_threshold = 1.f, uint32_t resolution_level = 0, const Vector3i& offset = Vector3i::Zero()) override;

			size_t collideWithTypes(const voxelmap::BitVectorVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithTypes(const voxellist::BitVectorVoxelList* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;

			/*!
			 * \brief insertPointCloudWithFreespaceCalculation Performs raycasting to explicitly mark cells between sensor and measurement as free.
			 * \param point_cloud_in_sensor_coords Input pointcloud in given wrt sensor coordinate system
			 * \param sensor_pose pose of sensor during capturing the data. Only position is evaluated, not orientation.
			 * \param free_space_resolution can be used to decrease calculation accuracy and save time
			 * \param occupied_space_resolution can be used to decrease calculation accuracy and save time
			 */
			void insertPointCloudWithFreespaceCalculation(const std::vector<Vector3f>& point_cloud_in_sensor_coords, const Matrix4f& sensor_pose,
				uint32_t free_space_resolution, uint32_t occupied_space_resolution);


			/*!
			 * \brief collideWithTypesConsideringUnknownCells This does a collision check with 'other' and delivers the voxel meanings that are in collision.
			 * Same as \code collideWithTypes \endcode but this considers 'unknown' octree cells as collisions.
			 * This is especially useful when colliding swept volumes and one wants to know, which subvolumes lie in collision.
			 * Only available for checks against BitVoxel-Types!
			 * \param other The map to do a collision check with.
			 * \param meanings_in_collision The voxel meanings in collision.
			 * \param coll_threshold The threshold when a collision is counted. Only valid for probabilistic maps.
			 * \param offset The offset in cell coordinates
			 * \return The severity of the collision, namely the number of voxels that lie in collision
			 */
			size_t collideWithTypesConsideringUnknownCells(GpuVoxelsMapSharedPtr other, BitVectorVoxel& types_in_collision, size_t& num_colls_with_unknown_cells,
				const Vector3i& offset = Vector3i::Zero());

		protected:
			virtual void insertVoxelData(const ThrustDeviceVector<Vector3ui>& d_voxels);

		private:
			Sensor m_sensor;

			ThrustDeviceVector<Voxel>* m_d_free_space_voxel2;
			ThrustDeviceVector<Voxel>* m_d_object_voxel2;

		};

	}  // end of ns
}  // end of ns

#endif /* GPUVOXELSNTREE_H_ */