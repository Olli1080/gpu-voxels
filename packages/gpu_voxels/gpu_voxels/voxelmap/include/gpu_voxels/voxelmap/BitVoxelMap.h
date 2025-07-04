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
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VOXELMAP_BIT_VOXELMAP_H_INCLUDED
#define GPU_VOXELS_VOXELMAP_BIT_VOXELMAP_H_INCLUDED

#include <gpu_voxels/voxelmap/TemplateVoxelMap.h>
#include <gpu_voxels/voxel/BitVoxel.h>
#include <gpu_voxels/helpers/CollisionInterfaces.h>

namespace gpu_voxels {
	namespace voxelmap {

		template<std::size_t length>
		class BitVoxelMap : public TemplateVoxelMap<BitVoxel<length>>,
			public CollidableWithBitVectorVoxelMap, public CollidableWithProbVoxelMap, public CollidableWithTypesBitVectorVoxelMap, public CollidableWithTypesProbVoxelMap
		{
		public:
			typedef BitVoxel<length> Voxel;
			typedef TemplateVoxelMap<Voxel> Base;

			BitVoxelMap(Vector3ui dim, float voxel_side_length, MapType map_type);
			BitVoxelMap(Voxel* dev_data, Vector3ui dim, float voxel_side_length, MapType map_type);

			~BitVoxelMap() override;

			virtual void clearBit(uint32_t bit_index);

			virtual void clearBits(BitVector<length> bits);

			/**
			 * @brief Collides two Bit-Voxelmaps and delivers the Voxelmeanings that lie in collision, if those are set in both maps.
			 * \param other The map to collide with
			 * \param collider The collider kernel to use
			 * \param colliding_meanings The result vector in which the colliding meanings are set to 1
			 * \param sv_offset Offset which is added while checking to ignore the first bits
			 */
			template<class Collider>
			uint32_t collisionCheckBitvector(const BitVoxelMap<length>* other, Collider collider,
				BitVector<length>& colliding_meanings, uint16_t sv_offset = 0);

			/**
			 * @brief Collides the Bit-Voxelmap with a probabilistic map and delivers the Voxelmeanings that lie in collision, if probabilistic voxel is occupied.
			 * \param other The map to collide with
			 * \param collider The collider kernel to use
			 * \param colliding_meanings The result vector in which the colliding meanings are set to 1
			 * \param sv_offset Offset which is added while checking to ignore the first bits
			 */
			template<class Collider>
			uint32_t collisionCheckBitvector(const voxelmap::ProbVoxelMap* other, Collider collider,
				BitVector<length>& colliding_meanings, uint16_t sv_offset = 0);

			/**
			 * @brief Shifts all swept-volume-IDs by shift_size towards lower IDs.
			 * Currently this is limited to a shift size <64
			 * @param shift_size Shift size of bitshift
			 */
			void shiftLeftSweptVolumeIDs(uint8_t shift_size);


			bool insertMetaPointCloudWithSelfCollisionCheck(const MetaPointCloud* meta_point_cloud,
				const std::vector<BitVoxelMeaning>& voxel_meanings = {},
				const std::vector<BitVector<length> >& collision_masks = {},
				BitVector<length>* colliding_meanings = nullptr) override;

			void clearBitVoxelMeaning(BitVoxelMeaning voxel_meaning) override;

			[[nodiscard]] MapType getTemplateType() const override { return MT_BITVECTOR_VOXELMAP; }

			// Collision Interface
			size_t collideWith(const voxelmap::BitVectorVoxelMap* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWith(const voxelmap::ProbVoxelMap* map, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithTypes(const voxelmap::BitVectorVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;
			size_t collideWithTypes(const voxelmap::ProbVoxelMap* map, BitVectorVoxel& types_in_collision, float coll_threshold = 1.0, const Vector3i& offset = Vector3i::Zero()) override;

		protected:
			virtual void clearVoxelMapRemoteLock(uint32_t bit_index);

		private:

			//these are used for self collision checks:
			size_t m_num_self_collisions_checked_entities;
			BitVector<length>* m_selfcolliding_subclouds_dev;
			BitVector<length>* m_collisions_masks_dev;
			BitVoxelMeaning* m_subcloud_meanings_dev;
		};

	} // end of namespace
} // end of namespace

#endif