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
 * \date    2013-12-11
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_ROBOTNODES_H_INCLUDED
#define GPU_VOXELS_OCTREE_ROBOTNODES_H_INCLUDED

#include <gpu_voxels/octree/Nodes.h>
#include <gpu_voxels/octree/DataTypes.h>
#include <gpu_voxels/helpers/BitVector.h>

namespace gpu_voxels {
	namespace NTree {

		namespace Environment {
			class LeafNode;
			class InnerNode;
		}

		namespace Robot {

			/*
			 * Data structure to store the occupancy information of different samples or swept volumes in a leaf node of a tree
			 */
			class LeafNode
			{
			public:
				// default constructor needed
				GVL_HOST_DEVICE __forceinline__ LeafNode()
				{

				}

				GVL_DEVICE __forceinline__
					void init_d()
				{
					m_voxel_meaning.clear();
				}

				GVL_DEVICE __forceinline__ void initLastLevel_d()
				{
					init_d();
				}

				GVL_HOST
					void init_h()
				{
					m_voxel_meaning.clear();
				}

				GVL_HOST
					void init_h(uint32_t level)
				{
					init_h();
				}

				GVL_HOST_DEVICE __forceinline__
					bool isOccupied()
				{
					return m_voxel_meaning.getBit(eBVM_OCCUPIED);
				}

				GVL_HOST_DEVICE __forceinline__
					void setOccupied(BitVoxelMeaning voxelMeaning)
				{
					m_voxel_meaning.setBit(voxelMeaning);
					m_voxel_meaning.setBit(eBVM_OCCUPIED);
				}

				GVL_HOST_DEVICE __forceinline__
					void setOccupied()
				{
					m_voxel_meaning.setBit(eBVM_OCCUPIED);
				}

				GVL_HOST_DEVICE __forceinline__
					void setChildPtr(void* child)
				{
					// dummy
				}

				GVL_HOST_DEVICE __forceinline__
					void* getChildPtr()
				{
					// function not supported by this type
					assert(0);
					return 0;
				}

				GVL_HOST_DEVICE __forceinline__
					void setFree(BitVoxelMeaning voxelMeaning)
				{
					m_voxel_meaning.setBit(voxelMeaning);
					m_voxel_meaning.setBit(eBVM_FREE);
				}

#ifdef DISABLE_SEPARATE_COMPILTION
				GVL_HOST_DEVICE
					bool isInConflict(Environment::LeafNode env_LeafNode);
#endif

				GVL_HOST_DEVICE __forceinline__
					bool isInConflict(Robot::LeafNode rob_LeafNode)
				{
					return isOccupied() & rob_LeafNode.isOccupied();
				}

			private:
				BitVector<BIT_VECTOR_LENGTH> m_voxel_meaning;
			}
			;

			class InnerNode
			{
			private:
				GVL_HOST_DEVICE __forceinline__
					void init(NodeStatus status)
				{
					m_child_low = 0;
					m_child_high = 0;
					m_status = status;
					alignment = 0;

				}

			public:

				// default constructor needed
				GVL_HOST_DEVICE __forceinline__ InnerNode()
				{

				}

				GVL_DEVICE __forceinline__
					void init_d()
				{
					init(ns_FREE);
					// Set all Bits
					m_voxel_meaning.clear();
					m_voxel_meaning = ~m_voxel_meaning;
				}

				GVL_DEVICE __forceinline__
					void initLastLevel_d()
				{
					init(NodeStatus(ns_FREE | ns_LAST_LEVEL));
					// Set all Bits
					m_voxel_meaning.clear();
					m_voxel_meaning = ~m_voxel_meaning;

				}

				GVL_HOST_DEVICE __forceinline__
					bool hasStatus(NodeStatus status)
				{
					return (getStatus() & status) > 0;
				}

				GVL_HOST_DEVICE __forceinline__
					bool isOccupied()
				{
					return hasStatus(NodeStatus(ns_OCCUPIED | ns_PART));
				}

				GVL_HOST_DEVICE __forceinline__
					void setOccupied()
				{
					// clear all flags except LAST_LEVEL and set PART_OCCUPIED
					setStatus(NodeStatus((getStatus() & ns_LAST_LEVEL) | ns_PART));
				}

				GVL_HOST_DEVICE __forceinline__
					NodeStatus getStatus()
				{
					return m_status;
				}

				GVL_HOST_DEVICE __forceinline__
					void setStatus(NodeStatus status)
				{
					assert(uint32_t(status) <= 0xFF);
					m_status = status;
				}

				GVL_HOST_DEVICE __forceinline__
					void* getChildPtr()
				{
					return (void*)((uint64_t(m_child_high) << (8 * sizeof(m_child_low))) | uint64_t(m_child_low));
				}

				GVL_HOST_DEVICE __forceinline__
					void setChildPtr(void* child)
				{
					assert(
						uint64_t(child) != 0
						&& uint64_t(child) < (uint64_t(1) << uint64_t(8 * (sizeof(m_child_high) + sizeof(m_child_low)))));
					m_child_high = (uint16_t)(uint64_t(child) >> (8 * sizeof(m_child_low)));
					m_child_low = (uint32_t)(uint64_t(child) & 0xFFFFFFFF);
				}

#ifdef DISABLE_SEPARATE_COMPILTION
				GVL_HOST_DEVICE
					bool isInConflict(Environment::InnerNode env_InnerNode);
#endif

				GVL_HOST_DEVICE __forceinline__
					bool isInConflict(Robot::InnerNode rob_InnerNode)
				{
					return isOccupied() & rob_InnerNode.isOccupied();
				}

			private:
				uint32_t m_child_low;
				uint16_t m_child_high;
				uint8_t m_status;
				uint16_t alignment;
				BitVector<BIT_VECTOR_LENGTH> m_voxel_meaning;
			};
		}

	} // end of ns
} // end of ns

#endif /* ROBOTNODES_H_ */