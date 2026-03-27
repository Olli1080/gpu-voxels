// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2013-12-11
 *
 */
 //----------------------------------------------------------------------/*
#include "RobotNodes.h"
#include "EnvironmentNodes.h"

namespace gpu_voxels {
	namespace NTree {
		namespace Robot {

#ifdef DISABLE_SEPARATE_COMPILTION
			GVL_HOST_DEVICE
				bool LeafNode::isInConflict(Environment::LeafNode env_LeafNode)
			{
				return isOccupied() & env_LeafNode.isOccupied();
			}

			GVL_HOST_DEVICE
				bool InnerNode::isInConflict(Environment::InnerNode env_InnerNode)
			{
				return isOccupied() & env_InnerNode.isOccupied();
			}
#endif

		} // end of ns
	} // end of ns
} // end of ns
