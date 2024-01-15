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
 * \date    2012-09-14
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_KINECT_H_INCLUDED
#define GPU_VOXELS_HELPERS_KINECT_H_INCLUDED

#include <pcl/point_types.h>
#include <pcl/io/openni2_grabber.h>

#include <gpu_voxels/voxelmap/VoxelMap.h>

namespace gpu_voxels
{
	class Kinect
	{
	public:
		//! Constructor
		Kinect(std::string identifier = "");

		//! Destructor
		~Kinect();

		//! Create a OpenNI grabber and start capturing
		void run();

		//! Stop Capturing and delete the OpenNI grabber
		void stop();

		//! Information about the capturing state.
		[[nodiscard]] bool isRunning() const;

		//! Direct access to the stored data
		const std::vector<Vector3f>& getDataPtr();

	private:

		std::shared_ptr<pcl::Grabber> m_interface;
		std::vector<Vector3f> m_data;

		bool m_running;

		std::string m_identifier;

		// Callback triggered when new data is available
		void cloud_callback(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud);

	};
} // end of namespace
#endif