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
 * \author  Andreas Hermann
 * \date    2014-06-12
 *
 */
//----------------------------------------------------------------------
#include <gpu_voxels/helpers/PcdFileReader.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <gpu_voxels/logging/logging_gpu_voxels_helpers.h>
#include <gpu_voxels/helpers/common_defines.h>

namespace gpu_voxels
{
    namespace file_handling
	{

        bool PcdFileReader::readPointCloud(const std::filesystem::path& filename, std::vector<Vector3f>& points)
        {
	        const pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

            if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename.string(), *cloud) == -1) //* load the file
            {
                LOGGING_ERROR(Gpu_voxels_helpers, "Could not open file " << filename.c_str() << " !" << endl);
                return false;
            }

            points.reserve(cloud->points.size());
            for (const auto& point : cloud->points)
                points.emplace_back(point.x, point.y, point.z);

            LOGGING_DEBUG(
                Gpu_voxels_helpers,
                "PCD Handler: loaded " << points.size() << " points (" << (points.size() * sizeof(Vector3f)) * cBYTE2MBYTE << " MB on CPU) from " << filename.c_str() << "." << endl);
            return true;
        }

    } // end of namespace
} // end of namespace