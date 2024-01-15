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
 * \date    2014-07-10
 *
 */
//----------------------------------------------------------------------

#include "gpu_voxels/helpers/common_defines.h"
#include "gpu_voxels/helpers/PointcloudFileHandler.h"

namespace gpu_voxels
{
    namespace file_handling
	{
        PointcloudFileHandler* PointcloudFileHandler::Instance()
        {
            static PointcloudFileHandler instance;
            return &instance;
        }

        PointcloudFileHandler::PointcloudFileHandler()
        {
            xyz_reader = std::make_unique<XyzFileReader>();
            binvox_reader = std::make_unique<BinvoxFileReader>();
#ifdef _BUILD_GVL_WITH_PCL_SUPPORT_
            pcd_reader = std::make_unique<PcdFileReader>();
#endif
        }

        PointcloudFileHandler::~PointcloudFileHandler() = default;

		/*!
		 * \brief loadPointCloud loads a PCD file and returns the points in a vector.
		 * \param path Filename
		 * \param points points are written into this vector
		 * \param shift_to_zero If true, the pointcloud is shifted, so its minimum coordinates lie at zero
		 * \param offset_XYZ Additional transformation offset
		 * \return true if succeeded, false otherwise
		 */
		bool PointcloudFileHandler::loadPointCloud(const std::string& _path, const bool use_model_path, std::vector<Vector3f>& points, const bool shift_to_zero,
			const Vector3f& offset_XYZ, const float scaling) const
		{
			// if param is true, prepend the environment variable GPU_VOXELS_MODEL_PATH
			std::string path = (getGpuVoxelsPath(use_model_path) / std::filesystem::path(_path)).string();

			LOGGING_DEBUG_C(
				Gpu_voxels_helpers,
				GpuVoxelsMap,
				"Loading Pointcloud file " << path << " ..." << endl);

			// is the file a simple xyz file?
			if (path.ends_with("xyz"))
			{
				if (!xyz_reader->readPointCloud(path, points))
					return false;

			} // is the file a simple pcl pcd file?
			else if (path.ends_with("pcd"))
			{
#ifdef _BUILD_GVL_WITH_PCL_SUPPORT_
				if (!pcd_reader->readPointCloud(path, points))
					return false;
#else
				LOGGING_ERROR_C(
					Gpu_voxels_helpers,
					GpuVoxelsMap,
					"Your GPU-Voxels was built without PCD support!" << endl);
				return false;
#endif

			} // is the file a binvox file?
			else if (path.ends_with("binvox"))
			{
				if (!binvox_reader->readPointCloud(path, points))
					return false;
			}
			else 
            {
				LOGGING_ERROR_C(
					Gpu_voxels_helpers,
					GpuVoxelsMap,
					path << " has no known file format." << endl);
				return false;
			}

			if (shift_to_zero)
				shiftPointCloudToZero(points);

			for (auto& point : points)
				point = (scaling * point) + offset_XYZ;

			return true;
		}

        /*!
         * \brief centerPointCloud Centers a pointcloud relative to its maximum coordinates
         * \param points Working cloud
         */
        void PointcloudFileHandler::centerPointCloud(std::vector<Vector3f>& points) const
        {
            Vector3f min_xyz = points[0];
            Vector3f max_xyz = points[0];

            for (size_t i = 1; i < points.size(); i++)
            {
                min_xyz = min_xyz.cwiseMin(points[i]);
                max_xyz = max_xyz.cwiseMax(points[i]);
            }

            const Vector3f center_offset_xyz = (min_xyz + max_xyz) / 2.f;

            for (auto& point : points)
                point -= center_offset_xyz;
        }

        /*!
         * \brief shiftPointCloudToZero Moves a pointcloud, so that its minimum coordinates are shifted to zero.
         * \param points Working cloud
         */
        void PointcloudFileHandler::shiftPointCloudToZero(std::vector<Vector3f>& points) const
        {
            Vector3f min_xyz = points[0];

            for (size_t i = 1; i < points.size(); i++)
                min_xyz = min_xyz.cwiseMin(points[i]);

            for (auto& point : points)
                point -= min_xyz;
        }

    }  // end of ns
}  // end of ns