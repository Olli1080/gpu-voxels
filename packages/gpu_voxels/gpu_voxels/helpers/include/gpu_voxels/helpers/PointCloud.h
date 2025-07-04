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
 * \author  Herbert Pietrzyk
 * \date    2016-05-25
 *
 * Class holding and manipulating PointClouds
 * as Arrays of Vector3fs on the GPU.
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_POINTCLOUD_H_INCLUDED
#define GPU_VOXELS_HELPERS_POINTCLOUD_H_INCLUDED

#include <cstdint> // for fixed size datatypes
#include <filesystem>
#include <vector>

#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <thrust/device_vector.h>

namespace gpu_voxels
{
	class PointCloud
	{
	public:

		/*!
		 * \brief PointCloud Constructs empty pointcloud
		 */
		PointCloud() = default;
		/*!
		 * \brief PointCloud Constructor
		 * \param points Vector of points
		 */
		explicit PointCloud(const std::vector<Vector3f>& points);
		/*!
		 * \brief PointCloud Constructor
		 * \param points Host pointer to points
		 * \param size Number of points
		 */
		explicit PointCloud(const Vector3f* points, uint32_t size);

		/*!
		 * \brief PointCloud Constructor loading a file
		 * \param path_to_file Path to the file to load
		 * \param use_model_path If true, file is searched relative to GPU_VOXELS_MODEL_PATH. If false, an absolute path can be given.
		 * Defaults to true.
		 */
		explicit PointCloud(const std::string& path_to_file, const std::filesystem::path& model_path);


		// Deep Copy Operators
		explicit PointCloud(const PointCloud& other);
		PointCloud& operator=(const PointCloud& other);

		~PointCloud() = default;

		// Deep equality check
		bool operator==(const PointCloud& other) const;

		/*!
		 * \brief add Adds the points of the input pointcloud to the currently existing points
		 * \param cloud Pointcloud
		 */
		void add(const PointCloud& cloud);
		/*!
		 * \brief add Adds the points to the currently existing points
		 * \param points Vector of points
		 */
		void add(const std::vector<Vector3f>& points);
		/*!
		 * \brief add Adds the points to the currently existing points
		 * \param points Host pointer to points
		 * \param size Number of points
		 */
		void add(const thrust::host_vector<Vector3f>& points);

		/*!
		 * \brief update Replaces all points with the points given by the input cloud.
		 * \param cloud Pointcloud.
		 */
		void update(const PointCloud& cloud);
		/*!
		 * \brief update Replaces all points with the input points
		 * \param points Vector of points.
		 */
		void update(const std::vector<Vector3f>& points);
		/*!
		 * \brief update Replaces all points with the input points
		 * \param points Host pointer to points.
		 * \param size Number of points
		 */
		void update(const thrust::host_vector<Vector3f>& points);

		/*!
		 * \brief transformSelf Applies the transformation to the own points and overrides the points
		 * \param transform Transformation matrix
		 */
		void transformSelf(const Matrix4f& transform);
		/*!
		 * \brief transform Applies a transformation to this cloud and returns the result in transformed_cloud
		 * \param transform Transformation matrix
		 * \param transformed_cloud Output cloud. Will be resized.
		 */
		void transform(const Matrix4f& transform, PointCloud& transformed_cloud) const;

		/*!
		 * \brief scaleSelf Applies scaling around origin to the own points and overrides the points
		 * \param scaling The scaling factors.
		 */
		void scaleSelf(const Vector3f& scaling);
		/*!
		 * \brief scale Applies scaling around origin to this cloud and returns the result in transformed_cloud
		 * \param scaling The scaling factors.
		 * \param scaled_cloud Output cloud. Will be resized.
		 */
		void scale(const Vector3f& scaling, PointCloud& scaled_cloud) const;

		thrust::device_vector<Vector3f>& getPointsDevice();
		const thrust::device_vector<Vector3f>& getPointsDevice() const;
		uint32_t getPointCloudSize() const;
		thrust::host_vector<Vector3f> getPoints() const;

		//for testing
		void print() const;

	private:

		//! Only allocates memory
		void resize(uint32_t new_number_of_points);

		thrust::device_vector<Vector3f> m_points_dev;
	};

}//end namespace gpu_voxels
#endif // GPU_VOXELS_HELPERS_POINTCLOUD_H_INCLUDED