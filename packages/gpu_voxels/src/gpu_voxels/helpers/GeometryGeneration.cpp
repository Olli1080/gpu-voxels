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
 * \date    2016-01-09
 *
 */
//----------------------------------------------------------------------

#include <gpu_voxels/helpers/GeometryGeneration.h>

namespace gpu_voxels
{
    namespace geometry_generation
    {
        template<typename T>
        class Range
        {
        public:

            Range(T start, T end, T spacing)
	            : start_(start), end_(end), spacing_(spacing)
            {
                if (start > end)
                    throw std::exception("Invalid range");

                size_ = (start <= end) ? 0 : static_cast<size_t>((end - start) / spacing);
            }

            class RangeIterator
            {
            public:

	            RangeIterator(const Range* instance, bool end = false)
		            : instance_(instance), current_((!end) ? instance->start_ : std::numeric_limits<T>::max())
	            {}

                RangeIterator operator++()
	            {
                    if (!done())
                        current_ += instance_->spacing_;
                        
                    return *this;
	            }

                bool operator!=(const RangeIterator& other) const
	            {
                    return (instance_ != other.instance_) || (current_ != other.current_ && done() != other.done());
	            }

                const T& operator*() const { return current_; }

                bool done() const { return current_ > instance_->end_; }

            private:

                const Range* instance_;
                T current_;
            };

            size_t size() const { return size_; }

            RangeIterator begin() const { return { this }; }
            RangeIterator end() const { return { this, true }; }

        private:

            T start_;
            T end_;
            T spacing_;

            size_t size_;
        };

        typedef Range<float> RangeF;
        typedef Range<uint32_t> RangeUI;

        void createOrientedBoxEdges(const OrientedBoxParams& params, float spacing, PointCloud& ret)
        {
            std::vector<Vector3f> cloud;

            const RangeF x_range(-params.dim.x(), params.dim.x(), spacing);
            const RangeF y_range(-params.dim.y(), params.dim.y(), spacing);
            const RangeF z_range(-params.dim.z(), params.dim.z(), spacing);

            cloud.reserve(4 * (x_range.size() + y_range.size() + z_range.size()));

            for (float x_dim : x_range)
            {
                cloud.emplace_back(x_dim, params.dim.y(), params.dim.z());
                cloud.emplace_back(x_dim, params.dim.y(), -params.dim.z());
                cloud.emplace_back(x_dim, -params.dim.y(), params.dim.z());
                cloud.emplace_back(x_dim, -params.dim.y(), -params.dim.z());
            }
            for (float y_dim : y_range)
            {
                cloud.emplace_back(params.dim.x(), y_dim, params.dim.z());
                cloud.emplace_back(params.dim.x(), y_dim, -params.dim.z());
                cloud.emplace_back(-params.dim.x(), y_dim, params.dim.z());
                cloud.emplace_back(-params.dim.x(), y_dim, -params.dim.z());
            }
            for (float z_dim : z_range)
            {
                cloud.emplace_back(params.dim.x(), params.dim.y(), z_dim);
                cloud.emplace_back(params.dim.x(), -params.dim.y(), z_dim);
                cloud.emplace_back(-params.dim.x(), params.dim.y(), z_dim);
                cloud.emplace_back(-params.dim.x(), -params.dim.y(), z_dim);
            }

            ret.update(cloud);

            const Matrix4f transformation = (Eigen::Translation3f(params.center) * createFromRPY(params.rot)).matrix();
            ret.transformSelf(transformation);
        }

        void createOrientedBox(const OrientedBoxParams& params, float spacing, PointCloud& ret)
        {
            std::vector<Vector3f> cloud;

            const RangeF x_range(-params.dim.x(), params.dim.x(), spacing);
            const RangeF y_range(-params.dim.y(), params.dim.y(), spacing);
            const RangeF z_range(-params.dim.z(), params.dim.z(), spacing);

            cloud.reserve(x_range.size() + y_range.size() + z_range.size());

            for (float x_dim : x_range)
            {
                for (float y_dim : y_range)
                {
                    for (float z_dim : z_range)
                    {
                        cloud.emplace_back(x_dim, y_dim, z_dim);
                    }
                }
            }

            ret.update(cloud);
            const Matrix4f transformation = (Eigen::Translation3f(params.center) * createFromRPY(params.rot)).matrix();
            ret.transformSelf(transformation);
        }


        std::vector<Vector3f> createBoxOfPoints(Vector3f min, Vector3f max, float delta)
        {
            std::vector<Vector3f> box_cloud;

            const RangeF x_range(min.x(), max.x(), delta);
            const RangeF y_range(min.y(), max.y(), delta);
            const RangeF z_range(min.z(), max.z(), delta);

            box_cloud.reserve(x_range.size() + y_range.size() + z_range.size());

            for (float x_dim : x_range)
            {
                for (float y_dim : y_range)
                {
                    for (float z_dim : z_range)
                    {
                        box_cloud.emplace_back(x_dim, y_dim, z_dim);
                    }
                }
            }
            return box_cloud;
        }

        std::vector<Vector3ui> createBoxOfPoints(Vector3f min, Vector3f max, float delta, float voxel_side_length)
        {
	        const Vector3f minCroped(min.x() >= 0.0f ? min.x() : 0.0f, min.y() >= 0.0f ? min.y() : 0.0f, min.z() >= 0.0f ? min.z() : 0.0f);
            const Vector3ui minimum = (minCroped / voxel_side_length).array().floor().matrix().cast<uint32_t>();
            const Vector3ui maximum = (max / voxel_side_length).array().ceil().matrix().cast<uint32_t>();

	        //const Vector3ui minimum(floor(minCroped.x() / voxel_side_length), floor(minCroped.y() / voxel_side_length), floor(minCroped.z() / voxel_side_length));
	        //const Vector3ui maximum(ceil(max.x() / voxel_side_length), ceil(max.y() / voxel_side_length), ceil(max.z() / voxel_side_length));
	        const uint32_t d = round(delta / voxel_side_length);

            std::vector<Vector3ui> box_coordinates;

            const RangeUI x_range(minimum.x(), maximum.x(), d);
            const RangeUI y_range(minimum.y(), maximum.y(), d);
            const RangeUI z_range(minimum.z(), maximum.z(), d);

            box_coordinates.reserve(x_range.size() + y_range.size() + z_range.size());

            for (uint32_t x_dim : x_range)
            {
                for (uint32_t y_dim : y_range)
                {
                    for (uint32_t z_dim : z_range)
                    {
                        box_coordinates.emplace_back(x_dim, y_dim, z_dim);
                    }
                }
            }
            return box_coordinates;
        }


        std::vector<Vector3f> createSphereOfPoints(Vector3f center, float radius, float delta)
        {
            std::vector<Vector3f> sphere_cloud;
            const Vector3f bbox_min(center - Vector3f::Constant(radius));
            const Vector3f bbox_max(center + Vector3f::Constant(radius));

            const RangeF x_range(bbox_min.x(), bbox_max.x(), delta);
            const RangeF y_range(bbox_min.y(), bbox_max.y(), delta);
            const RangeF z_range(bbox_min.z(), bbox_max.z(), delta);

            for (float x_dim : x_range)
            {
                for (float y_dim : y_range)
                {
                    for (float z_dim : z_range)
                    {
                        Vector3f point = Vector3f(x_dim, y_dim, z_dim);

                        if ((center - point).norm() <= radius)
                        {
                            sphere_cloud.emplace_back(point);
                        }
                    }
                }
            }
            return sphere_cloud;
        }

        std::vector<Vector3ui> createSphereOfPoints(Vector3f center, float radius, float delta, float voxel_side_length)
        {
            std::vector<Vector3ui> sphere_coordinates;
            const Vector3f bbox_min(center - Vector3f::Constant(radius));
            const Vector3f bbox_max(center + Vector3f::Constant(radius));

            const RangeF x_range(bbox_min.x(), bbox_max.x(), delta);
            const RangeF y_range(bbox_min.y(), bbox_max.y(), delta);
            const RangeF z_range(bbox_min.z(), bbox_max.z(), delta);

            for (float x_dim : x_range)
            {
                for (float y_dim : y_range)
                {
                    for (float z_dim : z_range)
                    {
                        Vector3f point = Vector3f(x_dim, y_dim, z_dim);

                        if ((center - point).norm() <= radius)
                        {
                            sphere_coordinates.emplace_back(
                                round(point.x() / voxel_side_length),
                                round(point.y() / voxel_side_length),
                                round(point.z() / voxel_side_length));
                        }
                    }
                }
            }
            return sphere_coordinates;
        }

        std::vector<Vector3f> createCylinderOfPoints(Vector3f center, float radius, float length_along_z, float delta)
        {
            std::vector<Vector3f> cylinder_cloud;
            const Vector3f bbox_min(center - Vector3f(radius, radius, length_along_z / 2.f));
            const Vector3f bbox_max(center + Vector3f(radius, radius, length_along_z / 2.f));

            const Range x_range(bbox_min.x(), bbox_max.x(), delta);
            const Range y_range(bbox_min.y(), bbox_max.y(), delta);
            const Range z_range(bbox_min.z(), bbox_max.z(), delta);
            
            for (float x_dim : x_range)
            {
                for (float y_dim : y_range)
                {
                    for (float z_dim : z_range)
                    {
                        Vector3f point = Vector3f(x_dim, y_dim, z_dim);

                        if ((center - point).norm() <= radius)
                        {
                            cylinder_cloud.push_back(point);
                        }
                    }
                }
            }
            return cylinder_cloud;
        }

        std::vector<Vector3ui> createCylinderOfPoints(Vector3f center, float radius, float length_along_z, float delta, float voxel_side_length)
        {
	        const Vector3f r(radius / voxel_side_length, radius / voxel_side_length, length_along_z / voxel_side_length / 2.f);
	        const Vector3f centerScaled(center.x() / voxel_side_length, center.y() / voxel_side_length, center.z() / voxel_side_length);
	        const Vector3ui minimum(floor(centerScaled.x() - r.x()), floor(centerScaled.y() - r.y()), floor(centerScaled.z() - r.z()));
	        const Vector3ui maximum(ceil(centerScaled.x() + r.x()), ceil(centerScaled.y() + r.y()), ceil(centerScaled.z() + r.z()));
	        const uint32_t d = round(delta / voxel_side_length);

	        const Vector3ui centerCoords(round(centerScaled.x()), round(centerScaled.y()), round(centerScaled.z()));
            std::vector<Vector3ui> cylinder_coordinates;

            const RangeUI x_range(minimum.x(), maximum.x(), d);
            const RangeUI y_range(minimum.y(), maximum.y(), d);
            const RangeUI z_range(minimum.z(), maximum.z(), d);

            for (uint32_t x_dim : x_range)
            {
                for (uint32_t y_dim : y_range)
                {
                    for (uint32_t z_dim : z_range)
                    {
                        Vector3ui point = Vector3ui(x_dim, y_dim, z_dim);

                        if ((centerCoords - point).norm() <= (radius / voxel_side_length))
                        {
                            cylinder_coordinates.push_back(point);
                        }
                    }
                }
            }
            return cylinder_coordinates;
        }

        void createEquidistantPointsInBox(const size_t max_nr_points,
            const Vector3ui max_coords,
            const float side_length,
            std::vector<Vector3f>& points)
        {
            uint32_t num_points = 0;
            for (uint32_t i = 0; i < (max_coords.x() - 1) / 2; i++)
            {
                for (uint32_t j = 0; j < (max_coords.y() - 1) / 2; j++)
                {
                    for (uint32_t k = 0; k < (max_coords.z() - 1) / 2; k++)
                    {
                        if (num_points >= max_nr_points)
                            return;

                        float x = i * 2 * side_length + side_length / 2.f;
                        float y = j * 2 * side_length + side_length / 2.f;
                        float z = k * 2 * side_length + side_length / 2.f;
                        points.emplace_back(x, y, z);
                        ++num_points;
                    }
                }
            }
        }


        void createNonOverlapping3dCheckerboard(const size_t max_nr_points,
            const Vector3ui max_coords,
            const float side_length,
            std::vector<Vector3f>& black_points,
            std::vector<Vector3f>& white_points)
        {
            uint32_t num_points = 0;
            for (uint32_t i = 0; i < (max_coords.x() - 1) / 2; i++)
            {
                for (uint32_t j = 0; j < (max_coords.y() - 1) / 2; j++)
                {
                    for (uint32_t k = 0; k < (max_coords.z() - 1) / 2; k++)
                    {
                        if (num_points >= max_nr_points)
                            return;

                        float x = i * 2 * side_length + side_length / 2.f;
                        float y = j * 2 * side_length + side_length / 2.f;
                        float z = k * 2 * side_length + side_length / 2.f;
                        black_points.emplace_back(x, y, z);
                        x = (i * 2 + 1) * side_length + side_length / 2.f;
                        y = (j * 2 + 1) * side_length + side_length / 2.f;
                        z = (k * 2 + 1) * side_length + side_length / 2.f;
                        white_points.emplace_back(x, y, z);
                        num_points++;
                    }
                }
            }
        }

    } // END OF NS gpu_voxels
} // END OF NS geometry_generation