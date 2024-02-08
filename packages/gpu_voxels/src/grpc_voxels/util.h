#pragma once

#include "Eigen/Dense"
#include "pcl/impl/point_types.hpp"
//#include "tiny_obj_loader.h"

//#include <state_observation/workspace_objects.hpp>

#include "depth_image.pb.h"
#include "vertex.pb.h"
#include "object_prototype.pb.h"
#include "hand_tracking.pb.h"
#include "meta_data.pb.h"
#include "robot.pb.h"

//#include "wrapper.hpp"
//#include <hand-pose-estimation/source/hand_pose_estimation/hololens_hand_data.hpp>

typedef std::chrono::duration<int64_t, std::ratio<1, 10'000'000>> hundreds_of_nanoseconds;

struct VoxelRobot
{
	float voxel_length;
	Eigen::Matrix4f robot_origin;
	std::vector<Eigen::Vector<uint32_t, 3>> voxels;
};

namespace server
{
	template<typename out, typename in>
	out convert(const in& v);

	template<typename out, typename in>
	out convert_meta(const in& v, bool send_meta);

	//template<typename out, typename in>
	//out convert(const in&& v);

	inline generated::Transformation_Meta gen_meta()
	{
		generated::Transformation_Meta out;

		auto& right = *out.mutable_right();
		right.set_axis(generated::X);
		right.set_direction(generated::POSITIVE);

		auto& forward = *out.mutable_forward();
		forward.set_axis(generated::Y);
		forward.set_direction(generated::POSITIVE);

		auto& up = *out.mutable_up();
		up.set_axis(generated::Z);
		up.set_direction(generated::POSITIVE);

		return out;
	}

	template<>
	inline generated::quaternion convert(const Eigen::Quaternionf& in)
	{
		generated::quaternion out;

		out.set_x(in.x());
		out.set_y(in.y());
		out.set_z(in.z());
		out.set_w(in.w());

		return out;
	}

	template<>
	inline Eigen::Quaternionf convert(const generated::quaternion& in)
	{
		return Eigen::Quaternionf(in.w(), in.x(), in.y(), in.z());
	}
	
	template<>
	inline generated::vertex_3d convert(const Eigen::Vector3f& v)
	{
		generated::vertex_3d out;
		out.set_x(v.x());
		out.set_y(v.y());
		out.set_z(v.z());

		return out;
	}

	template<>
	inline generated::index_3d convert(const Eigen::Vector<uint32_t, 3>& v)
	{
		generated::index_3d out;
		out.set_x(v.x());
		out.set_y(v.y());
		out.set_z(v.z());

		return out;
	}

	template<>
	inline generated::size_3d convert(const Eigen::Vector3f& in)
	{
		generated::size_3d out;
		out.set_x(in.x());
		out.set_y(in.y());
		out.set_z(in.z());

		return out;
	}

	template<>
	inline generated::color convert(const pcl::RGB& in)
	{
		generated::color out;

		out.set_r(in.r);
		out.set_g(in.g);
		out.set_b(in.b);
		out.set_a(in.a);

		return out;
	}

	/*
	template<>
	inline generated::aabb convert(const state_observation::aabb& in)
	{
		generated::aabb out;
		*out.mutable_diagonal() =
			convert<generated::size_3d, Eigen::Vector3f>(in.diagonal);

		*out.mutable_translation() =
			convert<generated::vertex_3d, Eigen::Vector3f>(in.translation);

		return out;
	}
	
	template<>
	inline google::protobuf::RepeatedField<google::protobuf::uint32> convert(
		const std::vector<tinyobj::index_t>& indices)
	{
		google::protobuf::RepeatedField<uint32_t> out;
		out.Reserve(indices.size());

		for (const auto& index : indices)
			out.AddAlreadyReserved(index.vertex_index);

		return out;
	}

	template<>
	inline google::protobuf::RepeatedPtrField<generated::vertex_3d> convert(
		const std::vector<tinyobj::real_t>& vertices)
	{
		google::protobuf::RepeatedPtrField<generated::vertex_3d> out;
		out.Reserve(vertices.size());

		for (size_t i = 0; i < vertices.size(); i += 3)
		{
			generated::vertex_3d outVertex;
			outVertex.set_x(vertices[i]);
			outVertex.set_y(vertices[i + 1]);
			outVertex.set_z(vertices[i + 2]);
			out.Add(std::move(outVertex));
		}
		return out;
	}*/

	template<int rows, int cols>
	Eigen::Matrix<float, rows, cols> convert(const generated::Matrix& m)
	{
		Eigen::Matrix<float, rows, cols> matrix(cols, rows);
		if constexpr (rows >= 0)
			if (m.rows() != rows)
				throw std::exception("Invalid rows");
		if constexpr (cols >= 0)
			if (m.rows() != cols)
				throw std::exception("Invalid cols");

		for (size_t y = 0; y < m.rows(); ++y)
			for (size_t x = 0; x < m.cols(); ++x)
				matrix(x, y) = m.data()[y * m.cols() + x];

		return matrix;
	}
	/*
	template<>
	inline generated::object_prototype convert(
		const state_observation::object_prototype::ConstPtr& in)
	{
		generated::object_prototype proto;
		*proto.mutable_bounding_box() =
			convert<generated::aabb>(in->get_bounding_box());

		*proto.mutable_mean_color() =
			convert<generated::color>(in->get_mean_color());

		auto& col = *proto.mutable_mean_color();
		auto f = [](int c)
		{
			float x = std::exp(10 * c / 255.f - 4.4);
			return static_cast<int>(x / (x + 1) * 255.f);
		};
		col.set_r(f(col.r()));
		col.set_g(f(col.g()));
		col.set_b(f(col.b()));

		proto.set_mesh_name(in->get_base_mesh()->get_path());
		proto.set_name(in->get_name());
		proto.set_type(in->get_type());

		return proto;
	}

	template<>
	inline generated::mesh_data convert(
		const std::pair<tinyobj::ObjReader, std::string>& obj_pair)
	{
		generated::mesh_data temp;
		const auto& reader = obj_pair.first;
		const auto& name = obj_pair.second;

		const auto& attr = reader.GetAttrib();
		const auto& vertices = attr.GetVertices();
		const auto& normals = attr.normals;

		const auto& indices = reader.GetShapes()[0].mesh.indices;

		std::vector<std::set<size_t>> vertex_normals_pre;
		vertex_normals_pre.resize(vertices.size() / 3);

		for (const auto& index : indices)
			vertex_normals_pre[index.vertex_index].emplace(index.normal_index);

		auto m_vertex_normals =
			temp.mutable_vertex_normals()->mutable_vertices();
		m_vertex_normals->Reserve(vertex_normals_pre.size());

		for (const auto& set : vertex_normals_pre)
		{
			Eigen::Matrix<float, 3, 1> normal;
			normal.setZero();

			for (const auto& idx : set)
			{
				normal[0] += normals[3 * idx + 0];
				normal[1] += normals[3 * idx + 1];
				normal[2] += normals[3 * idx + 2];
			}
			normal.normalize();
			m_vertex_normals->Add(convert<generated::vertex_3d>(normal));
		}

		*temp.mutable_vertices() =
			convert<google::protobuf::RepeatedPtrField<generated::vertex_3d>>(vertices);

		*temp.mutable_indices() =
			convert<google::protobuf::RepeatedField<google::protobuf::uint32>>(indices);

		temp.set_name(name);

		return temp;
	}*/

	template<>
	inline pcl::PointXYZ convert(const Eigen::Vector3f& in)
	{
		return pcl::PointXYZ(in.x(), in.y(), in.z());
	}

	template<>
	inline pcl::PointXYZ convert(const generated::vertex_3d& in)
	{
		return pcl::PointXYZ(in.x(), in.y(), in.z());
	}
	
	template<>
	inline Eigen::Vector4f convert(const pcl::PointXYZ& in)
	{
		return Eigen::Vector4f(in.x, in.y, in.z, 1.f);
	}

	template<int rows, int cols>
	generated::Matrix convert(const Eigen::Matrix<float, rows, cols>& in)
	{
		generated::Matrix out;
		out.set_rows(in.rows());
		out.set_cols(in.cols());
		auto data = out.mutable_data();
		data->Reserve(in.rows() * in.cols());

		for (size_t y = 0; y < in.rows(); ++y)
			for (size_t x = 0; x < in.cols(); ++x)
				if constexpr (in.Options == Eigen::RowMajor)
				{
					*data->Add() = in(y, x);
				}
				else
					*data->Add() = in(x, y);
		return out;
	}
	
	template<>
	inline Eigen::Vector3f convert(const generated::size_3d& in)
	{
		return Eigen::Vector3f(in.x(), in.y(), in.z());
	}

	template<>
	inline Eigen::Vector3f convert(const generated::vertex_3d& in)
	{
		return Eigen::Vector3f(in.x(), in.y(), in.z());
	}

	/*template<>
	inline Eigen::Quaternionf convert(const generated::vertex_3d& in)
	{
		return Eigen::Quaternionf(
			Eigen::AngleAxisf(in.x(), Eigen::Vector3f::UnitX()) *
			Eigen::AngleAxisf(in.y(), Eigen::Vector3f::UnitY()) *
			Eigen::AngleAxisf(in.z(), Eigen::Vector3f::UnitZ()));
	}*/
	/*
	template<>
	inline state_observation::obb convert(const generated::obb& in)
	{
		const auto& a_aligned = in.axis_aligned();
		const auto& rot = in.rotation();

		return state_observation::obb(
			convert<Eigen::Vector3f>(a_aligned.diagonal()),
			convert<Eigen::Vector3f>(a_aligned.translation()),
			convert<Eigen::Quaternionf>(rot)
		);
	}

	template<>
	inline hand_pose_estimation::hololens::hand_index convert(const generated::hand_index& in)
	{
		return (hand_pose_estimation::hololens::hand_index)in;
	}

	template<>
	inline hand_pose_estimation::hololens::tracking_status convert(const generated::tracking_status& in)
	{
		return (hand_pose_estimation::hololens::tracking_status)in;
	}
	
	template<>
	inline hand_pose_estimation::hololens::hand_data convert(const generated::hand_data& in)
	{
		hand_pose_estimation::hololens::hand_data out;
		out.valid = in.valid();

		out.hand = convert<hand_pose_estimation::hololens::hand_index>(in.hand());
		out.tracking_stat = convert<hand_pose_estimation::hololens::tracking_status>(in.tracking_stat());
		
		out.grip_position = convert<Eigen::Vector3f>(in.grip_position());
		out.grip_rotation = convert<Eigen::Quaternionf>(in.grip_rotation());

		out.aim_position = convert<Eigen::Vector3f>(in.aim_position());
		out.aim_rotation = convert<Eigen::Quaternionf>(in.aim_rotation());

		if (in.hand_key_positions_size() != in.hand_key_radii_size() ||
			in.hand_key_radii_size() != in.hand_key_rotations_size())
			throw std::exception("hand_pose_estimation::hololens::hand_data <-> generated::hand_data: size mismatch");

		const size_t size = in.hand_key_radii_size();

		for (size_t i = 0; i < size; ++i)
		{
			out.key_data[i] = hand_pose_estimation::hololens::hand_key_data
			{
				convert<Eigen::Vector3f>(in.hand_key_positions(i)),
				convert<Eigen::Quaternionf>(in.hand_key_rotations(i)),
				in.hand_key_radii(i)
			};
		}

		out.is_grasped = in.is_grasped();
		out.utc_timestamp = std::chrono::time_point<std::chrono::utc_clock, hundreds_of_nanoseconds>{ 
			hundreds_of_nanoseconds{in.utc_timestamp()} 
		};
		
		return out;
	}

	template<>
	inline holo_pointcloud convert(const generated::pcl_data& pcl_data)
	{
		auto recv_timestamp = std::chrono::file_clock::now();

		holo_pointcloud out;

		out.recv_timestamp = recv_timestamp;
		out.timestamp = std::chrono::file_clock::time_point(
			hundreds_of_nanoseconds(pcl_data.timestamp()));

		out.pcl = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
		
		out.pcl->header.stamp = 
			std::chrono::duration_cast<std::chrono::microseconds>(
				std::chrono::file_clock::to_utc(out.timestamp).time_since_epoch()).count();

		out.pcl->reserve(pcl_data.vertices_size());

		for (const auto& p : pcl_data.vertices())
			out.pcl->emplace_back(server::convert<pcl::PointXYZ>(p));

		return out;
	}*/

	template<>
	inline generated::Voxels convert(const VoxelRobot& v)
	{
		generated::Voxels out;
		out.set_voxel_side_length(v.voxel_length);
		*out.mutable_robot_origin() = convert(v.robot_origin);

		auto& voxel_coords = *out.mutable_voxel_indices();
		voxel_coords.Reserve(v.voxels.size());
		for (const auto& voxel_coord : v.voxels)
			voxel_coords.Add(convert<generated::index_3d>(voxel_coord));

		return out;
	}

	template<>
	inline generated::Joints convert(const Eigen::Vector<float, 7>& v)
	{
		generated::Joints out;
		out.set_theta_1(v[0]);
		out.set_theta_2(v[1]);
		out.set_theta_3(v[2]);
		out.set_theta_4(v[3]);
		out.set_theta_5(v[4]);
		out.set_theta_6(v[5]);
		out.set_theta_7(v[6]);

		return out;
	}

	template<>
	inline generated::Tcps convert(const std::vector<Eigen::Vector3f>& v)
	{
		generated::Tcps out;
		auto& out_data = *out.mutable_points();
		out_data.Reserve(v.size());
		for (const auto& val : v)
			out_data.Add(convert<generated::vertex_3d>(val));

		return out;
	}

	template<>
	inline generated::Tcps_TF_Meta convert_meta(const std::vector<Eigen::Vector3f>& v, bool send_meta)
	{
		generated::Tcps_TF_Meta out;
		*out.mutable_tcps() = convert<generated::Tcps>(v);

		if (send_meta)
			*out.mutable_transformation_meta() = gen_meta();

		return out;
	}

	template<>
	inline generated::Voxel_TF_Meta convert_meta(const VoxelRobot& v, bool send_meta)
	{
		generated::Voxel_TF_Meta out;
		*out.mutable_voxels() = convert<generated::Voxels>(v);

		if (send_meta)
			*out.mutable_transformation_meta() = gen_meta();

		return out;
	}
}