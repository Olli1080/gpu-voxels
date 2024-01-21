#pragma once

#include <vector>
#include <mutex>
#include <condition_variable>

#include "util.h"

#include <grpc/grpc.h>

#include <Eigen/Dense>
#include <boost/signals2/signal.hpp>

#include "robot.grpc.pb.h"

class VoxelService : public generated::robot_com::Service
{
public:

	VoxelService();
	~VoxelService() override;

	grpc::Status transmit_voxels(::grpc::ServerContext* context, const ::google::protobuf::Empty* request, ::grpc::ServerWriter<::generated::voxels>* writer) override;
	grpc::Status transmit_joints(grpc::ServerContext* context, const google::protobuf::Empty* request, grpc::ServerWriter<generated::joints>* writer) override;
	grpc::Status transmit_tcps(grpc::ServerContext* context, const google::protobuf::Empty* request, grpc::ServerWriter<generated::tcps>* writer) override;

	boost::signals2::slot<void(const VoxelRobot&)> voxel_slot;
	boost::signals2::slot<void(const Eigen::Vector<float, 7>&)> joint_slot;
	boost::signals2::slot<void(const std::vector<Eigen::Vector3f>&)> tcps_slot;

private:

	void stop();

	void handle_voxels(const VoxelRobot& data);

	std::unique_ptr<generated::voxels> buffer;

	std::condition_variable cv;
	std::mutex mutex;
	bool stop_flag = false;


	std::unique_ptr<generated::joints> joint_buffer;
	std::condition_variable joint_cv;
	std::mutex joint_mutex;


	std::unique_ptr<generated::tcps> tcps_buffer;
	std::condition_variable tcps_cv;
	std::mutex tcps_mutex;
};

class VoxelServer
{
public:

	void run_server();

	VoxelService voxel_service;

	std::unique_ptr<grpc::Server> server;

private:

};