#include "grpc_server.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

VoxelService::VoxelService()
	: voxel_slot([this](const VoxelRobot& data) { handle_voxels(data); }),
	joint_slot([this](const Eigen::Vector<float, 7>& data)
	{
			std::unique_lock lock(joint_mutex);
			if (stop_flag)
				return;

			joint_buffer = std::make_unique<generated::Joints>(server::convert<generated::Joints>(data));
			joint_cv.notify_one();
	}),
	tcps_slot([this](const std::vector<Eigen::Vector3f>& data)
	{
			std::unique_lock lock(tcps_mutex);
			if (stop_flag)
				return;

			tcps_buffer = std::make_unique<std::vector<Eigen::Vector3f>>(data);
			tcps_cv.notify_one();
	})
{}

VoxelService::~VoxelService()
{
	stop();
}

grpc::Status VoxelService::transmit_voxels(
	::grpc::ServerContext* context, 
	const::google::protobuf::Empty* request, 
	::grpc::ServerWriter<::generated::Voxel_TF_Meta>* writer)
{
	context->set_compression_algorithm(GRPC_COMPRESS_GZIP);
	writer->SendInitialMetadata();

	bool first = true;
	while (true)
	{
		std::unique_lock lock(mutex);
		cv.wait(lock, [this]() { return !!buffer || stop_flag; });

		if (stop_flag)
			break;

		const auto voxel_send = server::convert_meta<generated::Voxel_TF_Meta>(*buffer, first);
		first = false;

		if (!writer->Write(voxel_send))
			return grpc::Status::CANCELLED;

		buffer.reset();
	}
	return grpc::Status::OK;
}

grpc::Status VoxelService::transmit_joints(grpc::ServerContext* context, const google::protobuf::Empty* request, grpc::ServerWriter<generated::Joints>* writer)
{
	writer->SendInitialMetadata();

	while (true)
	{
		std::unique_lock lock(joint_mutex);
		cv.wait(lock, [this]() { return !!joint_buffer || stop_flag; });

		if (stop_flag)
			break;

		if (!writer->Write(*joint_buffer))
			return grpc::Status::CANCELLED;

		joint_buffer.reset();
	}
	return grpc::Status::OK;
}

grpc::Status VoxelService::transmit_tcps(grpc::ServerContext* context, const google::protobuf::Empty* request, grpc::ServerWriter<generated::Tcps_TF_Meta>* writer)
{
	writer->SendInitialMetadata();
	bool first = true;

	while (true)
	{
		std::unique_lock lock(tcps_mutex);
		cv.wait(lock, [this]() { return !!tcps_buffer || stop_flag; });

		if (stop_flag)
			break;

		const auto tcps_send = server::convert_meta<generated::Tcps_TF_Meta>(*tcps_buffer, first);
		first = false;

		if (!writer->Write(tcps_send))
			return grpc::Status::CANCELLED;

		tcps_buffer.reset();
	}
	return grpc::Status::OK;
}

void VoxelService::handle_voxels(const VoxelRobot& data)
{
	std::unique_lock lock(mutex);
	if (stop_flag)
		return;
	//overwrite buffer if still didn't change
	//we don't need outdated voxel visuals anyway
	buffer = std::make_unique<VoxelRobot>(data);
	cv.notify_one();
}

void VoxelService::stop()
{
	std::scoped_lock lock(mutex, joint_mutex, tcps_mutex);
	stop_flag = true;
	cv.notify_all();
	joint_cv.notify_all();
	tcps_cv.notify_all();
}




void VoxelServer::run_server()
{
	std::string server_address("0.0.0.0:50051");

	grpc::ServerBuilder builder;
	builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
	builder.RegisterService(&voxel_service);
	server = builder.BuildAndStart();
	std::cout << "Server listening on " << server_address << std::endl;
	server->Wait();
}