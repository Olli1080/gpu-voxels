#include "grpc_server.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

VoxelService::VoxelService()
	: voxel_slot([this](const VoxelRobot& data) { handle_voxels(data); })
{}

VoxelService::~VoxelService()
{
	stop();
}

grpc::Status VoxelService::transmit_voxels(
	::grpc::ServerContext* context, 
	const::google::protobuf::Empty* request, 
	::grpc::ServerWriter<::generated::voxels>* writer)
{
	context->set_compression_algorithm(GRPC_COMPRESS_GZIP);
	writer->SendInitialMetadata();

	while (true)
	{
		std::unique_lock lock(mutex);
		cv.wait(lock, [this]() { return !!buffer || stop_flag; });

		if (stop_flag)
			break;

		if (!writer->Write(*buffer))
			return grpc::Status::CANCELLED;

		buffer.reset();
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
	buffer = std::make_unique<generated::voxels>(server::convert<generated::voxels>(data));
	cv.notify_one();
}

void VoxelService::stop()
{
	std::unique_lock lock(mutex);
	stop_flag = true;
	cv.notify_one();
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