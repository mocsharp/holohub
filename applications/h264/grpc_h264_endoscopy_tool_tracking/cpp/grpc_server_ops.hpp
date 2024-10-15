#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_SERVER_OPS_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_SERVER_OPS_HPP

#include <memory>
#include <queue>

#include <grpcpp/grpcpp.h>

#include <holoscan.pb.h>
#include <tensor_proto.hpp>

#include "entity_server.hpp"
#include "resource_queue.hpp"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

using grpc::Server;
using grpc::ServerBuilder;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class GrpcServerResponseOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcServerResponseOp)
  GrpcServerResponseOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Entity>("system");
    spec.input<nvidia::gxf::Entity>("device");

    spec.param(device_to_system_tensors_,
               "device_to_system_tensors",
               "Device Memory to System Memory",
               "Copies tensors from device memory to sytem memory.");
    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC results.");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto tensors = 0;
    auto response = std::make_shared<EntityResponse>();

    auto maybe_system_message = op_input.receive<holoscan::gxf::Entity>("system");
    if (maybe_system_message) {
      holoscan::ops::TensorProto::tensor_to_entity_response(maybe_system_message.value(), response);
      tensors++;
    }

    auto maybe_device_message = op_input.receive<holoscan::gxf::Entity>("device");
    if (maybe_device_message) {
      holoscan::ops::TensorProto::tensor_to_entity_response(maybe_device_message.value(), response);
      tensors++;
    }

    if (tensors > 0) { response_queue_->push(response); }
  }

 private:
  Parameter<std::vector<std::string>> device_to_system_tensors_;
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>>
      response_queue_;
};

class GrpcServerRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcServerRequestOp)

  GrpcServerRequestOp() = default;

  void start() override {
    HOLOSCAN_LOG_INFO("grpc: Starting gRPC server...");
    server_thread_ = std::thread(&GrpcServerRequestOp::StartInternal, this);
  }

  void stop() override {
    HOLOSCAN_LOG_INFO("grpc: Stopping gRPC server...");
    server_->Shutdown();
    if (server_thread_.joinable()) { server_thread_.join(); }
  }

  void initialize() override { Operator::initialize(); }

  void setup(OperatorSpec& spec) override {
    spec.param(server_address_, "server_address", "Server Address", "gRPC Server Address.");
    spec.param(request_queue_, "request_queue", "Request Queue", "Incoming gRPC requests.");
    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC responses.");
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

    spec.output<holoscan::gxf::Entity>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    if (!request_queue_->empty()) {
      auto request = request_queue_->pop();
      auto result = nvidia::gxf::Entity(std::move(*request));
      op_output.emit(result, "output");
    }
  }

 private:
  void StartInternal() {
    HoloscanEntityServiceImpl service(
        request_queue_.get(),
        response_queue_.get(),
        // Handle incoming requests
        [this](EntityRequest& request) {
          HOLOSCAN_LOG_INFO("grpc: server OnReadDone callback: request received");
          auto route = request.service();
          if (route == "endoscopy_tool_tracking") {
            auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                fragment()->executor().context(), allocator_->gxf_cid());
            auto out_message = nvidia::gxf::Entity::New(fragment()->executor().context());

            holoscan::ops::TensorProto::entity_request_to_tensor(
                &request, out_message.value(), gxf_allocator.value());

            auto entity = std::make_shared<nvidia::gxf::Entity>(out_message.value());
            return entity;
          }
        });
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    HOLOSCAN_LOG_INFO("grpc: Server listening on {}", server_address_);
    server->Wait();
  }

  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>>>
      request_queue_;
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>>
      response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> server_address_;
  std::thread server_thread_;
  std::unique_ptr<Server> server_;
};

}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_SERVER_OPS_HPP */
