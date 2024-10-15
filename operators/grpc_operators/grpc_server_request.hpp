#ifndef GRPC_SERVER_REQUEST_HPP
#define GRPC_SERVER_REQUEST_HPP

#include <holoscan.pb.h>
#include <entity_server.hpp>

using grpc::Server;
using grpc::ServerBuilder;

namespace holoscan::ops {
using namespace holoscan::ops;

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

}  // namespace holoscan::ops

#endif /* GRPC_SERVER_REQUEST_HPP */
