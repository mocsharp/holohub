#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP

#include <memory>
#include <queue>

#include <grpcpp/grpcpp.h>
// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/health_check_service_interface.h>

#include <holoscan.pb.h>
#include <tensor_proto.hpp>

#include "entity_server.hpp"
#include "resource_queue.hpp"

using holoscan::entity::EntityRequest;

using grpc::Server;
using grpc::ServerBuilder;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class GrpcClientOperator : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcClientOperator);

  GrpcClientOperator() = default;

  void start() override {
    HOLOSCAN_LOG_INFO("Starting gRPC client...");
    grpc_client_ = std::make_shared<EntityClient>(EntityClient(
        grpc::CreateChannel(server_address_.get(), grpc::InsecureChannelCredentials())));
  }

  void setup(OperatorSpec& spec) override {
    spec.param<std::string>(
        server_address_, "server_address", "gRPC server address", "gRPC server address");
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

    spec.input<nvidia::gxf::Entity>("input");
    spec.output<nvidia::gxf::Entity>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) { throw std::runtime_error("Failed to receive input"); }

    // Create output message
    auto out_message = nvidia::gxf::Entity::New(context.context());

    // Get allocator
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                             allocator_->gxf_cid());
    auto entity = maybe_entity.value();
    auto request = holoscan::ops::TensorProto::tensor_to_entity_request(entity);
    grpc_client_->EndoscopyToolTracking(request, out_message.value(), gxf_allocator.value());

    auto result = nvidia::gxf::Entity(std::move(out_message.value()));
    op_output.emit(result, "output");
  }

 private:
  Parameter<std::string> server_address_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  std::shared_ptr<EntityClient> grpc_client_;
};

class GrpcResponseOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcResponseOp)
  GrpcResponseOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Entity>("in");

    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC results.");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_input_message = op_input.receive<holoscan::gxf::Entity>("in");
    if (!maybe_input_message) {
      HOLOSCAN_LOG_ERROR("Failed to receive input message");
      return;
    }
    response_queue_->push(maybe_input_message.value());
  }

 private:
  Parameter<std::shared_ptr<ResponseQueue>> response_queue_;
  ;
};

class GrpcRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcRequestOp)

  GrpcRequestOp() = default;

  void start() override {
    HOLOSCAN_LOG_INFO("Starting gRPC server...");
    server_thread_ = std::thread(&GrpcRequestOp::StartInternal, this);
    condition_->event_state(AsynchronousEventState::EVENT_WAITING);
  }

  void stop() override {
    condition_->event_state(AsynchronousEventState::EVENT_NEVER);
    HOLOSCAN_LOG_INFO("Stopping gRPC server...");
    server_->Shutdown();
    if (server_thread_.joinable()) { server_thread_.join(); }
  }

  void initialize() override {
    if (condition_.has_value()) { add_arg(condition_.get()); }
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.param(server_address_, "server_address", "Server Address", "gRPC Server Address.");
    spec.param(request_queue_, "request_queue", "Request Queue", "Incoming gRPC requests.");
    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC responses.");
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
    spec.param(condition_, "condition", "Asynchronous Condition", "Asynchronous Condition");

    spec.output<holoscan::gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto request = request_queue_->pop();
    auto result = nvidia::gxf::Entity(std::move(request));
    op_output.emit(result, "out");
    condition_->event_state(AsynchronousEventState::EVENT_WAITING);
  }

 private:
  void StartInternal() {
    HoloscanEntityServiceImpl service(
        request_queue_, response_queue_, allocator_, fragment()->executor().context());
    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();
    ServerBuilder builder;
    builder.AddListeningPort(server_address_, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    HOLOSCAN_LOG_INFO("Server listening on {}", server_address_);
    server->Wait();
  }

  Parameter<std::shared_ptr<AsynchronousCondition>> condition_;
  Parameter<std::shared_ptr<RequestQueue>> request_queue_;
  Parameter<std::shared_ptr<ResponseQueue>> response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> server_address_;
  std::thread server_thread_;
  std::unique_ptr<Server> server_;
};

class TestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestOp)

  TestOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("in");
    spec.output<holoscan::gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_input_message = op_input.receive<holoscan::gxf::Entity>("in");
    if (!maybe_input_message) {
      HOLOSCAN_LOG_ERROR("Failed to receive input message");
      return;
    }
    auto input_image = maybe_input_message.value().get<Tensor>();
    if (!input_image) {
      HOLOSCAN_LOG_ERROR("Failed to get image from message");
      return;
    }
  }
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP */
