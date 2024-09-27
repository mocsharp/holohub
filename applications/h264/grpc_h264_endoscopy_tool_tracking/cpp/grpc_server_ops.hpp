#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_SERVER_OPS_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_SERVER_OPS_HPP

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
using holoscan::entity::EntityResponse;

using grpc::Server;
using grpc::ServerBuilder;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class GrpcServerResponseOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcServerResponseOp)
  GrpcServerResponseOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Entity>("input");

    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC results.");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_input_message = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_input_message) {
      HOLOSCAN_LOG_ERROR("grpc: Failed to receive input message");
      return;
    }
    auto response = std::make_shared<EntityResponse>();
    holoscan::ops::TensorProto::tensor_to_entity_response(maybe_input_message.value(), response);
    response_queue_->push(response);
  }

 private:
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>>
      response_queue_;
  ;
};

class GrpcServerRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcServerRequestOp)

  GrpcServerRequestOp() = default;

  void start() override {
    HOLOSCAN_LOG_INFO("grpc: Starting gRPC server...");
    server_thread_ = std::thread(&GrpcServerRequestOp::StartInternal, this);
    // condition_->event_state(AsynchronousEventState::EVENT_WAITING);
  }

  void stop() override {
    // condition_->event_state(AsynchronousEventState::EVENT_NEVER);
    HOLOSCAN_LOG_INFO("grpc: Stopping gRPC server...");
    server_->Shutdown();
    if (server_thread_.joinable()) { server_thread_.join(); }
  }

  void initialize() override {
    // if (condition_.has_value()) { add_arg(condition_.get()); }
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.param(server_address_, "server_address", "Server Address", "gRPC Server Address.");
    spec.param(request_queue_, "request_queue", "Request Queue", "Incoming gRPC requests.");
    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC responses.");
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
    // spec.param(condition_, "condition", "Asynchronous Condition", "Asynchronous Condition");

    spec.output<holoscan::gxf::Entity>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    HOLOSCAN_LOG_INFO("grpc: GrpcServerRequestOp::compute");
    if (!request_queue_->empty()) {
      auto request = request_queue_->pop();
      auto result = nvidia::gxf::Entity(std::move(*request));
      op_output.emit(result, "output");
    }
    // if (request_queue_->empty()) {
    //   HOLOSCAN_LOG_INFO("GrpcServerRequestOp::compute: request_queue_ is empty");
    //   condition_->event_state(AsynchronousEventState::EVENT_WAITING);
    // }
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

  // Parameter<std::shared_ptr<AsynchronousCondition>> condition_;
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>>>
      request_queue_;
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>>
      response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> server_address_;
  std::thread server_thread_;
  std::unique_ptr<Server> server_;
};

// class TestOp : public holoscan::Operator {
//  public:
//   HOLOSCAN_OPERATOR_FORWARD_ARGS(TestOp)

//   TestOp() = default;

//   void setup(OperatorSpec& spec) override {
//     spec.input<holoscan::gxf::Entity>("in");
//     spec.output<holoscan::gxf::Entity>("out");
//   }

//   void compute(InputContext& op_input, OutputContext& op_output,
//                ExecutionContext& context) override {
//     auto maybe_input_message = op_input.receive<holoscan::gxf::Entity>("in");
//     if (!maybe_input_message) {
//       HOLOSCAN_LOG_ERROR("Failed to receive input message");
//       return;
//     }
//     auto input_image = maybe_input_message.value().get<Tensor>();
//     if (!input_image) {
//       HOLOSCAN_LOG_ERROR("Failed to get image from message");
//       return;
//     }
//   }
// };

using namespace holoscan;
class TestConvertOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TestConvertOp)

  TestConvertOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_input_message = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_input_message) {
      HOLOSCAN_LOG_ERROR("grpc: Failed to receive input message");
      return;
    }

    auto request = std::make_shared<EntityRequest>();
    holoscan::ops::TensorProto::tensor_to_entity_request(maybe_input_message.value(), request);

    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());
    auto out_message = nvidia::gxf::Entity::New(fragment()->executor().context()).value();

    holoscan::ops::TensorProto::entity_request_to_tensor(
        request.get(), out_message, gxf_allocator.value());

    auto result = nvidia::gxf::Entity(std::move(out_message));
    op_output.emit(result, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_SERVER_OPS_HPP */
