#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP

#include <memory>
#include <queue>

#include <grpcpp/grpcpp.h>
// #include <grpcpp/ext/proto_server_reflection_plugin.h>
// #include <grpcpp/health_check_service_interface.h>

#include <holoscan.pb.h>
#include <tensor_proto.hpp>

#include "entity_client.hpp"
#include "entity_server.hpp"
#include "resource_queue.hpp"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

using grpc::Server;
using grpc::ServerBuilder;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class GrpcClientResponseOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcClientResponseOp)

  GrpcClientResponseOp() = default;

  void start() override { condition_->event_state(AsynchronousEventState::EVENT_WAITING); }

  void stop() override { condition_->event_state(AsynchronousEventState::EVENT_NEVER); }

  void initialize() override {
    if (condition_.has_value()) { add_arg(condition_.get()); }
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC responses.");
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
    spec.param(condition_, "condition", "Asynchronous Condition", "Asynchronous Condition");

    spec.output<holoscan::gxf::Entity>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto response = response_queue_->pop();
    auto result = nvidia::gxf::Entity(std::move(response));
    op_output.emit(result, "output");
    condition_->event_state(AsynchronousEventState::EVENT_WAITING);
  }

 private:
  Parameter<std::shared_ptr<AsynchronousCondition>> condition_;
  Parameter<std::shared_ptr<AsynchronousConditionQueue>> response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};

class GrpcClientRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcClientRequestOp)
  GrpcClientRequestOp() = default;

  void start() override {
    entity_client_ = make_shared<EntityClient>(EntityClient("localhost:50051"));
  }

  void setup(OperatorSpec& spec) override {
    spec.input<nvidia::gxf::Entity>("input");

    spec.param(request_queue_, "request_queue", "Request Queue", "Outgoing gRPC requests.");
    spec.param(response_queue_, "response_queue", "Response Queue", "Incoming gRPC responses.");
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_input_message = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_input_message) {
      HOLOSCAN_LOG_ERROR("Failed to receive input message");
      return;
    }
    auto request = std::make_shared<EntityRequest>();
    holoscan::ops::TensorProto::tensor_to_entity_request(maybe_input_message.value(), request);
    request_queue_->push(request);
    if (!stream_channel_started_) {
      entity_client_->EndoscopyToolTracking(
          // Handle outgoing requests
          [this](std::shared_ptr<EntityRequest>& request) {
            if (request_status_ == EntityClient::DONE) [[unlikely]] { return EntityClient::DONE; }

            if (request_queue_->empty()) { return EntityClient::WAIT; }

            request = request_queue_->pop();
            request->set_service("endoscopy_tool_tracking");

            HOLOSCAN_LOG_DEBUG("endoscopy_tool_tracking: Sending request");
            return EntityClient::WRITE;
          },
          // Handle incoming responses
          [this](EntityResponse& response) {
            auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                fragment()->executor().context(), allocator_->gxf_cid());
            auto out_message = nvidia::gxf::Entity::New(fragment()->executor().context());
            holoscan::ops::TensorProto::entity_response_to_tensor(
                response, out_message.value(), gxf_allocator.value());
            response_queue_->push(out_message.value());
            HOLOSCAN_LOG_DEBUG("Response received and queued");
          },
          // Complete the requests
          [this](const grpc::Status& status) {
            if (!status.ok()) {
              HOLOSCAN_LOG_ERROR("gRPC call failed: {}", status.error_message());
              return;
            }
            HOLOSCAN_LOG_INFO("gRPC call completed");
          });
      stream_channel_started_ = true;
    }
  }

 private:
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>>> request_queue_;
  Parameter<std::shared_ptr<AsynchronousConditionQueue>> response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  std::shared_ptr<EntityClient> entity_client_;
  bool stream_channel_started_ = false;
  EntityClient::RequestStatus request_status_ = EntityClient::WAIT;
};

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
      HOLOSCAN_LOG_ERROR("Failed to receive input message");
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
    HOLOSCAN_LOG_INFO("Starting gRPC server...");
    server_thread_ = std::thread(&GrpcServerRequestOp::StartInternal, this);
    condition_->event_state(AsynchronousEventState::EVENT_WAITING);
  }

  void stop() override {
    shutdown_ = true;
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

    spec.output<holoscan::gxf::Entity>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto request = request_queue_->pop();
    auto result = nvidia::gxf::Entity(std::move(request));
    op_output.emit(result, "output");

    if (request_queue_->empty()) { condition_->event_state(AsynchronousEventState::EVENT_WAITING); }
  }

 private:
  void StartInternal() {
    HoloscanEntityServiceImpl service(
        // Handle incoming requests
        [this](EntityRequest& request) {
          auto route = request.service();
          if (route == "endoscopy_tool_tracking") {
            auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
                fragment()->executor().context(), allocator_->gxf_cid());
            auto out_message = nvidia::gxf::Entity::New(fragment()->executor().context());

            holoscan::ops::TensorProto::entity_request_to_tensor(
                &request, out_message.value(), gxf_allocator.value());

            request_queue_->push(out_message.value());
            HOLOSCAN_LOG_INFO("endoscopy_tool_tracking: tensor received and queued");
          }
        },
        // Handle outgoing responses
        [this](std::shared_ptr<EntityResponse>& response) {
          if (response_queue_->empty()) { return HoloscanEntityServiceImpl::WAIT; }
          response = response_queue_->pop();
          HOLOSCAN_LOG_INFO("response message set");
          return HoloscanEntityServiceImpl::WRITE;
        },
        std::ref(shutdown_));
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
  Parameter<std::shared_ptr<AsynchronousConditionQueue>> request_queue_;
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>>
      response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::string> server_address_;
  std::thread server_thread_;
  std::unique_ptr<Server> server_;
  std::atomic<bool> shutdown_ = false;
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
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP */
