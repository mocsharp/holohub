#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP

#include <memory>
#include <queue>

#include <holoscan.pb.h>
#include <tensor_proto.hpp>

#include "resource_queue.hpp"

using holoscan::entity::EntityRequest;

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

    spec.input<std::string>("input");
    spec.output<std::string>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) { throw std::runtime_error("Failed to receive input"); }
    auto entity = maybe_entity.value();
    auto request = holoscan::ops::TensorProto::tensor_to_entity_request(entity);
    auto reply = grpc_client_->EndoscopyToolTracking(request, context, allocator_.get());
    op_output.emit(reply, "output");
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
    auto& input = spec.input<nvidia::gxf::Entity>("in");
    spec.param(in_, "in", "Input", "Input port.", &input);

    spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC results.");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto in_message = op_input.receive<nvidia::gxf::Entity>("in").value();

    response_queue_->push(in_message);
  }

 private:
  Parameter<holoscan::IOSpec*> in_;
  Parameter<std::shared_ptr<ResponseQueue>> response_queue_;
  ;
};

class GrpcRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcRequestOp)

  GrpcRequestOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(request_queue_, "request_queue", "Request Queue", "Incoming gRPC requests.");
    spec.param(processing_queue_, "processing_queue", "Processing Queue", "In processing queue.");
    spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

    spec.output<std::string>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    request_queue_->block_until_data_available();
    auto request = request_queue_->pop();
    processing_queue_->push(std::get<0>(request));
    auto tensor = holoscan::ops::TensorProto::entity_request_to_tensor(
        std::get<1>(request), context, allocator_.get());
    op_output.emit(tensor, "out");
  }

 private:
  Parameter<std::shared_ptr<RequestQueue>> request_queue_;
  Parameter<std::shared_ptr<ProcessingQueue>> processing_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_GRPC_OPS_HPP */
