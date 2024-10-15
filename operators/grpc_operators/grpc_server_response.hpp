#ifndef GRPC_SERVER_RESPONSE_HPP
#define GRPC_SERVER_RESPONSE_HPP

#include <holoscan.pb.h>

namespace holoscan::ops {

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
}  // namespace holoscan::ops
#endif /* GRPC_SERVER_RESPONSE_HPP */
