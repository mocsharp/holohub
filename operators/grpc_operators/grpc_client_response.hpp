/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRPC_CLIENT_RESPONSE_HPP
#define GRPC_CLIENT_RESPONSE_HPP

#include <holoscan.pb.h>

namespace holoscan::ops {
using namespace std;

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
    std::shared_ptr<nvidia::gxf::Entity> response = response_queue_->pop();

    if (response) {
      auto result = nvidia::gxf::Entity(std::move(*response));
      op_output.emit(result, "output");
      condition_->event_state(AsynchronousEventState::EVENT_WAITING);
    }
  }

 private:
  Parameter<std::shared_ptr<AsynchronousCondition>> condition_;
  Parameter<std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>>
      response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
};
}  // namespace holoscan::ops

#endif /* GRPC_CLIENT_RESPONSE_HPP */
