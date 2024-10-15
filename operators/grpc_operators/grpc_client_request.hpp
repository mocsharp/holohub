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

#ifndef GRPC_CLIENT_REQUEST_HPP
#define GRPC_CLIENT_REQUEST_HPP

#include <holoscan.pb.h>

#include "asynchronous_condition_queue.hpp"
#include "conditional_variable_queue.hpp"
#include "entity_client.hpp"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

class GrpcClientRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcClientRequestOp)
  GrpcClientRequestOp() = default;

  void start() override {
    entity_client_ = std::make_shared<EntityClient>(
        EntityClient("localhost:50051", request_queue_.get(), response_queue_.get()));
    streaming_thread_ = std::thread(&GrpcClientRequestOp::EndoscopyToolTrackingStreaming, this);
  }

  void stop() override { streaming_thread_.join(); }

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
      HOLOSCAN_LOG_ERROR("grpc: Failed to receive input message");
      return;
    }
    auto request = std::make_shared<EntityRequest>();
    holoscan::ops::TensorProto::tensor_to_entity_request(maybe_input_message.value(), request);
    request_queue_->push(request);
    HOLOSCAN_LOG_INFO("grpc: request converted and queued for transmission");
  }

 private:
  void EndoscopyToolTrackingStreaming() {
    entity_client_->EndoscopyToolTracking(
        // Handle incoming responses
        [this](EntityResponse& response) {
          auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
              fragment()->executor().context(), allocator_->gxf_cid());
          auto out_message = nvidia::gxf::Entity::New(fragment()->executor().context());
          holoscan::ops::TensorProto::entity_response_to_tensor(
              response, out_message.value(), gxf_allocator.value());
          auto entity = std::make_shared<nvidia::gxf::Entity>(out_message.value());
          return entity;
        });
  }

  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>>> request_queue_;
  Parameter<std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>>
      response_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;
  std::shared_ptr<EntityClient> entity_client_;
  std::thread streaming_thread_;
};
}  // namespace holoscan::ops
#endif /* GRPC_CLIENT_REQUEST_HPP */
