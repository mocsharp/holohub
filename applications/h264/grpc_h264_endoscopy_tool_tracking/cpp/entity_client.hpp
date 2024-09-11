/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_CLIENT_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_CLIENT_HPP

#include <fmt/format.h>
#include <grpcpp/grpcpp.h>
#include <tensor_proto.hpp>

#include "holoscan.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::ServerContext;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

#include <gxf/std/tensor.hpp>
#include <tensor_proto.hpp>

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class EntityClient {
 public:
  explicit EntityClient(std::shared_ptr<Channel> channel) : stub_(Entity::NewStub(channel)) {}

  const nvidia::gxf::Entity EndoscopyToolTracking(EntityRequest& request,
                                                  ExecutionContext& execution_context,
                                                  std::shared_ptr<Allocator> allocator) {
    request.set_service("endoscopy_tool_tracking");

    EntityResponse reply;
    ClientContext context;

    Status status = stub_->Metadata(&context, request, &reply);

    // Act upon its status.
    if (status.ok()) {
      auto entity = holoscan::ops::TensorProto::entity_response_to_tensor(
          reply, execution_context, allocator);
      return entity;
    } else {
      throw std::runtime_error(
          fmt::format("RPC Failed with code: {}: {}", status.error_code(), status.error_message()));
    }
  }

 private:
  std::unique_ptr<Entity::Stub> stub_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_CLIENT_HPP */
