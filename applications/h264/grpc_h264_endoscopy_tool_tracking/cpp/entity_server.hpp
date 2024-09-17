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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_SERVER_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_SERVER_HPP

#include <fmt/format.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <algorithm>
#include <functional>
#include <random>
#include <string>

#include <tensor_proto.hpp>

#include "holoscan.grpc.pb.h"
#include "holoscan.pb.h"
#include "resource_queue.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class HoloscanEntityServiceImpl final : public Entity::Service {
 public:
  HoloscanEntityServiceImpl(std::shared_ptr<RequestQueue> request_queue,
                            std::shared_ptr<ResponseQueue> response_queue,
                            std::shared_ptr<Allocator> allocator, void* gxf_context)
      : request_queue_(request_queue),
        response_queue_(response_queue),
        allocator_(allocator),
        gxf_context_(gxf_context) {}

  Status Metadata(ServerContext* context, const EntityRequest* request,
                  EntityResponse* reply) override {
    auto service = request->service();

    if (service == "endoscopy_tool_tracking") {
      auto out_message = nvidia::gxf::Entity::New(gxf_context_);
      auto gxf_allocator =
          nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context_, allocator_->gxf_cid());
      holoscan::ops::TensorProto::entity_request_to_tensor(
          request, out_message.value(), gxf_allocator.value());

      request_queue_->push(out_message.value());
      // response_queue_->block_until_data_available();
      // auto response = response_queue_->pop();
      // holoscan::ops::TensorProto::tensor_to_entity_response(response, reply);
      return Status::OK;
    }

    return grpc::Status(grpc::StatusCode::NOT_FOUND, fmt::format("Service {} not found", service));
  }

 private:
  std::shared_ptr<RequestQueue> request_queue_;
  std::shared_ptr<ResponseQueue> response_queue_;
  std::shared_ptr<Allocator> allocator_;
  void* gxf_context_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_SERVER_HPP */
