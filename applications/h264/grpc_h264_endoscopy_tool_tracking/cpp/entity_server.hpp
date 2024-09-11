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
                            std::shared_ptr<ProcessingQueue> processing_queue,
                            std::shared_ptr<ResponseQueue> response_queue)
      : request_queue_(request_queue),
        processing_queue_(processing_queue),
        response_queue_(response_queue) {}

  Status Metadata(ServerContext* context, const EntityRequest* request,
                  EntityResponse* reply) override {
    auto service = request->service();

    if (service == "endoscopy_tool_tracking") {
      ;
      std::string request_id = generate_request_id();
      request_queue_->push(request_id, request);
      response_queue_->block_until_data_available();
      auto response = response_queue_->pop();
      holoscan::ops::TensorProto::tensor_to_entity_response(response, reply);
      return Status::OK;
    }

    return grpc::Status(grpc::StatusCode::NOT_FOUND, fmt::format("Service {} not found", service));
  }

 private:
  std::string generate_request_id(size_t length = 16) {
    const std::string characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::random_device random_device;
    std::mt19937 generator(random_device());
    std::uniform_int_distribution<> distribution(0, characters.size() - 1);

    std::string random_string;
    for (size_t i = 0; i < length; ++i) { random_string += characters[distribution(generator)]; }

    return random_string;
  }

  std::shared_ptr<RequestQueue> request_queue_;
  std::shared_ptr<ProcessingQueue> processing_queue_;
  std::shared_ptr<ResponseQueue> response_queue_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_SERVER_HPP */
