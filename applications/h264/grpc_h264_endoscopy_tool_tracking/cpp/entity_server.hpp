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

using grpc::CallbackServerContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class HoloscanEntityServiceImpl final : public Entity::CallbackService {
 public:
  using on_new_request_received_callback =
      std::function<std::shared_ptr<nvidia::gxf::Entity>(EntityRequest& request)>;

  explicit HoloscanEntityServiceImpl(
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>> request_queue,
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>> response_queue,
      on_new_request_received_callback&& request_cb)
      : request_queue_(request_queue), response_queue_(response_queue), request_cb_(request_cb) {
        auto x = 1;
      }

  grpc::ServerBidiReactor<EntityRequest, EntityResponse>* EntityStream(
      CallbackServerContext* context) override {
    class EntityStreamInternal : public grpc::ServerBidiReactor<EntityRequest, EntityResponse> {
     public:
      EntityStreamInternal(HoloscanEntityServiceImpl* server) : server_(server) {
        Read();
        writer_thread_ = std::thread(&EntityStreamInternal::ProcessOutgoingQueue, this);
      }

      ~EntityStreamInternal() {
        if (writer_thread_.joinable()) { writer_thread_.join(); }
      }

      void OnWriteDone(bool ok) override {
        if (!ok) { HOLOSCAN_LOG_WARN("grpc: write failed"); }
      }

      void OnReadDone(bool ok) override {
        if (ok) {
          auto entity = server_->request_cb_(request_);
          server_->request_queue_->push(entity);
          HOLOSCAN_LOG_INFO("grpc: Request received and queued");
          Read();
        }
      }

      void OnDone() override {
        HOLOSCAN_LOG_DEBUG("grpc server: server streaming complete");
        delete this;
      }

     private:
      void Read() {
        request_.Clear();
        StartRead(&request_);
      }
      void Write() {
        if (!server_->response_queue_->empty()) {
          std::shared_ptr<EntityResponse> response;
          response = server_->response_queue_->pop();
          StartWrite(&*response);
          HOLOSCAN_LOG_INFO("grpc: Sending response");
        }
      }
      void ProcessOutgoingQueue() {
        while (true) { Write(); }
      }

      HoloscanEntityServiceImpl* server_;
      EntityRequest request_;
      std::thread writer_thread_;
    };

    return new EntityStreamInternal(this);
  }

 private:
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>> request_queue_;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>> response_queue_;
  on_new_request_received_callback request_cb_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_SERVER_HPP */
