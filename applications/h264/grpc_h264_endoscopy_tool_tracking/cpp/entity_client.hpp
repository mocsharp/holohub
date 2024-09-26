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

#include <chrono>
#include <thread>

#include <fmt/format.h>
#include <grpcpp/grpcpp.h>
#include <gxf/std/tensor.hpp>
#include <tensor_proto.hpp>

#include "holoscan.grpc.pb.h"
#include "resource_queue.hpp"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class EntityClient {
 public:
  using on_new_response_available_callback =
      std::function<std::shared_ptr<nvidia::gxf::Entity>(EntityResponse& response)>;

  explicit EntityClient(
      const std::string& server_address,
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue,
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>> response_queue)
      : request_queue_(request_queue), response_queue_(response_queue) {
    channel_ = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
    if (auto status = channel_->GetState(false); status == GRPC_CHANNEL_TRANSIENT_FAILURE) {
      HOLOSCAN_LOG_ERROR("Error initializing channel. Please check the server address.");
      throw std::runtime_error{"Error initializing channel"};
    }
    stub_ = Entity::NewStub(channel_);
  }

  void EndoscopyToolTracking(on_new_response_available_callback&& response_cb) {
    new EntityStreamInternal(this, response_cb);
  }

 private:
  class EntityStreamInternal : public grpc::ClientBidiReactor<EntityRequest, EntityResponse> {
   public:
    EntityStreamInternal(EntityClient* client, on_new_response_available_callback& response_cb)
        : client_(client), response_cb_(response_cb) {
      client_->stub_->async()->EntityStream(&context_, this);
      AddHold();
      Write();
      Read();
      StartCall();
      writer_thread_ = std::thread(&EntityStreamInternal::ProcessOutgoingQueue, this);
    }

    ~EntityStreamInternal() { writer_thread_.join(); }

    void OnWriteDone(bool ok) override {
      // if (ok) {
      //   Write();
      // }
      // else {
      //   StartWritesDone();
      // }
      if (!ok) {
        HOLOSCAN_LOG_WARN("grpc: write failed");
        }
    }

    void OnReadDone(bool ok) override {
      if (ok) {
        auto entity = response_cb_(response_);

        client_->response_queue_->push(entity);
        HOLOSCAN_LOG_INFO("grpc: Response received and queued");
      }
    }

    void OnDone(const grpc::Status& status) override {
      if (!status.ok()) {
        HOLOSCAN_LOG_ERROR("grpc: call failed: {}", status.error_message());
        return;
      }
      HOLOSCAN_LOG_DEBUG("grpc client: client streaming complete");
      delete this;
    }

   private:
    void Read() {
      response_.Clear();
      StartRead(&response_);
    }

    void Write() {
      if (!client_->request_queue_->empty()) {
        std::shared_ptr<EntityRequest> request;
        request = client_->request_queue_->pop();
        request->set_service("endoscopy_tool_tracking");
        StartWrite(&*request);
        HOLOSCAN_LOG_INFO("grpc: Sending request");
      }
    }

    void ProcessOutgoingQueue() {
      while (true) { Write(); }
    }

    EntityClient* client_;
    EntityResponse response_;
    ClientContext context_;
    on_new_response_available_callback response_cb_;

    std::mutex request_mutex_;
    std::condition_variable request_cv_;
    std::thread writer_thread_;
  };

  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue_;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>> response_queue_;

  std::shared_ptr<Channel> channel_;
  std::unique_ptr<Entity::Stub> stub_;
  EntityStreamInternal* reactor_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_CLIENT_HPP */
