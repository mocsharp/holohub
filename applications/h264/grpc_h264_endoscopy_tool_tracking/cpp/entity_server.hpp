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
  enum RequestStatus { WAIT, WRITE };

  using on_new_request_received_callback = std::function<void(EntityRequest& request)>;
  using on_new_response_queued_callback =
      std::function<RequestStatus(std::shared_ptr<EntityResponse>& response)>;

  explicit HoloscanEntityServiceImpl(on_new_request_received_callback&& request_cb,
                                     on_new_response_queued_callback&& response_cb,
                                     const std::atomic<bool>& shutdown)
      : request_cb_(request_cb), response_cb_(response_cb), shutdown_(shutdown) {}

  grpc::ServerBidiReactor<EntityRequest, EntityResponse>* EntityStream(
      CallbackServerContext* context) override {
    class EntityStreamInternal : public grpc::ServerBidiReactor<EntityRequest, EntityResponse> {
     public:
      EntityStreamInternal(on_new_request_received_callback& request_cb,
                           on_new_response_queued_callback& response_cb,
                           const std::atomic<bool>& shutdown)
          : request_cb_(request_cb), response_cb_(response_cb), shutdown_(shutdown) {
        Read();
        Write();
      }

      void OnDone() override { delete this; }

      void OnReadDone(bool ok) override {
        if (!ok) {
          HOLOSCAN_LOG_WARN("Error reading request stream");
          done_reading_ = true;
          return FinishIfDone();
        }
        request_cb_(request_);
        Read();
      }

      void OnWriteDone(bool ok) override {
        if (!ok) {
          HOLOSCAN_LOG_WARN("Error writing response stream");
          done_writing_ = true;
          status_ = Status(grpc::StatusCode::UNKNOWN, "write failed");
        }
        Write();
      }

     private:
      void Read() {
        request_.Clear();
        StartRead(&request_);
      }
      void Write() {
        if (response_ != nullptr) response_->Clear();
        switch (response_cb_(response_)) {
          case RequestStatus::WRITE:
            return StartWrite(&*response_);
        }
      }

      void FinishIfDone() {
        if (!finish_sent_ && ((done_reading_ && done_writing_) || shutdown_)) {
          Finish(status_);
          finish_sent_ = true;
          return;
        }
      }
      EntityRequest request_;
      std::shared_ptr<EntityResponse> response_;
      on_new_request_received_callback request_cb_;
      on_new_response_queued_callback response_cb_;
      const std::atomic<bool>& shutdown_;
      Status status_;
      bool done_reading_ = false;
      bool done_writing_ = false;
      bool finish_sent_ = false;
    };
    return new EntityStreamInternal(request_cb_, response_cb_, std::ref(shutdown_));
  }

 private:
  on_new_request_received_callback request_cb_;
  on_new_response_queued_callback response_cb_;
  const std::atomic<bool>& shutdown_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_SERVER_HPP */
