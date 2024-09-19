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
#include "resource_queue.hpp"

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class EntityClient {
 public:
  explicit EntityClient(
      std::shared_ptr<Channel> channel,
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue,
      std::shared_ptr<AsynchronousConditionQueue> response_queue)
      : stub_(Entity::NewStub(channel)),
        request_queue_(request_queue),
        response_queue_(response_queue) {}

  void EndoscopyToolTracking(std::shared_ptr<Allocator> allocator, void* gxf_context) {
    class EntityStreamInternal : public grpc::ClientBidiReactor<EntityRequest, EntityResponse> {
     public:
      explicit EntityStreamInternal(
          Entity::Stub* stub,
          std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue,
          std::shared_ptr<AsynchronousConditionQueue> response_queue,
          std::shared_ptr<Allocator> allocator, void* gxf_context)
          : request_queue_(request_queue),
            response_queue_(response_queue),
            allocator_(allocator),
            gxf_context_(gxf_context) {
        stub->async()->EntityStream(&context_, this);
        NextWrite();
        StartRead(&response_);
        StartCall();
      }
      void OnWriteDone(bool /*ok*/) override { NextWrite(); }
      void OnReadDone(bool ok) override {
        if (ok) {
          auto out_message = nvidia::gxf::Entity::New(gxf_context_);
          auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
              gxf_context_, allocator_->gxf_cid());
          holoscan::ops::TensorProto::entity_response_to_tensor(
              response_, out_message.value(), gxf_allocator.value());
          response_queue_->push(out_message.value());
          StartRead(&response_);
        }
      }
      void OnDone(const Status& s) override {
        std::unique_lock<std::mutex> l(mu_);
        status_ = s;
        done_ = true;
        cv_.notify_one();
      }
      Status Await() {
        std::unique_lock<std::mutex> l(mu_);
        cv_.wait(l, [this] { return done_; });
        return std::move(status_);
      }

     private:
      void NextWrite() {
        if (!request_queue_->empty()) {
          auto request = request_queue_->pop();
          // request.set_service("endoscopy_tool_tracking");
          StartWrite(&*request);
        }
        // else {
        //  StartWritesDone();
        //  }
      }
      ClientContext context_;
      EntityResponse response_;
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue_;
      std::shared_ptr<AsynchronousConditionQueue> response_queue_;
      std::shared_ptr<Allocator> allocator_;
      void* gxf_context_;
      std::mutex mu_;
      std::condition_variable cv_;
      Status status_;
      bool done_ = false;
    };

    EntityStreamInternal entity_stream(
        stub_.get(), request_queue_, response_queue_, allocator, gxf_context);
    Status status = entity_stream.Await();
    if (!status.ok()) {
      HOLOSCAN_LOG_ERROR("endoscopy_tool_tracking rpc failed: {}", status.error_message());
    }
  }

 private:
  std::unique_ptr<Entity::Stub> stub_;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue_;
  std::shared_ptr<AsynchronousConditionQueue> response_queue_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_ENTITY_CLIENT_HPP */
