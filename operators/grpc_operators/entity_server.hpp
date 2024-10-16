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

#ifndef ENTITY_SERVER_HPP
#define ENTITY_SERVER_HPP

#include <fmt/format.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <algorithm>
#include <functional>
#include <random>
#include <string>

#include "conditional_variable_queue.hpp"
#include "holoscan.grpc.pb.h"
#include "holoscan.pb.h"
#include "tensor_proto.hpp"

using grpc::CallbackServerContext;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::Status;
using holoscan::entity::Entity;
using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

class HoloscanEntityServiceImpl final : public Entity::CallbackService {
 public:
  using on_new_request_received_callback =
      std::function<std::shared_ptr<nvidia::gxf::Entity>(EntityRequest& request)>;

  HoloscanEntityServiceImpl(
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>> request_queue,
      std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>> response_queue,
      on_new_request_received_callback&& request_cb, uint32_t rpc_timeout)
      : request_queue_(request_queue),
        response_queue_(response_queue),
        request_cb_(request_cb),
        rpc_timeout_(rpc_timeout) {
    auto x = 1;
  }

  grpc::ServerBidiReactor<EntityRequest, EntityResponse>* EntityStream(
      CallbackServerContext* context) override {
    class EntityStreamInternal : public grpc::ServerBidiReactor<EntityRequest, EntityResponse> {
     public:
      EntityStreamInternal(HoloscanEntityServiceImpl* server) : server_(server) {
        last_network_activity_ = std::chrono::time_point<std::chrono::system_clock>::min();
        Read();
        writer_thread_ = std::thread(&EntityStreamInternal::ProcessOutgoingQueue, this);
      }

      ~EntityStreamInternal() {
        if (writer_thread_.joinable()) { writer_thread_.join(); }
      }

      void OnWriteDone(bool ok) override {
        last_network_activity_ = std::chrono::high_resolution_clock::now();
        if (!ok) { HOLOSCAN_LOG_WARN("grpc server: write failed, error writing response"); }
        write_mutext_.unlock();
      }

      void OnReadDone(bool ok) override {
        last_network_activity_ = std::chrono::high_resolution_clock::now();
        if (ok) {
          auto entity = server_->request_cb_(request_);
          server_->request_queue_->push(entity);
          HOLOSCAN_LOG_INFO("grpc server: Request received and queued for processing");
          Read();
        }
      }

      void OnDone() override {
        HOLOSCAN_LOG_INFO("grpc server: server streaming complete");
        delete this;
      }

     private:
      void Read() {
        request_.Clear();
        StartRead(&request_);
      }
      void Write() {
        if (!server_->response_queue_->empty()) {
          write_mutext_.lock();
          std::shared_ptr<EntityResponse> response;
          response = server_->response_queue_->pop();
          StartWrite(&*response);
          HOLOSCAN_LOG_INFO("grpc server: Sending response to client");
        }
      }
      void ProcessOutgoingQueue() {
        while (true) {
          if (processing_timed_out()) {
            HOLOSCAN_LOG_INFO("grpc server: sending finish event");
            Finish(grpc::Status::OK);
            break;
          }
          Write();
        }
      }

      bool processing_timed_out() {
        if (last_network_activity_ == std::chrono::time_point<std::chrono::system_clock>::min())
          return false;
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(end - last_network_activity_);
        return elapsed.count() > server_->rpc_timeout_;
      }

      HoloscanEntityServiceImpl* server_;
      EntityRequest request_;
      std::chrono::time_point<std::chrono::system_clock> last_network_activity_;

      std::mutex read_mutext_;
      std::mutex write_mutext_;
      std::thread writer_thread_;
    };

    return new EntityStreamInternal(this);
  }

 private:
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>> request_queue_;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>> response_queue_;
  on_new_request_received_callback request_cb_;
  uint32_t rpc_timeout_;
};
}  // namespace holoscan::ops

#endif /* ENTITY_SERVER_HPP */
