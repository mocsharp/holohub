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
#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_RESOURCE_QUEUE_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_RESOURCE_QUEUE_HPP

#include <memory>
#include <queue>

#include <gxf/std/tensor.hpp>
#include <holoscan/holoscan.hpp>

#include <holoscan.pb.h>

using namespace std;

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class RequestQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(RequestQueue, Resource)

  RequestQueue() { queue_ = new queue<tuple<const string, const EntityRequest*>>(); }

  ~RequestQueue() { delete queue_; }

  void push(const string request_id, const EntityRequest* request) {
    queue_->push(std::make_tuple(request_id, request));
    lock_guard<mutex> lock(request_available_mutex_);
    request_available_condition_.notify_all();
  }

  void block_until_data_available() {
    unique_lock<mutex> lock(request_available_mutex_);
    request_available_condition_.wait(lock, [this] { return this->queue_->size() > 0; });
  }

  tuple<const string, const EntityRequest*> pop() {
    auto item = queue_->front();
    queue_->pop();
    return item;
  }

 private:
  queue<tuple<const string, const EntityRequest*>>* queue_;
  condition_variable request_available_condition_;
  mutex request_available_mutex_;
};

class ProcessingQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ProcessingQueue, Resource)

  ProcessingQueue() : queue_() {}

  void push(const string value) { queue_.push_back(value); }
  bool contains(const string value) {
    return find(queue_.begin(), queue_.end(), value) != queue_.end();
  }

 private:
  deque<string> queue_;
};

class ResponseQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ResponseQueue, Resource)

  ResponseQueue() : queue_() {}

  void push(const nvidia::gxf::Entity value) {
    queue_.push(value);
    lock_guard<mutex> lock(response_available_mutex_);
    response_available_condition_.notify_all();
  }

  const nvidia::gxf::Entity pop() {
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  void block_until_data_available() {
    unique_lock<mutex> lock(response_available_mutex_);
    response_available_condition_.wait(lock, [this] { return this->queue_.size() > 0; });
  }

 private:
  queue<nvidia::gxf::Entity> queue_;
  condition_variable response_available_condition_;
  mutex response_available_mutex_;
};

}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_RESOURCE_QUEUE_HPP */
