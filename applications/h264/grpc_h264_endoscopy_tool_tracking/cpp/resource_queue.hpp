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

class AsynchronousConditionQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(AsynchronousConditionQueue, Resource)

  explicit AsynchronousConditionQueue(shared_ptr<AsynchronousCondition> request_available_condition)
      : data_available_condition_(request_available_condition) {
    queue_ = new queue<nvidia::gxf::Entity>();
  }

  ~AsynchronousConditionQueue() { delete queue_; }

  void push(nvidia::gxf::Entity entity) {
    queue_->push(entity);
    if (data_available_condition_->event_state() == AsynchronousEventState::EVENT_WAITING) {
      data_available_condition_->event_state(AsynchronousEventState::EVENT_DONE);
    }
  }

  nvidia::gxf::Entity pop() {
    auto item = queue_->front();
    queue_->pop();
    return item;
  }

 private:
  shared_ptr<AsynchronousCondition> data_available_condition_;
  queue<nvidia::gxf::Entity>* queue_;
};

template <typename DataT>
class ConditionVariableQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ConditionVariableQueue, Resource)

  ConditionVariableQueue() : queue_() {}

  void push(DataT value) {
    queue_.push(value);
    lock_guard<mutex> lock(response_available_mutex_);
    response_available_condition_.notify_all();
  }

  DataT pop() {
    std::unique_lock<std::mutex> lock(response_available_mutex_);
    response_available_condition_.wait(lock, [this]() { return !queue_.empty(); });
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  bool empty() { return queue_.empty(); }

 private:
  queue<DataT> queue_;
  condition_variable response_available_condition_;
  mutex response_available_mutex_;
};

}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_RESOURCE_QUEUE_HPP */
