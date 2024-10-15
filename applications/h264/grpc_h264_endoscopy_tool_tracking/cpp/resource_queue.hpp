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

using namespace holoscan;

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

template <typename DataT>
class AsynchronousConditionQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(AsynchronousConditionQueue, Resource)

  explicit AsynchronousConditionQueue(
      std::shared_ptr<AsynchronousCondition> request_available_condition)
      : queue_(), data_available_condition_(request_available_condition) {}

  void push(DataT entity) {
    queue_.push(entity);
    if (data_available_condition_->event_state() == AsynchronousEventState::EVENT_WAITING) {
      data_available_condition_->event_state(AsynchronousEventState::EVENT_DONE);
    }
  }

  DataT pop() {
    if (empty()) { return nullptr; }
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  bool empty() { return queue_.empty(); }

 private:
  std::shared_ptr<AsynchronousCondition> data_available_condition_;
  std::queue<DataT> queue_;
};

template <typename DataT>
class ConditionVariableQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ConditionVariableQueue, Resource)

  ConditionVariableQueue() : queue_() {}

  void push(DataT value) {
    std::lock_guard<std::mutex> lock(response_available_mutex_);
    queue_.push(value);
    data_available_condition_.notify_one();
  }

  DataT pop() {
    std::unique_lock<std::mutex> lock(response_available_mutex_);
    data_available_condition_.wait(lock, [this]() { return !queue_.empty(); });
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  bool empty() {
    std::lock_guard<std::mutex> lock(response_available_mutex_);
    return queue_.empty();
  }

 private:
  std::queue<DataT> queue_;
  std::condition_variable data_available_condition_;
  std::mutex response_available_mutex_;
};

}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_RESOURCE_QUEUE_HPP */
