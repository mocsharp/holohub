/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dds_camera_info_subscriber.hpp"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <sstream>
#include "dds/topic/find.hpp"
#include "gxf/multimedia/video.hpp"
#include "holoscan/core/gxf/entity.hpp"

#define CUDA_TRY(stmt)                                                                       \
  {                                                                                          \
    cudaError_t cuda_status = stmt;                                                          \
    if (cudaSuccess != cuda_status) {                                                        \
      HOLOSCAN_LOG_ERROR("CUDA runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                              \
                         __LINE__,                                                           \
                         __FILE__,                                                           \
                         cudaGetErrorString(cuda_status),                                    \
                         int(cuda_status));                                                  \
      throw std::runtime_error("CUDA runtime call failed");                                  \
    }                                                                                        \
  }

namespace holoscan::ops {

void DDSCameraInfoSubscriberOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  // Change output type to Tensor
  spec.output<nvidia::gxf::Entity>("video");
  spec.output<nvidia::gxf::Entity>("overlay");
  spec.output<std::vector<ops::HolovizOp::InputSpec>>("overlay_specs");


  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
  spec.param(topic_, "topic", "Topic", "Topic name", std::string("topic_wrist_camera_data_rgb"));
}

void DDSCameraInfoSubscriberOp::initialize() {
  register_converter<std::vector<ops::HolovizOp::InputSpec>>();
  DDSOperatorBase::initialize();

  // Create the subscriber
  dds::sub::Subscriber subscriber(participant_);

  // Create the CameraInfo topic
  auto topic = dds::topic::find<dds::topic::Topic<CameraInfo>>(participant_, topic_.get());
  if (topic == dds::core::null) {
    topic = dds::topic::Topic<CameraInfo>(participant_, topic_.get());
  }

  // Create the reader for the CameraInfo
  reader_ = dds::sub::DataReader<CameraInfo>(
      subscriber, topic, qos_provider_.datareader_qos(reader_qos_.get()));

  // Obtain the reader's status condition
  status_condition_ = dds::core::cond::StatusCondition(reader_);

  // Enable the 'data available' status
  status_condition_.enabled_statuses(dds::core::status::StatusMask::data_available());

  // Attach the status condition to the waitset
  waitset_ += status_condition_;
}

void DDSCameraInfoSubscriberOp::compute(InputContext& op_input, OutputContext& op_output,
                                        ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("DDSCameraInfoSubscriberOp::compute");
  // record start time
  auto start_time = std::chrono::high_resolution_clock::now();

  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());

  auto output = nvidia::gxf::Entity::New(context.context());
  if (!output) { throw std::runtime_error("Failed to allocate message for output"); }

  auto overlay_entity = gxf::Entity::New(&context);
  if (!overlay_entity) { throw std::runtime_error("Failed to allocate overlay entity"); }
  auto overlay_specs = std::vector<HolovizOp::InputSpec>();

  bool output_written = false;
  auto start_time2 = std::chrono::high_resolution_clock::now();
  // Wait for a new frame
  // record time for the take call
  auto start_time_take = std::chrono::high_resolution_clock::now();
  dds::sub::LoanedSamples<CameraInfo> frames = reader_.take();
  auto end_time_take = std::chrono::high_resolution_clock::now();
  auto duration_take =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time_take - start_time_take);
  HOLOSCAN_LOG_INFO("Time taken to take CameraInfo: {} μs", duration_take.count());

  // record time between here and first line of loop
  auto start_time_loop = std::chrono::high_resolution_clock::now();
  for (const auto& frame : frames) {
    HOLOSCAN_LOG_INFO("Received CameraInfo: {}", frames.length());
    auto end_time_loop = std::chrono::high_resolution_clock::now();
    auto duration_loop =
        std::chrono::duration_cast<std::chrono::microseconds>(end_time_loop - start_time_loop);
    HOLOSCAN_LOG_INFO("Time taken to loop through frames: {} μs", duration_loop.count());

    // record time to check valid call
    auto start_time_valid = std::chrono::high_resolution_clock::now();
    if (frame.info().valid()) {
      auto end_time_valid = std::chrono::high_resolution_clock::now();
      auto duration_valid =
          std::chrono::duration_cast<std::chrono::microseconds>(end_time_valid - start_time_valid);
      HOLOSCAN_LOG_INFO("Time taken to check valid: {} μs", duration_valid.count());

      // record start time
      auto shape = nvidia::gxf::Shape{
          static_cast<int>(frame.data().height()), static_cast<int>(frame.data().width()), 3};

      auto tensor = output.value().add<nvidia::gxf::Tensor>("");
      if (!tensor) { throw std::runtime_error("Failed to allocate tensor"); }

      // Allocate memory and reshape the tensor
      tensor.value()->reshape<uint8_t>(
          shape, nvidia::gxf::MemoryStorageType::kDevice, allocator.value());

      // Copy the data instead of wrapping it
      auto type = nvidia::gxf::PrimitiveType::kUnsigned8;
      auto bytes_per_element = nvidia::gxf::PrimitiveTypeSize(type);
      size_t data_size = frame.data().width() * frame.data().height() * 3 * bytes_per_element;

      // record time to copy data to tensor
      auto start_time_tensor = std::chrono::high_resolution_clock::now();
      // Copy data from frame to tensor
      CUDA_TRY(cudaMemcpy(tensor.value()->pointer(),
                          frame.data().data().data(),
                          data_size,
                          cudaMemcpyHostToDevice));

      // record time to copy data to tensor
      // calculate and print time elapsed in microseconds
      auto end_time_tensor = std::chrono::high_resolution_clock::now();
      auto duration_tensor = std::chrono::duration_cast<std::chrono::microseconds>(
          end_time_tensor - start_time_tensor);
      HOLOSCAN_LOG_INFO("Time taken to copy data to tensor: {} μs", duration_tensor.count());

      // generte overlay specs
      std::stringstream ss;
      ss << "Robot Index: " << frame.data().robot_index();
      HolovizOp::InputSpec robot_index_spec;
      robot_index_spec.tensor_name_ = "robot_index";
      robot_index_spec.type_ = HolovizOp::InputType::TEXT;
      robot_index_spec.color_ = {0.0f, 1.0f, 0.0f};
      robot_index_spec.text_.clear();
      robot_index_spec.text_.push_back(ss.str());
      robot_index_spec.priority_ = 1;
      overlay_specs.push_back(robot_index_spec);
      add_data<1, 2>(overlay_entity, "robot_index", {{{0.01f, 0.01f}}}, context);

      ss.str("");
      ss.clear();
      for (auto index = 0 ; index < frame.data().joint_names().size(); index++) {
        ss << frame.data().joint_names()[index] << ": " << frame.data().joint_positions()[index] << "\n";
      }
      HolovizOp::InputSpec joint_positions_spec;
      joint_positions_spec.tensor_name_ = "joint_positions";
      joint_positions_spec.type_ = HolovizOp::InputType::TEXT;
      joint_positions_spec.color_ = {1.0f, 0.0f, 0.0f};
      joint_positions_spec.text_.clear();
      joint_positions_spec.text_.push_back(ss.str());
      joint_positions_spec.priority_ = 1;
      overlay_specs.push_back(joint_positions_spec);
      add_data<1, 2>(overlay_entity, "joint_positions", {{{0.01f, 0.06f}}}, context);

      output_written = true;
    } else {
      HOLOSCAN_LOG_INFO("Invalid CameraInfo");
    }
  }


  if (output_written) {
    // Output the buffer
    auto result = gxf::Entity(std::move(output.value()));
    op_output.emit(result, "video");
    op_output.emit(overlay_entity, "overlay");
    op_output.emit(overlay_specs, "overlay_specs");
    HOLOSCAN_LOG_INFO("Emit complete");
  }

  // calculate and print time elapsed
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  HOLOSCAN_LOG_INFO("Time taken to compute: {} μs", duration.count());
}


template <std::size_t N, std::size_t C>
void DDSCameraInfoSubscriberOp::add_data(gxf::Entity& entity, const char* name,
                             const std::array<std::array<float, C>, N>& data,
                             ExecutionContext& context) {
  // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  // Add a tensor
  auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
  // Reshape the tensor to the size of the data
  tensor->reshape<float>(
      nvidia::gxf::Shape({N, C}), nvidia::gxf::MemoryStorageType::kHost, allocator.value());
  // Copy the data to the tensor
  std::memcpy(tensor->pointer(), data.data(), N * C * sizeof(float));
}

}  // namespace holoscan::ops
