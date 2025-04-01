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

#ifndef CAMERA_INFO_DDS_CAMERA_INFO_SUBSCRIBER_DDS_CAMERA_INFO_SUBSCRIBER_HPP
#define CAMERA_INFO_DDS_CAMERA_INFO_SUBSCRIBER_DDS_CAMERA_INFO_SUBSCRIBER_HPP


#include <dds/sub/ddssub.hpp>

#include "dds_operator_base.hpp"
#include "CameraInfo.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"
#include <holoscan/operators/holoviz/holoviz.hpp>

namespace holoscan::ops {

/**
 * @brief Operator class to subscribe to a DDS hid stream.
 */
class DDSCameraInfoSubscriberOp : public DDSOperatorBase {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(DDSCameraInfoSubscriberOp, DDSOperatorBase)

  DDSCameraInfoSubscriberOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  template <std::size_t N, std::size_t C>
  void add_data(gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context);
 
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<std::string> reader_qos_;
  Parameter<std::string> topic_;

  dds::sub::DataReader<CameraInfo> reader_ = dds::core::null;
  dds::core::cond::StatusCondition status_condition_ = dds::core::null;
  dds::core::cond::WaitSet waitset_;
};

}  // namespace holoscan::ops


#endif /* CAMERA_INFO_DDS_CAMERA_INFO_SUBSCRIBER_DDS_CAMERA_INFO_SUBSCRIBER_HPP */
