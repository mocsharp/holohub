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

#ifndef HID_DDS_HID_SUBSCRIBER_HID_RENDERER_HPP
#define HID_DDS_HID_SUBSCRIBER_HID_RENDERER_HPP

#include <array>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "InputCommand.hpp"

using namespace holoscan;

namespace holoscan::ops {

class HIDRendererOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HIDRendererOp)

  HIDRendererOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  /**
   * @brief Helper function to add a tensor with data to an entity.
   */
  template <std::size_t N, std::size_t C>
  void add_data(gxf::Entity& entity, const char* name,
                const std::array<std::array<float, C>, N>& data, ExecutionContext& context);

  void add_dynamic_text(std::vector<HolovizOp::InputSpec>& specs, gxf::Entity& entity,
                        ExecutionContext& context, const int32_t priority,
                        const std::string& tensor_name, const std::string& text,
                        const std::vector<float> color,
                        const std::array<std::array<float, 2UL>, 1UL>& offset);

  void process_commands(const std::vector<InputCommand> commands);
  void update_tensors_specs(ExecutionContext& context, gxf::Entity& entity,
                            std::vector<HolovizOp::InputSpec>& specs);
  // Function to convert Linux input key codes to ASCII characters
  char keycode_to_ascii(unsigned int keycode, bool shift);

  Parameter<std::shared_ptr<Allocator>> allocator_;
  Parameter<std::vector<ops::HolovizOp::InputSpec>> tensors_;
  Parameter<uint32_t> width_;
  Parameter<uint32_t> height_;

  std::map<std::string, std::tuple<ops::HolovizOp::InputSpec, std::array<float, 2>>>
      tensor_locations_;
  std::string user_text_;
  float zoom_level_ = 1.0f;

  // Track the last joystick value for each axis
  // Key: tensor name
  // Value: last joystick x value, x step size, y value, y step size
  std::map<std::string, std::array<float, 4>> last_joystick_values_;

  // Track modifier key states
  bool shift_pressed_;
  bool caps_lock_active_;
};

}  // namespace holoscan::ops

#endif /* HID_DDS_HID_SUBSCRIBER_HID_RENDERER_HPP */
