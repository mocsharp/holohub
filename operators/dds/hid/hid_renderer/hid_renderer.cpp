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

#include "hid_renderer.hpp"

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace holoscan::ops {

void HIDRendererOp::setup(OperatorSpec& spec) {
  // Inputs from DDSHIDSubscriber
  spec.input<std::vector<InputCommand>>("input");

  // Outputs to Holoviz
  spec.output<gxf::Entity>("outputs");
  spec.output<std::vector<ops::HolovizOp::InputSpec>>("output_specs");

  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(tensors_,
             "tensors",
             "Tensors",
             "Tensors configured for Holoviz.",
             std::vector<ops::HolovizOp::InputSpec>());
  spec.param(width_, "width", "Width", "Width of the Holoviz window.");
  spec.param(height_, "height", "Height", "Height of the Holoviz window.");
}

void HIDRendererOp::initialize() {
  register_converter<std::vector<ops::HolovizOp::InputSpec>>();
  Operator::initialize();

  // Initialize positions of the HID devices.
  for (const auto& input_spec : tensors_.get()) {
    if (input_spec.tensor_name_ == "") continue;
    // Store device positions in actual pixel coordinates instead of normalized 0-1 values
    auto location = std::array<float, 2>{width_.get() * 0.5f, height_.get() * 0.5f};
    tensor_locations_[input_spec.tensor_name_] = std::make_tuple(input_spec, location);
    HOLOSCAN_LOG_INFO("HID device {} initialized at x:{} y:{}",
                      input_spec.tensor_name_,
                      location[0],
                      location[1]);
  }

  // Initialize modifier key states
  shift_pressed_ = false;
  caps_lock_active_ = false;
}

void HIDRendererOp::compute(InputContext& op_input, OutputContext& op_output,
                            ExecutionContext& context) {
  auto entity = gxf::Entity::New(&context);
  auto specs = std::vector<HolovizOp::InputSpec>();

  // Get the input commands from the DDSHIDSubscriber.
  auto commands = op_input.receive<std::vector<InputCommand>>("input").value();
  process_commands(commands);
  update_tensors_specs(context, entity, specs);

  // Output to Holoviz.
  op_output.emit(entity, "outputs");
  op_output.emit(specs, "output_specs");
}

void HIDRendererOp::process_commands(const std::vector<InputCommand> commands) {
  for (const auto& command : commands) {
    const auto tensor_name = command.device_name();
    auto& location = std::get<1>(tensor_locations_[tensor_name]);
    if (command.device_type() == HIDDeviceType::JOYSTICK) {
      auto value = command.value() / 32767.0f;  // Normalize joystick input to [-1, 1]

      // Calculate step size in actual pixels
      // Default for D-pad
      float x_step = 1.0f;
      float y_step = 1.0f;

      if (command.number() == 0) {  // Left joystick x-axis
        x_step = 5.0f;
      } else if (command.number() == 1) {  // Left joystick y-axis
        y_step = 5.0f;
      } else if (command.number() == 2) {  // Right joystick x-axis
        x_step = 10.0f;
      } else if (command.number() == 5) {  // Right joystick y-axis
        y_step = 10.0f;
      }

      if (command.number() == 0 || command.number() == 2 || command.number() == 8)  // x-axis
      {
        location = {location[0] + std::round(value * x_step), location[1]};
      } else if (command.number() == 1 || command.number() == 5 || command.number() == 9)  // y-axis
      {
        location = {location[0], location[1] + std::round(value * y_step)};
      }
    } else if (command.device_type() == HIDDeviceType::KEYBOARD) {
      // Handle modifier keys first
      if (command.number() == 42 || command.number() == 54) {  // LEFT_SHIFT or RIGHT_SHIFT
        shift_pressed_ = (command.value() == 1);               // 1 when pressed, 0 when released
      } else if (command.number() == 58 && command.value() == 1) {  // CAPS_LOCK (toggle on press)
        caps_lock_active_ = !caps_lock_active_;
      }

      // Only process key press events (value=1) or key repeat events (value=2)
      if (command.value() == 1 || command.value() == 2) {
        switch (command.number()) {
          case 103:  // Up arrow (KEY_UP)
            location = {location[0], location[1] - 1.0f};
            break;
          case 108:  // Down arrow (KEY_DOWN)
            location = {location[0], location[1] + 1.0f};
            break;
          case 106:  // Right arrow (KEY_RIGHT)
            location = {location[0] + 1.0f, location[1]};
            break;
          case 105:  // Left arrow (KEY_LEFT)
            location = {location[0] - 1.0f, location[1]};
            break;
          case 14:  // Backspace (KEY_BACKSPACE)
            if (!user_text_.empty()) { user_text_.pop_back(); }
            break;
          default:
            // Determine if uppercase is needed (Shift XOR Caps Lock)
            bool use_uppercase = shift_pressed_ != caps_lock_active_;

            // Map Linux key codes to ASCII characters with proper capitalization
            char c = keycode_to_ascii(command.number(), use_uppercase);
            if (c != 0) { user_text_ += c; }
            break;
        }
      }
    } else if (command.device_type() == HIDDeviceType::MOUSE) {
      if (command.number() == 0x00) {  // EV_REL_X
        location = {location[0] + command.value(), location[1]};
      } else if (command.number() == 0x01) {  // EV_REL_Y
        location = {location[0], location[1] + command.value()};
      } else if (command.number() == 0x08 || command.number() == 0x06) {  // REL_WHEEL
        // Vertical scroll - use a multiplier for more pronounced effect
        float scroll_speed = 1.0f;  // Adjust this value based on your needs
        zoom_level_ = zoom_level_ + command.value() * scroll_speed;
      }
    }
    // Clamp values to the canvas size
    location[0] = std::clamp(location[0], 0.0f, static_cast<float>(width_.get()));
    location[1] = std::clamp(location[1], 0.0f, static_cast<float>(height_.get()));
  }
}

void HIDRendererOp::update_tensors_specs(ExecutionContext& context, gxf::Entity& entity,
                                         std::vector<HolovizOp::InputSpec>& specs) {
  int32_t priority = 0;
  for (const auto& [tensor_name, tensor_location] : tensor_locations_) {
    auto& tensor = std::get<0>(tensor_location);
    auto& location = std::get<1>(tensor_location);

    // Normalize the coordinates for Holoviz rendering
    float norm_x = location[0] / width_.get();
    float norm_y = location[1] / height_.get();

    if (tensor.type_ == HolovizOp::InputType::CROSSES) {
      add_data<1, 3>(entity, tensor_name.c_str(), {{{norm_x, norm_y, 0.10f}}}, context);

      std::stringstream ss;
      ss << std::fixed << std::setprecision(0);
      ss << tensor_name << ": [" << location[0] << "," << location[1] << "]";
      add_dynamic_text(specs,
                       entity,
                       context,
                       priority++,
                       tensor_name,
                       ss.str(),
                       {0.0f, 1.0f, 0.0f, 1.0f},
                       {{{0.01f, 0.01f}}});
    } else if (tensor.type_ == HolovizOp::InputType::OVALS) {
      add_data<1, 4>(entity, tensor_name.c_str(), {{{norm_x, norm_y, 0.05f, 0.05f}}}, context);

      std::stringstream ss;
      ss << std::fixed << std::setprecision(0);
      ss << tensor_name << ": [" << location[0] << "," << location[1] << "]: " << user_text_;
      add_dynamic_text(specs,
                       entity,
                       context,
                       priority++,
                       tensor_name,
                       ss.str(),
                       {0.5f, 0.0f, 1.0f, 1.0f},
                       {{{0.01f, 0.05f}}});
    } else if (tensor.type_ == HolovizOp::InputType::TRIANGLES) {
      // Create a triangle centered at the cursor position with size based on zoom level
      float triangle_size = 0.01f * zoom_level_;  // Base size scaled by zoom level

      // Create three points for the triangle around the center point
      std::array<std::array<float, 2>, 3> triangle_points = {{
          {norm_x, norm_y - triangle_size},                  // Top point
          {norm_x - triangle_size, norm_y + triangle_size},  // Bottom left
          {norm_x + triangle_size, norm_y + triangle_size}   // Bottom right
      }};

      add_data<3, 2>(entity, tensor_name.c_str(), triangle_points, context);

      // Add the text display with zoom level information
      std::stringstream ss;
      ss << std::fixed << std::setprecision(1);
      ss << tensor_name << ": [" << location[0] << "," << location[1] << "] Zoom: " << zoom_level_;
      add_dynamic_text(specs,
                       entity,
                       context,
                       priority++,
                       tensor_name,
                       ss.str(),
                       {1.0f, 0.0f, 1.0f, 1.0f},
                       {{{0.01f, 0.09f}}});
    }
  }
}

template <std::size_t N, std::size_t C>
void HIDRendererOp::add_data(gxf::Entity& entity, const char* name,
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

void HIDRendererOp::add_dynamic_text(std::vector<HolovizOp::InputSpec>& specs, gxf::Entity& entity,
                                     ExecutionContext& context, const int32_t priority,
                                     const std::string& tensor_name, const std::string& text,
                                     const std::vector<float> color,
                                     const std::array<std::array<float, 2UL>, 1UL>& offset) {
  HolovizOp::InputSpec spec;
  spec.tensor_name_ = tensor_name + "_text";
  spec.type_ = HolovizOp::InputType::TEXT;
  spec.color_ = color;
  spec.text_.clear();
  spec.text_.push_back(text);
  spec.priority_ = priority;
  specs.push_back(spec);
  add_data<1, 2>(entity, spec.tensor_name_.c_str(), offset, context);
}

char HIDRendererOp::keycode_to_ascii(unsigned int keycode, bool shift) {
  // Simple mapping for common keys
  static const std::map<unsigned int, std::pair<char, char>> keymap = {
      {16, {'q', 'Q'}},  {17, {'w', 'W'}}, {18, {'e', 'E'}},  {19, {'r', 'R'}}, {20, {'t', 'T'}},
      {21, {'y', 'Y'}},  {22, {'u', 'U'}}, {23, {'i', 'I'}},  {24, {'o', 'O'}}, {25, {'p', 'P'}},
      {30, {'a', 'A'}},  {31, {'s', 'S'}}, {32, {'d', 'D'}},  {33, {'f', 'F'}}, {34, {'g', 'G'}},
      {35, {'h', 'H'}},  {36, {'j', 'J'}}, {37, {'k', 'K'}},  {38, {'l', 'L'}}, {44, {'z', 'Z'}},
      {45, {'x', 'X'}},  {46, {'c', 'C'}}, {47, {'v', 'V'}},  {48, {'b', 'B'}}, {49, {'n', 'N'}},
      {50, {'m', 'M'}},

      {57, {' ', ' '}},  // Space
      {11, {'0', ')'}},  {2, {'1', '!'}},  {3, {'2', '@'}},   {4, {'3', '#'}},  {5, {'4', '$'}},
      {6, {'5', '%'}},   {7, {'6', '^'}},  {8, {'7', '&'}},   {9, {'8', '*'}},  {10, {'9', '('}},

      {12, {'-', '_'}},  {13, {'=', '+'}}, {26, {'[', '{'}},  {27, {']', '}'}}, {39, {';', ':'}},
      {40, {'\'', '"'}}, {41, {'`', '~'}}, {43, {'\\', '|'}}, {51, {',', '<'}}, {52, {'.', '>'}},
      {53, {'/', '?'}}};

  auto it = keymap.find(keycode);
  if (it != keymap.end()) { return shift ? it->second.second : it->second.first; }
  return 0;  // Return 0 for unmapped keys
}

}  // namespace holoscan::ops