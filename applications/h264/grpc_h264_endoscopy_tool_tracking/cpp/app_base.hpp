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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_BASE_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_BASE_HPP

#include <holoscan/holoscan.hpp>

namespace holohub::grpc_h264_endoscopy_tool_tracking {
using namespace holoscan;

class AppBase : public holoscan::Application {
 protected:
  std::string datapath_ = "data/endoscopy";
  uint width_ = 854;
  uint height_ = 480;

 public:
  void set_datapath(const std::string& path) { datapath_ = path; }
  void set_width(const uint width) { width_ = width; }
  void set_height(const uint height) { height_ = height; }
};

}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_BASE_HPP */
