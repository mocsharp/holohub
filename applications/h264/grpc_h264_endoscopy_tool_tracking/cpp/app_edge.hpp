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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_EDGE_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_EDGE_HPP

#include <holoscan/holoscan.hpp>
#include "app_base.hpp"
#include "resource_queue.hpp"
#include "video_input_fragment.hpp"
#include "viz_fragment.hpp"

namespace holohub::grpc_h264_endoscopy_tool_tracking {

using namespace holoscan;

class AppEdge : public AppBase {
 public:
  void set_datapath(const std::string& path) { datapath_ = path; }

  void compose() override {
    HOLOSCAN_LOG_INFO("===============AppEdge===============");
    using namespace holoscan;
    auto width = 854;
    auto height = 480;

    auto video_in = make_fragment<VideoInputFragment>(
        "video_in", datapath_);
    auto viz = make_fragment<VizFragment>("viz", width, height);

    add_flow(video_in,
             viz,
             {{"decoder_output_format_converter.tensor", "visualizer_op.receivers"},
              {"incoming_responses.output", "visualizer_op.receivers"}});

  }

 private:
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking

#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_EDGE_HPP */
