/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING

#include <getopt.h>

// #include <holoscan/operators/format_converter/format_converter.hpp>
// #include <holoscan/operators/holoviz/holoviz.hpp>
#include <iostream>
#include <memory>
#include <string>

#include "holoscan/holoscan.hpp"

#include "entity_service_server.cc"
#include "operators.cc"

using namespace holoscan;
using namespace holohub::applications::h264;


class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath_ = path; }
  void compose() override {
    auto width = 854;
    auto height = 480;

    auto video_in = make_fragment<VideInputFragment>("video_in", datapath_);
    auto video_in_fragment = std::dynamic_pointer_cast<VideInputFragment>(video_in);
    auto cloud_inference =
        make_fragment<CloudInferenceFragment>("inference", datapath_, width, height);
    auto viz = make_fragment<VizFragment>("viz", width, height);

    add_flow(video_in, cloud_inference, {{"bitstream_reader.output_transmitter", "no_op.input"}});
    add_flow(video_in, viz, {{"decoder_output_format_converter.tensor", "holoviz.receivers"}});
    add_flow(cloud_inference,
             viz,
             {{"tool_tracking_postprocessor.out_coords", "holoviz.receivers"},
              {"tool_tracking_postprocessor.out_mask", "holoviz.receivers"}});
    server_->Start();
  }

 private:
  std::string datapath_ = "data/endoscopy";
  std::unique_ptr<EntityServiceServer> server_;
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

/** Main function */
int main(int argc, char** argv) {
  auto libpath = std::getenv("LD_LIBRARY_PATH");
  auto hlibpath = std::getenv("HOLOSCAN_LIB_PATH");
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/endoscopy_tool_tracking.yaml";
    app->config(config_path);
    HOLOSCAN_LOG_INFO("Using config file from {}", config_path.string());
  }

  if (data_path != "") {
    app->set_datapath(data_path);
    HOLOSCAN_LOG_INFO("Using video from {}", data_path);
  } else {
    auto data_directory = std::getenv("HOLOSCAN_INPUT_PATH");
    if (data_directory != nullptr && data_directory[0] != '\0') {
      app->set_datapath(data_directory);
      HOLOSCAN_LOG_INFO("Using video from {}", data_directory);
    }
  }
  app->run();

  return 0;
}

#endif
