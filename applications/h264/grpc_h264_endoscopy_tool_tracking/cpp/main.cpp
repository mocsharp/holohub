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

#include <getopt.h>
#include <iostream>
#include <string>

#include <holoscan/holoscan.hpp>

#include "app_edge.hpp"
#include "app_cloud.hpp"

using namespace holoscan;
using namespace holohub::grpc_h264_endoscopy_tool_tracking;

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::shared_ptr<AppBase>& application,
                     std::string& data_path, std::string& config_path) {
  static struct option long_options[] = {
      {"data", required_argument, 0, 'd'}, {"config", required_argument, 0, 'c'}, {0, 0, 0, 0}};

  int c;
  while (optind < argc) {
    if ((c = getopt_long(argc, argv, "d:c:", long_options, NULL)) != -1) {
      switch (c) {
        case 'c':
          config_path = optarg;
          break;
        case 'd':
          data_path = optarg;
          break;
        default:
          holoscan::log_error("Unhandled option '{}'", static_cast<char>(c));
          return false;
      }
    } else {
      std::string command = argv[optind++];
      if (command == "edge") {
        application = holoscan::make_application<AppEdge>();
      } else if (command == "cloud") {
        application = holoscan::make_application<AppCloud>();
      }
      optind++;
    }
  }

  return true;
}

/** Main function */
int main(int argc, char** argv) {
  auto width = 854;
  auto height = 480;

  std::shared_ptr<AppBase> app = nullptr;
  // Parse the arguments
  std::string config_path = "";
  std::string data_directory = "";
  if (!parse_arguments(argc, argv, app, data_directory, config_path)) { return 1; }

  if (app == nullptr) {
    HOLOSCAN_LOG_ERROR(
        "Application not specified; use one of the following: edge, or cloud.");
    exit(-1);
  }

  if (data_directory.empty()) {
    // Get the input data environment variable
    auto input_path = std::getenv("HOLOSCAN_INPUT_PATH");
    if (input_path != nullptr && input_path[0] != '\0') {
      data_directory = std::string(input_path);
    } else if (std::filesystem::is_directory(std::filesystem::current_path() / "data/endoscopy")) {
      data_directory = std::string((std::filesystem::current_path() / "data/endoscopy").c_str());
    } else {
      HOLOSCAN_LOG_ERROR(
          "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable.");
      exit(-1);
    }
  }

  if (config_path.empty()) {
    // Get the input data environment variable
    auto config_file_path = std::getenv("HOLOSCAN_CONFIG_PATH");
    if (config_file_path == nullptr || config_file_path[0] == '\0') {
      auto config_file = std::filesystem::canonical(argv[0]).parent_path();
      config_path = config_file / std::filesystem::path("endoscopy_tool_tracking.yaml");
    } else {
      config_path = config_file_path;
    }
  }

  HOLOSCAN_LOG_INFO("Using configuration file from {}", config_path);
  app->config(config_path);

  HOLOSCAN_LOG_INFO("Using input data from {}", data_directory);
  app->set_datapath(data_directory);

  // auto scheduler = app->make_scheduler<holoscan::MultiThreadScheduler>(
  //   "event-scheduler",
  //   Arg("worker_thread_number", 3L),
  //   Arg("stop_on_deadlock", false)
  // );
  // app->scheduler(scheduler);
  app->run();

  return 0;
}
