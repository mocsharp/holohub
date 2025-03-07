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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>

#include "dds_hid_publisher.hpp"
#include "dds_hid_subscriber.hpp"
#include "hid_renderer.hpp"

#include "dds_video_publisher.hpp"
#include "dds_video_subscriber.hpp"

#include <getopt.h>

class RobotApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto allocator = make_resource<UnboundedAllocator>("allocator");
    // Capture video from the camera
    auto video_capture = make_operator<ops::V4L2VideoCaptureOp>(
        "video_capture", Arg("allocator") = allocator, from_config("robot.video"));

    // Subscribe to HID events
    auto hid_subscriber = make_operator<ops::DDSHIDSubscriberOp>(
        "hid_subscriber",
        make_condition<PeriodicCondition>("periodic-condition",
                                          Arg("recess_period") = std::string("60hz")),
        from_config("robot.hid"));

    // HID Renderer
    auto hid_renderer = make_operator<ops::HIDRendererOp>(
        "hid_renderer", from_config("robot.holoviz"), Arg("allocator") = allocator);

    // Render on Holoviz
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz", Arg("allocator") = allocator, from_config("robot.holoviz"));

    add_flow(video_capture, holoviz, {{"signal", "receivers"}});
    add_flow(hid_subscriber, hid_renderer, {{"output", "input"}});
    add_flow(hid_renderer, holoviz, {{"outputs", "receivers"}, {"output_specs", "input_specs"}});

    // Publish rendered video to DDS
    auto video_publisher = make_operator<ops::DDSVideoPublisherOp>(
        "video_publisher", from_config("robot.video_publisher"));

    add_flow(holoviz, video_publisher, {{"render_buffer_output", "input"}});
  }
};

class SurgeonApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Capture HID events and publish them to DDS
    auto hid_publisher =
        make_operator<ops::DDSHIDPublisherOp>("hid_publisher", from_config("surgeon.hid"));
    add_operator(hid_publisher);

    // Subscribe to the recorded video stream
    auto video_subscriber = make_operator<ops::DDSVideoSubscriberOp>(
        "video_subscriber",
        from_config("surgeon.video"),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool"));

    // Render the video stream
    auto holoviz = make_operator<ops::HolovizOp>("holoviz", from_config("surgeon.holoviz"));
    add_flow(video_subscriber, holoviz, {{"output", "receivers"}});
  }
};

void usage() {
  std::cout << "Usage: dds_video {-r | -s} [options]" << std::endl
            << std::endl
            << "Options" << std::endl
            << "  -s,       --surgeon        Run as a surgeon" << std::endl
            << "  -r,       --robot          Run as a robot" << std::endl
            << "  -c PATH,  --config=PATH    Path to the config file" << std::endl;
}

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, bool& surgeon, bool& robot, std::string& config_path) {
  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"surgeon", no_argument, 0, 's'},
                                  {"robot", no_argument, 0, 'r'},
                                  {"config", required_argument, 0, 'c'},
                                  {0, 0, 0, 0}};

  int c;
  while (optind < argc) {
    if ((c = getopt_long(argc, argv, "hsrc:", long_options, NULL)) != -1) {
      switch (c) {
        case 'h':
          usage();
          return false;
        case 's':
          surgeon = true;
          break;
        case 'r':
          robot = true;
          break;
        case 'c':
          config_path = optarg;
          break;
        default:
          HOLOSCAN_LOG_ERROR("Unhandled option '{}'", static_cast<char>(c));
      }
    }
  }

  return true;
}

int main(int argc, char** argv) {
  bool surgeon = false;
  bool robot = false;
  std::string config_path = "";

  if (!parse_arguments(argc, argv, surgeon, robot, config_path)) { return 1; }

  if (!surgeon && !robot) {
    HOLOSCAN_LOG_ERROR("Must provide either -s or -p for surgeon or robot, respectively");
    usage();
    return -1;
  }

  if (config_path.empty()) {
    // Get the input data environment variable
    auto config_file_path = std::getenv("HOLOSCAN_CONFIG_PATH");
    if (config_file_path == nullptr || config_file_path[0] == '\0') {
      auto config_file = std::filesystem::canonical(argv[0]).parent_path();
      config_path = config_file / std::filesystem::path("telesurgery.yaml");
    }
  }

  if (!std::filesystem::exists(config_path)) {
    HOLOSCAN_LOG_ERROR("Config file {} does not exist", config_path);
    return -1;
  }

  if (surgeon) {
    HOLOSCAN_LOG_INFO("Starting surgeon app with config {}", config_path);
    auto app = holoscan::make_application<SurgeonApp>();
    app->config(config_path);
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
        "scheduler", app->from_config("surgeon.scheduler")));
    app->run();
  } else if (robot) {
    HOLOSCAN_LOG_INFO("Starting robot app with config {}", config_path);
    auto app = holoscan::make_application<RobotApp>();
    app->config(config_path);
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>(
        "scheduler", app->from_config("robot.scheduler")));
    app->run();
  }

  return 0;
}
