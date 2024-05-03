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

#include <getopt.h>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <iostream>
#include <lstm_tensor_rt_inference.hpp>
#include <string>
#include <tool_tracking_postprocessor.hpp>
#include "holoscan/holoscan.hpp"
#include "video_decoder.hpp"
#include "video_encoder.hpp"
#include "video_read_bitstream.hpp"
#include "video_write_bitstream.hpp"

#ifdef YUAN_QCAP
#include <qcap_source.hpp>
#endif

using namespace holoscan;
class VideInputFragment : public holoscan::Fragment {
 private:
  std::shared_ptr<holoscan::Operator> input_op_;
  std::string input_dir_;

 public:
  VideInputFragment(const std::string& input_dir) : input_dir_(input_dir) {}

  void init() {
    input_op_ = make_operator<ops::VideoReadBitstreamOp>(
        "bitstream_reader",
        from_config("bitstream_reader"),
        Arg("input_file_path", input_dir_ + "/surgical_video.264"),
        make_condition<CountCondition>(2000),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));
  }

  void compose() override { add_operator(input_op_); }
};

class NoOpOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NoOpOp);

  NoOpOp() = default;

  void initialize() override { Operator::initialize(); }
  void setup(OperatorSpec& spec) override {
    spec.input<holoscan::gxf::Entity>("input");
    spec.output<holoscan::gxf::Entity>("output");
  }
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) { throw std::runtime_error("Failed to receive input"); }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    op_output.emit(entity);
  }
};

class CloudInferenceFragment : public holoscan::Fragment {
 private:
  std::string model_dir_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;

 public:
  CloudInferenceFragment(const std::string& model_dir, const uint32_t width, const uint32_t height)
      : model_dir_(model_dir), width_(width), height_(height) {}

  void compose() override {
    auto no_op = make_operator<NoOpOp>("no_op");
    auto response_condition = make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context = make_resource<ops::VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request = make_operator<ops::VideoDecoderRequestOp>(
        "video_decoder_request",
        from_config("video_decoder_request"),
        request_condition,
        Arg("async_scheduling_term") = request_condition,
        Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<ops::VideoDecoderResponseOp>(
        "video_decoder_response",
        from_config("video_decoder_response"),
        response_condition,
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("videodecoder_context") = video_decoder_context);

    auto decoder_output_format_converter = make_operator<ops::FormatConverterOp>(
        "decoder_output_format_converter",
        from_config("decoder_output_format_converter"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    auto rgb_float_format_converter = make_operator<ops::FormatConverterOp>(
        "rgb_float_format_converter",
        from_config("rgb_float_format_converter"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    const std::string model_file_path = model_dir_ + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = model_dir_ + "/engines";

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        from_config("tool_tracking_postprocessor"),
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    add_flow(no_op, video_decoder_request, {{"output", "input_frame"}});
    // add_operator(video_decoder_request);
    add_flow(video_decoder_response,
             decoder_output_format_converter,
             {{"output_transmitter", "source_video"}});
    add_flow(
        decoder_output_format_converter, rgb_float_format_converter, {{"tensor", "source_video"}});
    add_flow(rgb_float_format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
  }
};

class VizFragment : public holoscan::Fragment {
 private:
  uint32_t width_ = 0;
  uint32_t height_ = 0;

 public:
  VizFragment(const uint32_t width, const uint32_t height) : width_(width), height_(height) {}

  void compose() override {
    std::shared_ptr<UnboundedAllocator> visualizer_allocator;

    auto visualizer_operator =
        make_operator<ops::HolovizOp>("holoviz",
                                      from_config("holoviz"),
                                      Arg("width") = width_,
                                      Arg("height") = height_,
                                      Arg("allocator") = visualizer_allocator);
    add_operator(visualizer_operator);
  }
};

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath_ = path; }

  void compose() override {
    using namespace holoscan;

    auto width = 854;
    auto height = 480;

    auto video_in = make_fragment<VideInputFragment>("video_in", datapath_);
    auto video_in_fragment = std::dynamic_pointer_cast<VideInputFragment>(video_in);
    video_in_fragment->init();
    auto cloud_inference =
        make_fragment<CloudInferenceFragment>("inference", datapath_, width, height);
    auto viz = make_fragment<VizFragment>("viz", width, height);

    add_flow(video_in, cloud_inference, {{"bitstream_reader.output_transmitter", "no_op.input"}});
    add_flow(
        cloud_inference, viz, {{"decoder_output_format_converter.tensor", "holoviz.receivers"}});
    add_flow(cloud_inference,
             viz,
             {{"tool_tracking_postprocessor.out_coords", "holoviz.receivers"},
              {"tool_tracking_postprocessor.out_mask", "holoviz.receivers"}});
  }

 private:
  std::string datapath_ = "data/endoscopy";
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
