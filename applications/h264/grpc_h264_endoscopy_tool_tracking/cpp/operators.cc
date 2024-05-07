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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_OPERATORS
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_OPERATORS

#include <string>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include "holoscan/holoscan.hpp"

#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "video_decoder.hpp"
#include "video_encoder.hpp"
#include "video_read_bitstream.hpp"
#include "video_write_bitstream.hpp"

using namespace holoscan;

namespace holohub {
namespace applications {
namespace h264 {
class VideInputFragment : public holoscan::Fragment {
 private:
  std::string input_dir_;

 public:
  VideInputFragment(const std::string& input_dir) : input_dir_(input_dir) {}

  void compose() override {
    auto bitstream_reader = make_operator<ops::VideoReadBitstreamOp>(
        "bitstream_reader",
        from_config("bitstream_reader"),
        Arg("input_file_path", input_dir_ + "/surgical_video.264"),
        make_condition<CountCondition>(2000),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));
    auto response_condition = make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context = make_resource<ops::VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request = make_operator<ops::VideoDecoderRequestOp>(
        "video_decoder_request",
        from_config("video_decoder_request"),
        Arg("async_scheduling_term") = request_condition,
        Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<ops::VideoDecoderResponseOp>(
        "video_decoder_response",
        from_config("video_decoder_response"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("videodecoder_context") = video_decoder_context);

    auto decoder_output_format_converter = make_operator<ops::FormatConverterOp>(
        "decoder_output_format_converter",
        from_config("decoder_output_format_converter"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    add_flow(bitstream_reader, video_decoder_request, {{"output_transmitter", "input_frame"}});
    add_flow(video_decoder_response,
             decoder_output_format_converter,
             {{"output_transmitter", "source_video"}});
  }
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

}  // namespace h264
}  // namespace applications
}  // namespace holohub

#endif

