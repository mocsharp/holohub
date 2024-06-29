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

#include <holoscan/core/resources/gxf/gxf_component_resource.hpp>
#include <holoscan/operators/gxf_codelet/gxf_codelet.hpp>

#include "holoscan/holoscan.hpp"

#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "video_encoder.hpp"

using namespace holoscan;

namespace holohub {
namespace applications {
namespace h264 {

// Import h.264 GXF codelets and components as Holoscan operators and resources
// Starting with Holoscan SDK v2.1.0, importing GXF codelets/components as Holoscan operators/
// resources can be done using the HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR and
// HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE macros. This new feature allows using GXF codelets
// and components in Holoscan applications without writing custom class wrappers (for C++) and
// Python wrappers (for Python) for each GXF codelet and component.
// For the VideoEncoderRequestOp class, since it needs to override the setup() to provide custom
// parameters and override the initialize() to register custom converters, it requires a custom
// class that extends the holoscan::ops::GXFCodeletOp class.

// The VideoDecoderResponseOp implements nvidia::gxf::VideoDecoderResponse and handles the output
// of the decoded H264 bit stream.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder context
//   Handle.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderResponseOp, "nvidia::gxf::VideoDecoderResponse")

// The VideoDecoderRequestOp implements nvidia::gxf::VideoDecoderRequest and handles the input
// for the H264 bit stream decode.
// Parameters:
// - inbuf_storage_type (uint32_t): Input Buffer storage type, 0:kHost, 1:kDevice.
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder
//   context Handle.
// - codec (uint32_t): Video codec to use, 0:H264, only H264 supported. Default:0.
// - disableDPB (uint32_t): Enable low latency decode, works only for IPPP case.
// - output_format (std::string): VidOutput frame video format, nv12pl and yuv420planar are
//   supported.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderRequestOp, "nvidia::gxf::VideoDecoderRequest")

// The VideoDecoderContext implements nvidia::gxf::VideoDecoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoDecoderContext, "nvidia::gxf::VideoDecoderContext")

// The VideoReadBitstreamOp implements nvidia::gxf::VideoReadBitStream and reads h.264 video files
// from the disk at the specified input file path.
// Parameters:
// - input_file_path (std::string): Path to image file
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data
// - outbuf_storage_type (int32_t): Output Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoReadBitstreamOp, "nvidia::gxf::VideoReadBitStream")

// The VideoWriteBitstreamOp implements nvidia::gxf::VideoWriteBitstream and writes bit stream to
// the disk at specified output path.
// Parameters:
// - output_video_path (std::string): The file path of the output video
// - frame_width (int): The width of the output video
// - frame_height (int): The height of the output video
// - inbuf_storage_type (int): Input Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoWriteBitstreamOp, "nvidia::gxf::VideoWriteBitstream")

// The VideoEncoderResponseOp implements nvidia::gxf::VideoEncoderResponse and handles the output
// of the encoded YUV frames.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - videoencoder_context (std::shared_ptr<holoscan::ops::VideoEncoderContext>): Encoder context
//   handle.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice. Default: 1.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoEncoderResponseOp, "nvidia::gxf::VideoEncoderResponse")

// The VideoEncoderContext implements nvidia::gxf::VideoEncoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoEncoderContext, "nvidia::gxf::VideoEncoderContext")

class VideInputFragment : public holoscan::Fragment {
 private:
  std::string input_dir_;

 public:
  VideInputFragment(const std::string& input_dir) : input_dir_(input_dir) {}

  void compose() override {
    auto bitstream_reader = make_operator<VideoReadBitstreamOp>(
        "bitstream_reader",
        from_config("bitstream_reader"),
        Arg("input_file_path", input_dir_ + "/surgical_video.264"),
        make_condition<CountCondition>(2000),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));
    auto response_condition = make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context = make_resource<VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request = make_operator<VideoDecoderRequestOp>(
        "video_decoder_request",
        from_config("video_decoder_request"),
        Arg("async_scheduling_term") = request_condition,
        Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<VideoDecoderResponseOp>(
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
    auto video_decoder_context = make_resource<VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request = make_operator<VideoDecoderRequestOp>(
        "video_decoder_request",
        from_config("video_decoder_request"),
        request_condition,
        Arg("async_scheduling_term") = request_condition,
        Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<VideoDecoderResponseOp>(
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

