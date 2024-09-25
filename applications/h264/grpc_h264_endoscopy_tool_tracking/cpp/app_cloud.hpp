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

#ifndef GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_CLOUD_HPP
#define GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_CLOUD_HPP

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "grpc_ops.hpp"
#include "resource_queue.hpp"

using holoscan::entity::EntityResponse;

namespace holohub::grpc_h264_endoscopy_tool_tracking {

class AppCloud : public AppBase {
 public:
  void compose() override {
    HOLOSCAN_LOG_INFO("===============AppCloud===============");
    using namespace holoscan;

    uint32_t width = 854;
    uint32_t height = 480;
    int64_t source_block_size = width * height * 3 * 4;
    int64_t source_num_blocks = 2;

    // auto bitstream_reader = make_operator<VideoReadBitstreamOp>(
    //     "bitstream_reader",
    //     from_config("bitstream_reader"),
    //     Arg("input_file_path", datapath_ + "/surgical_video.264"),
    //     make_condition<CountCondition>(750),
    //     make_condition<PeriodicCondition>("periodic-condition",
    //                                       Arg("recess_period") = std::string("25hz")),
    //     Arg("pool") =
    //         make_resource<BlockMemoryPool>(
    //             "pool", 0, source_block_size, source_num_blocks));

    // auto response_condition =
    //     make_condition<AsynchronousCondition>("response_condition");
    // auto video_decoder_context = make_resource<VideoDecoderContext>(
    //     "decoder-context", Arg("async_scheduling_term") = response_condition);

    // auto request_condition =
    //     make_condition<AsynchronousCondition>("request_condition");
    // auto video_decoder_request = make_operator<VideoDecoderRequestOp>(
    //     "video_decoder_request",
    //     from_config("video_decoder_request"),
    //     // make_condition<CountCondition>(30),
    //     Arg("async_scheduling_term") = request_condition,
    //     Arg("videodecoder_context") = video_decoder_context);

    // auto video_decoder_response = make_operator<VideoDecoderResponseOp>(
    //     "video_decoder_response",
    //     from_config("video_decoder_response"),
    //     // make_condition<CountCondition>(30),
    //     Arg("pool") =
    //         make_resource<BlockMemoryPool>(
    //             "pool", 1, source_block_size, source_num_blocks),
    //     Arg("videodecoder_context") = video_decoder_context);

    // auto decoder_output_format_converter =
    //     make_operator<ops::FormatConverterOp>("decoder_output_format_converter",
    //         from_config("decoder_output_format_converter"),
    //         Arg("pool") = make_resource<BlockMemoryPool>(
    //             "pool", 1, source_block_size, source_num_blocks));

    std::shared_ptr<BlockMemoryPool> visualizer_allocator =
        make_resource<BlockMemoryPool>(
            "allocator", 1, source_block_size, source_num_blocks);
    auto visualizer = make_operator<ops::HolovizOp>("holoviz",
        from_config("holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("enable_render_buffer_input") = false,
        Arg("enable_render_buffer_output") = false,
        Arg("allocator") = visualizer_allocator);

    // add_flow(bitstream_reader, video_decoder_request,
    //     {{"output_transmitter", "input_frame"}});
    // add_flow(video_decoder_response, decoder_output_format_converter,
    //     {{"output_transmitter", "source_video"}});
    // add_flow(decoder_output_format_converter, visualizer,
    //     {{"tensor", "receivers"}});


    auto request_available_condition =
        make_condition<AsynchronousCondition>("request_available_condition");
    request_queue_ =
        make_resource<AsynchronousConditionQueue>("request_queue", request_available_condition);
    response_queue_ =
        make_resource<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>("response_queue");

    auto grpc_request_op = make_operator<GrpcServerRequestOp>(
        "grpc_request_op",
        Arg("server_address") = std::string("0.0.0.0:50051"),
    //     make_condition<PeriodicCondition>("periodic-condition",
    //                                       Arg("recess_period") = std::string("60hz")),
        Arg("request_queue") = request_queue_,
        Arg("response_queue") = response_queue_,
        Arg("condition") = request_available_condition,
        Arg("allocator") = make_resource<UnboundedAllocator>("pool"));

    auto response_condition = make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context = make_resource<VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request =
        make_operator<VideoDecoderRequestOp>("video_decoder_request",
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

    auto rgb_float_format_converter = make_operator<ops::FormatConverterOp>(
        "rgb_float_format_converter",
        from_config("rgb_float_format_converter"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    const std::string model_file_path = datapath_ + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = datapath_ + "/engines";

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

    // auto grpc_response = make_operator<GrpcServerResponseOp>(
    //     "grpc_response", Arg("response_queue") = response_queue_);

    add_flow(grpc_request_op, video_decoder_request, {{"output", "input_frame"}});
    // add_flow(bitstream_reader, video_decoder_request,
    //     {{"output_transmitter", "input_frame"}});


    add_flow(video_decoder_response,
             decoder_output_format_converter,
             {{"output_transmitter", "source_video"}});
    add_flow(
        decoder_output_format_converter, rgb_float_format_converter, {{"tensor", "source_video"}});
    add_flow(rgb_float_format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    // add_flow(tool_tracking_postprocessor,
    //          grpc_response,
    //          {{"out_coords", "input"}, {"out_mask", "input"}});
    //     Arg("pool") = make_resource<UnboundedAllocator>("pool"));
    // add_flow(tool_tracking_postprocessor,
    //          grpc_response,
    //          {{"out_coords", "input"}, {"out_mask", "input"}});

    add_flow(tool_tracking_postprocessor,
              visualizer,
              {{"out_coords", "receivers"}, {"out_mask", "receivers"}});
    add_flow(decoder_output_format_converter, visualizer,
        {{"tensor", "receivers"}});

// visualizer,
//     //     {{"tensor", "receivers"}});

  }

 private:
  std::shared_ptr<AsynchronousConditionQueue> request_queue_;
  std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>> response_queue_;
};
}  // namespace holohub::grpc_h264_endoscopy_tool_tracking
#endif /* GRPC_H264_ENDOSCOPY_TOOL_TRACKING_CPP_APP_CLOUD_HPP */
