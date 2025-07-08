# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from argparse import ArgumentParser
import cupy as cp
import holoscan as hs
import numpy as np

from holoscan.gxf import Entity
from holoscan.core import Application, Operator, Tracker, OperatorSpec
from holoscan.operators import InferenceOp, HolovizOp, VideoStreamReplayerOp, FormatConverterOp
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    RMMAllocator,
    UnboundedAllocator,
)

class FormatInferenceInputOp(Operator):
    """Operator to format input image for inference"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        # Get input message
        in_message = op_input.receive("in")

        # Transpose
        tensor = cp.asarray(in_message.get("source_video")).get()
        # OBS: Numpy conversion and moveaxis is needed to avoid strange
        # strides issue when doing inference
        tensor = np.moveaxis(tensor, 2, 0)[None]
        tensor = cp.asarray(tensor)

        # Create output message
        out_message = Entity(context)
        out_message.add(hs.as_tensor(tensor), "source_video")
        op_output.emit(out_message, "out")


class PostprocessorOp(Operator):
    """Operator that does postprocessing before sending resulting image to Holoviz"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, spec: OperatorSpec):
        """
        input:  "input_image"     - Input tensor representing the RGB image
        output: "output_image"    - The image for Holoviz to display

        Returns:
            None
        """
        spec.input("input_image")
        spec.output("output_image")

    def clamp(self, value, min_value=0, max_value=1):
        """Clamp value between [min_value, max_value]"""
        return max(min_value, min(max_value, value))

    # Update size of holoviz framer buffer which will be used to calculate self.ratio
    def framebuffer_size_callback(self, *args):
        self.framebuffer_size = args[0]

    def compute(self, op_input, op_output, context):
        # Get input message
        in_image = op_input.receive("input_image")
        image = cp.asarray(in_image.get("inference_output_decoder"))
        image = cp.transpose(image, (1, 2, 0))

        # Tensor is in range [-1, 1], so we need to scale it to [0, 255]
        image = (image - image.min()) / (image.max() - image.min())
        image = (image * 255).astype(cp.uint8)

        # Create output message
        out_message = {"image": hs.as_tensor(image)}
        op_output.emit(out_message, "output_image")

class StatsOp(Operator):
    def __init__(self, app, *args, **kwargs):
        self.encode_latency = []
        self.decode_latency = []
        self.jitter_time = []
        self.fps = []
        self.first_frame_ignored = False
        super().__init__(app, *args, **kwargs)

    def setup(self, spec):
        spec.input("input")

    def compute(self, op_input, op_output, context):
        _ = op_input.receive("input")
        if not self.first_frame_ignored:
            self.first_frame_ignored = True
            return

        self.encode_latency.append(self.metadata["video_encoder_encode_latency_ms"])
        self.decode_latency.append(self.metadata["video_decoder_decode_latency_ms"])
        self.jitter_time.append(self.metadata["jitter_time"])
        self.fps.append(self.metadata["fps"])

        print(f"Encode Latency (min, max, avg): {min(self.encode_latency):.3f}, {max(self.encode_latency):.3f}, {sum(self.encode_latency) / len(self.encode_latency):.3f}")
        print(f"Decode Latency (min, max, avg): {min(self.decode_latency):.3f}, {max(self.decode_latency):.3f}, {sum(self.decode_latency) / len(self.decode_latency):.3f}")
        print(f"Jitter Time (min, max, avg): {min(self.jitter_time):.3f}, {max(self.jitter_time):.3f}, {sum(self.jitter_time) / len(self.jitter_time):.3f}")
        print(f"FPS (min, max, avg): {min(self.fps):.3f}, {max(self.fps):.3f}, {sum(self.fps) / len(self.fps):.3f}")


class CosmosVideoTokenizeApp(Application):
    def __init__(self, data):
        """Initialize the NVIDIA Video Codec application

        Parameters
        ----------
            data: str, optional
                The path to the data directory. If not provided, the data directory will be
                set to the HOLOSCAN_INPUT_PATH environment variable.
        """
        super().__init__()

        # set name
        self.name = "Cosmos Video Tokenize App"

        if data == "none":
            data = os.environ.get("HOLOHUB_DATA_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        width = 854
        height = 480
        source_block_size = width * height * 3 * 4
        source_num_blocks = 2
        pool = UnboundedAllocator(self, name="host_allocator")

        video_dir = self.sample_data_path
        if not os.path.exists(video_dir):
            raise ValueError(f"Could not find video data: {video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=video_dir,
            allocator=RMMAllocator(self, name="video_replayer_allocator"),
            **self.kwargs("replayer"),
        )

        preprocessor = FormatConverterOp(
            self,
            name="preprocessor",
            pool=pool,
            **self.kwargs("preprocessor"),
        )
        format_inference_input = FormatInferenceInputOp(
            self,
            name="format_inference_input",
        )


        encoder_args = self.kwargs("encoder")
        encoder_args["model_path_map"] = {
            "cosmos_encoder": os.path.join(self.sample_data_path, encoder_args["model_file"])
        }
        encoder_args["allocator"] = pool
        del encoder_args["model_file"]

        encoder = InferenceOp(
            self,
            name="inference_encoder",
            **encoder_args,
        )

        decoder_args = self.kwargs("decoder")
        decoder_args["model_path_map"] = {
            "cosmos_decoder": os.path.join(self.sample_data_path, decoder_args["model_file"])
        }
        decoder_args["allocator"] = pool
        del decoder_args["model_file"]

        decoder = InferenceOp(
            self,
            name="inference_decoder",
            **decoder_args,
        )

        postprocessor = PostprocessorOp(self, name="postprocessor")


        visualizer = HolovizOp(
            self,
            name="visualizer",
            allocator=CudaStreamPool(
                self,
                name="cuda_stream",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
            ),
            **self.kwargs("holoviz"),
        )

        stats = StatsOp(self, name="stats")
        self.add_flow(source, preprocessor, {("output", "source_video")})
        self.add_flow(preprocessor, format_inference_input, {("tensor", "in")})
        self.add_flow(format_inference_input, encoder, {("out", "receivers")})
        self.add_flow(encoder, decoder, {("transmitter", "receivers")})
        self.add_flow(decoder, postprocessor, {("transmitter", "input_image")})
        self.add_flow(postprocessor, visualizer, {("output_image", "receivers")})
        # self.add_flow(encoder, stats, {("transmitter", "input")})



if __name__ == "__main__":
    default_data_path = f"{os.getcwd()}/data/endoscopy"
    # Parse args
    parser = ArgumentParser(description="Cosmos Video Tokenize demo application.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default=os.environ.get("HOLOSCAN_INPUT_PATH", default_data_path),
        help=("Set the data path (default: %(default)s)."),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "cosmos_video_tokenize.yaml")
    else:
        config_file = args.config

    # handle case where HOLOSCAN_INPUT_PATH is set with no value
    if len(args.data) == 0:
        args.data = default_data_path

    if not os.path.isdir(args.data):
        raise ValueError(
            f"Data path '{args.data}' does not exist. Use --data or set HOLOSCAN_INPUT_PATH environment variable."
        )

    app = CosmosVideoTokenizeApp(data=args.data)
    app.config(config_file)
    with Tracker(app) as tracker:
        app.run()
        tracker.print()

