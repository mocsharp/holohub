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

from holoscan.core import Application, Operator, Tracker
from holoscan.operators import InferenceOp, HolovizOp, VideoStreamReplayerOp, FormatConverterOp
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    RMMAllocator,
    UnboundedAllocator,
)

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
            pool=BlockMemoryPool(
                self,
                name="pool",
                storage_type=MemoryStorageType.DEVICE,
                block_size=source_block_size,
                num_blocks=source_num_blocks,
            ),
            **self.kwargs("preprocessor"),
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

        # decoder = InferenceOp(
        #     self,
        #     name="inference_decoder",
        #     **decoder_args,
        # )

        # visualizer = HolovizOp(
        #     self,
        #     name="visualizer",
        #     allocator=CudaStreamPool(
        #         self,
        #         name="cuda_stream",
        #         dev_id=0,
        #         stream_flags=0,
        #         stream_priority=0,
        #         reserved_size=1,
        #         max_size=5,
        #     ),
        #     **self.kwargs("holoviz"),
        # )

        stats = StatsOp(self, name="stats")
        self.add_flow(source, preprocessor, {("output", "source_video")})
        self.add_flow(preprocessor, encoder, {("tensor", "receivers")})
        self.add_flow(encoder, stats, {("transmitter", "input")})
        # self.add_flow(encoder, decoder, {("transmitter", "receivers")})
        # self.add_flow(decoder, visualizer, {("transmitter", "receivers")})


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

