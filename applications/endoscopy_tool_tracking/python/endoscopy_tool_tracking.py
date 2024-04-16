# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Application, Fragment
from holoscan.operators import (
    AJASourceOp,
    FormatConverterOp,
    HolovizOp,
    VideoStreamReplayerOp,
)
from holoscan.resources import (
    BlockMemoryPool,
    CudaStreamPool,
    MemoryStorageType,
    UnboundedAllocator,
)

from holohub.lstm_tensor_rt_inference import LSTMTensorRTInferenceOp

# Enable this line for Yuam capture card
# from holohub.qcap_source import QCAPSourceOp
from holohub.tool_tracking_postprocessor import ToolTrackingPostprocessorOp


class VideoInputFragment(Fragment):
    def __init__(self, app, name, video_dir):
        super().__init__(app, name)
        self._video_dir = video_dir

        if not os.path.exists(self._video_dir):
            raise ValueError(f"Could not find video data: {self._video_dir=}")

    def compose(self):
        if not os.path.exists(self._video_dir):
            raise ValueError(f"Could not find video data: {self._video_dir=}")
        source = VideoStreamReplayerOp(
            self,
            name="replayer",
            directory=self._video_dir,
            **self.kwargs("replayer"),
        )

        self.add_operator(source)


class CloudInferenceFragment(Fragment):
    def __init__(self, app, name, model_dir, width, height):
        super().__init__(app, name)
        self._model_dir = model_dir
        self._width = width
        self._height = height

    def compose(self):
        config_key_name = "format_converter_replayer"
        self.rdma = False
        # 4 bytes/channel, 3 channels
        source_block_size = self._width * self._height * 3 * 4
        source_num_blocks = 2

        source_pool_kwargs = dict(
            storage_type=MemoryStorageType.DEVICE,
            block_size=source_block_size,
            num_blocks=source_num_blocks,
        )

        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        format_converter = FormatConverterOp(
            self,
            name="format_converter",
            pool=BlockMemoryPool(self, name="pool", **source_pool_kwargs),
            cuda_stream_pool=cuda_stream_pool,
            **self.kwargs(config_key_name),
        )

        lstm_inferer_block_size = 107 * 60 * 7 * 4
        lstm_inferer_num_blocks = 2 + 5 * 2
        model_file_path = os.path.join(self._model_dir, "tool_loc_convlstm.onnx")
        engine_cache_dir = os.path.join(self._model_dir, "engines")
        lstm_inferer = LSTMTensorRTInferenceOp(
            self,
            name="lstm_inferer",
            pool=BlockMemoryPool(
                self,
                name="device_allocator",
                storage_type=MemoryStorageType.DEVICE,
                block_size=lstm_inferer_block_size,
                num_blocks=lstm_inferer_num_blocks,
            ),
            cuda_stream_pool=cuda_stream_pool,
            model_file_path=model_file_path,
            engine_cache_dir=engine_cache_dir,
            **self.kwargs("lstm_inference"),
        )

        # tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4
        # tool_tracking_postprocessor_num_blocks = 2
        tool_tracking_postprocessor = ToolTrackingPostprocessorOp(
            self,
            name="tool_tracking_postprocessor",
            device_allocator=UnboundedAllocator(
                self,
                name="device_allocator",
                # storage_type=MemoryStorageType.DEVICE,
                # block_size=tool_tracking_postprocessor_block_size,
                # num_blocks=tool_tracking_postprocessor_num_blocks,
            ),
            host_allocator=UnboundedAllocator(self, name="host_allocator"),
        )

        self.add_flow(format_converter, lstm_inferer)
        self.add_flow(lstm_inferer, tool_tracking_postprocessor, {("tensor", "in")})


class VizFragment(Fragment):
    def __init__(self, app, name, width, height):
        super().__init__(app, name)
        self._width = width
        self._height = height

    def compose(self):
        cuda_stream_pool = CudaStreamPool(
            self,
            name="cuda_stream",
            dev_id=0,
            stream_flags=0,
            stream_priority=0,
            reserved_size=1,
            max_size=5,
        )

        visualizer = HolovizOp(
            self,
            name="holoviz",
            width=self._width,
            height=self._height,
            allocator=None,
            # cuda_stream_pool=cuda_stream_pool,
            **self.kwargs("holoviz"),
        )
        self.add_operator(visualizer)


class EndoscopyApp(Application):
    def __init__(self, data):
        """Initialize the endoscopy tool tracking application"""
        super().__init__()

        # set name
        self.name = "Endoscopy App"

        if data == "none":
            data = os.environ.get("HOLOSCAN_INPUT_PATH", "../data")

        self.sample_data_path = data

    def compose(self):
        width = 854
        height = 480
        source = VideoInputFragment(self, "video_in", self.sample_data_path)
        cloud = CloudInferenceFragment(
            self, name="inference", model_dir=self.sample_data_path, width=width, height=height
        )
        viz = VizFragment(self, "viz", width, height)

        # Flow definition
        self.add_flow(
            source,
            cloud,
            {("replayer", "format_converter")}
        )
        self.add_flow(
            cloud,
            viz,
            {
                ("tool_tracking_postprocessor.out_coords", "holoviz.receivers"),
                ("tool_tracking_postprocessor.out_mask", "holoviz.receivers"),
            },
        )
        self.add_flow(
            source,
            viz,
            {("replayer.output", "holoviz.receivers")},
        )


if __name__ == "__main__":
    # get the Application's arguments
    app_argv = Application().argv

    # Parse args
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    parser.add_argument(
        "-d",
        "--data",
        default="none",
        help=("Set the data path"),
    )
    args = parser.parse_args(app_argv[1:])

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "endoscopy_tool_tracking.yaml")
    else:
        config_file = args.config

    app = EndoscopyApp(data=args.data)
    app.config(config_file)
    app.run()
