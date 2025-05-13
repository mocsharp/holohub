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

import logging
from typing import Callable
import queue
import time
from holoscan.conditions import AsynchronousCondition, AsynchronousEventState, PeriodicCondition
from holoscan.core import Operator, OperatorSpec, Tensor,Application
from holoscan.operators import FormatConverterOp, HolovizOp, GXFCodeletOp
from holoscan.resources import (
    GXFComponentResource,
    UnboundedAllocator,
    BlockMemoryPool,
    MemoryStorageType,
)
from holohub.tensor_to_video_buffer import TensorToVideoBufferOp
from holohub.reshape_op import ReshapeOp
import warp
import numpy as np
import cupy as cp

class AsyncDataPushForDDS(Operator):
    def __init__(self, fragment, max_queue_size: int = 0, condition: AsynchronousCondition = None, *args, **kwargs):
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._logger = logging.getLogger(__name__)
        self._printed_image_size = False
        self.condition = condition
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("image")

    def start(self):
        self.condition.event_state = AsynchronousEventState.EVENT_WAITING

    def stop(self):
        self.condition.event_state = AsynchronousEventState.EVENT_NEVER

    def compute(self, op_input, op_output, context):
        data = self._queue.get()

        print(f"Type: {type(data)}")
        op_output.emit({"": Tensor.as_tensor(data)}, "image")

        self.condition.event_state = AsynchronousEventState.EVENT_WAITING

    def push_data(self, data):
        self._queue.put(data)
        self.condition.event_state = AsynchronousEventState.EVENT_DONE


class VideoEncoderContext(GXFComponentResource):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoEncoderContext", *args, **kwargs)

class VideoEncoderRequestOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoEncoderRequest", *args, **kwargs)

class VideoEncoderResponseOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoEncoderResponse", *args, **kwargs)

class VideoDecoderRequestOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoDecoderRequest", *args, **kwargs)

class VideoDecoderResponseOp(GXFCodeletOp):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoDecoderResponse", *args, **kwargs)


class VideoDecoderContext(GXFComponentResource):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, "nvidia::gxf::VideoDecoderContext", *args, **kwargs)
class PatientApp(Application):
    """A Holoscan application for transmitting data over RoCE (RDMA over Converged Ethernet).

    This application sets up a data transmission pipeline that can either transmit data
    over a RoCE network interface or display the data locally using Holoviz if no RoCE
    device is available.

    Args:
        ibv_name (str): Name of the InfiniBand verb (IBV) device to use for RoCE transmission.
        ibv_port (int): Port number for the IBV device.
        hololink_ip (str): IP address of the Hololink receiver.
        ibv_qp (int): Queue pair number for the IBV device.
        tx_queue_size (int): Size of the transmission queue.
        buffer_size (int): Size of the buffer for data transmission.
    """

    def __init__(
        self,
        hid_event_callback: Callable,
    ):
        """Initialize the TransmitterApp.

        Args:
            tx_queue_size (int): Size of the transmission queue.
            buffer_size (int): Size of the buffer for data transmission.
        """
        self._hid_event_callback = hid_event_callback
        self._logger = logging.getLogger(__name__)

        self._async_data_push = None
        super().__init__()

    def compose(self):
        """Compose the application workflow.

        Sets up the data transmission pipeline by creating and connecting the necessary operators.
        If a RoCE device is available, creates a RoceTransmitterOp for network transmission.
        Otherwise, creates a HolovizOp for local visualization.
        """

        source_block_size = 1920 * 1080 * 3 * 4
        source_num_blocks = 2
        source_rate_hz = 60  # messages sent per second
        period_source_ns = int(1e9 / source_rate_hz)  # period in nanoseconds

        if True:
            self._async_data_push = AsyncDataPushForDDS(
                self,
                name="Async Data Push",
                condition=AsynchronousCondition(self)
            )

            rgba_to_rgb_format_converter = FormatConverterOp(
                self,
                name="rgba_to_rgb_format_converter",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("rgba_to_rgb_format_converter"),
            )

            rgb_to_yuv420_format_converter = FormatConverterOp(
                self,
                name="rgb_to_yuv420_format_converter",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("rgb_to_yuv420_format_converter"),
            )

            tensor_to_video_buffer = TensorToVideoBufferOp(
                self, name="tensor_to_video_buffer", **self.kwargs("tensor_to_video_buffer")
            )
            encoder_async_condition = AsynchronousCondition(self, "encoder_async_condition")
            video_encoder_context = VideoEncoderContext(
                self, scheduling_term=encoder_async_condition
            )
            video_encoder_request = VideoEncoderRequestOp(
                self,
                name="video_encoder_request",
                videoencoder_context=video_encoder_context,
                **self.kwargs("video_encoder_request"),
            )
            video_encoder_response = VideoEncoderResponseOp(
                self,
                name="video_encoder_response",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                videoencoder_context=video_encoder_context,
                **self.kwargs("video_encoder_response"),
            )

            response_condition = AsynchronousCondition(self, "response_condition")
            video_decoder_context = VideoDecoderContext(self, async_scheduling_term=response_condition)

            request_condition = AsynchronousCondition(self, "request_condition")
            video_decoder_request = VideoDecoderRequestOp(
                self,
                name="video_decoder_request",
                async_scheduling_term=request_condition,
                videodecoder_context=video_decoder_context,
                **self.kwargs("video_decoder_request"),
            )

            video_decoder_response = VideoDecoderResponseOp(
                self,
                name="video_decoder_response",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                videodecoder_context=video_decoder_context,
                **self.kwargs("video_decoder_response"),
            )

            decoder_output_format_converter = FormatConverterOp(
                self,
                name="decoder_output_format_converter",
                pool=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ),
                **self.kwargs("decoder_output_format_converter"),
            )

            visualizer = HolovizOp(
                self,
                name="visualizer",
                window_title="Encode & Decode",
                width=300,
                height=300,
                tensors=[
                    HolovizOp.InputSpec("", HolovizOp.InputType.COLOR),
                ],
            )

            self.add_flow(self._async_data_push, rgba_to_rgb_format_converter, {("image", "source_video")})
            self.add_flow(rgba_to_rgb_format_converter, rgb_to_yuv420_format_converter, {("tensor", "source_video")})
            self.add_flow(rgb_to_yuv420_format_converter, tensor_to_video_buffer, {("tensor", "in_tensor")})
            self.add_flow(tensor_to_video_buffer, video_encoder_request, {("out_video_buffer", "input_frame")})
            
            reshape_op = ReshapeOp(self, name="reshape_op", allocator=BlockMemoryPool(
                    self,
                    name="pool",
                    storage_type=MemoryStorageType.DEVICE,
                    block_size=source_block_size,
                    num_blocks=source_num_blocks,
                ), out_storage_type=1)
            self.add_flow(
                video_encoder_response, reshape_op, {("output_transmitter", "in")}
            )
            self.add_flow(
                reshape_op, video_decoder_request, {("out", "input_frame")}
            )
            self.add_flow(
                video_decoder_response,
                decoder_output_format_converter,
                {("output_transmitter", "source_video")},
            )
            self.add_flow(decoder_output_format_converter, visualizer, {("tensor", "receivers")})

    def push_data(self, data):
        """Push data into the transmission pipeline.

        Args:
            data: The data to be transmitted or displayed.
        """

        if self._async_data_push is not None:
            self._async_data_push.push_data(data)
        else:
            self._logger.warning("AsyncDataPushOp is not initialized")


import os
from argparse import ArgumentParser
from holoscan.gxf import load_extensions

def callback():
    pass

if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Endoscopy tool tracking demo application.")

    parser.add_argument(
        "-c",
        "--config",
        default="none",
        help=("Set config path to override the default config file location"),
    )
    args = parser.parse_args()

    if args.config == "none":
        config_file = os.path.join(os.path.dirname(__file__), "h264_video_decode.yaml")
    else:
        config_file = args.config

    app = PatientApp(callback)

    context = app.executor.context_uint64
    exts = [
        "libgxf_videodecoder.so",
        "libgxf_videodecoderio.so",
        "libgxf_videoencoder.so",
        "libgxf_videoencoderio.so",
    ]
    # load_extensions(context, exts)

    # create a thread to call app.push_data() with a random image
    import threading
    import time

    def random_image_thread():
        frame_num = 0
        width, height = 1280, 720  # Smaller than current 1920x1080
        while True:
            print("pushing data")
            # Create a random NumPy array
            np_image = np.random.randint(0, 256, size=(height, width, 4), dtype=np.uint8)
            # Convert NumPy array to Warp array (assuming CUDA device)
            image = warp.array(np_image, dtype=warp.types.uint8)
            app.push_data(image)
            frame_num += 1
            time.sleep(1)

    t = threading.Thread(target=random_image_thread)
    t.start()

    app.config(config_file)
    app.run()
    t.join()