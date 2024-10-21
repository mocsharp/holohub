# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import base64
import logging
import os
import sys

import requests.exceptions
import torch
from halo import Halo
from holoscan.core import Application, Operator, OperatorSpec
from openai import APIConnectionError, AuthenticationError, OpenAI
from requests.models import PreparedRequest

logging.getLogger("httpx").setLevel(logging.WARN)
logging.getLogger("openai").setLevel(logging.WARN)

logger = logging.getLogger("NVIDIA_NIM_CHAT")
logging.basicConfig(level=logging.INFO)


class OpenAIOperator(Operator):
    def __init__(
        self,
        fragment,
        *args,
        name,
        spinner,
        base_url=None,
        api_key=None,
        model="nvidia/nvclip",
        encoding_format="float",
    ):
        self.spinner = spinner
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.encoding_format = encoding_format

        print(f"======================self.api_key={self.api_key}")

        if not self.api_key:
            self.api_key = os.getenv("API_KEY", None)
        if not self.api_key:
            logger.warning(f"Setting up connection to {base_url} without an API key.")
            logger.warning(
                "Set 'api-key' in the nvidia_nim.yaml config file or set the environment variable 'API_KEY'."
            )
            print("")
        self.client = OpenAI(base_url=base_url, api_key=self.api_key)

        # Need to call the base class constructor last
        super().__init__(fragment, name=name)

    def setup(self, spec: OperatorSpec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        input = op_input.receive("in")

        message = input[0].copy()
        message.append(input[1])

        try:
            embeddings_data = self.client.embeddings.create(
                input=message, model=self.model, encoding_format=self.encoding_format
            )

            all_embeddings = [data.embedding for data in embeddings_data.data]
            image_embeddings = [torch.tensor(embedding) for embedding in all_embeddings[:-1]]

            image_embeddings = torch.stack(image_embeddings)
            text_embeddings = torch.tensor(all_embeddings[-1])

            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            probabilities = (100.0 * text_embeddings @ image_embeddings.T).softmax(dim=-1)

            probabilities = {
                f"Image {i+1}": round(float(d), 2) * 100 for i, d in enumerate(probabilities)
            }

            print(f"\nPrompt: {input[1]}")
            print("Output:")
            for prob in probabilities:
                print(f"{prob}: {probabilities[prob]}%")

        except (AuthenticationError, APIConnectionError) as e:
            if not self.api_key or "401" in e.args[0]:
                logger.error(
                    "%s: Hey there! It looks like you might have forgotten to set your API key or the key is invalid. No worries, it happens to the best of us! 😊",
                    e,
                )
            else:
                logger.error(str(e))
        except Exception as e:
            print(type(e))
            logger.error("Oops! Something went wrong: %s", repr(e))
        print(" ")
        self.spinner.stop()


class ExamplesOp(Operator):
    def __init__(self, fragment, *args, spinner, example, **kwargs):
        self.spinner = spinner
        self.example = example

        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        user_images = []
        user_text = ""
        use_example = False
        while True:
            print("\nEnter prompt or hit Enter to use default example: ", end="")
            user_input = sys.stdin.readline().strip()
            if user_input != "":
                user_text = user_input
            else:
                use_example = True
                user_images = self.example["images"]
                user_text = self.example["text"]
            break

        while True and not use_example:
            print("\nEnter URL to an image or hit Enter to send the request: ", end="")
            user_input = sys.stdin.readline().strip()
            if user_input == "":
                break
            user_images.append(self.pre_process_input(user_input))

        self.spinner.start()
        op_output.emit((user_images, user_text), "out")

    def pre_process_input(self, data):
        prepared_request = PreparedRequest()
        try:
            prepared_request.prepare_url(data, None)
            print("Downloading image...")
            return base64.b64encode(requests.get(prepared_request.url).content).decode("utf-8")
        except Exception:
            return data


class NVClipNIMApp(Application):
    def __init__(self):
        super().__init__()

    def compose(self):
        spinner = Halo(
            text="Generating...",
            spinner="dots",
        )

        input_op = ExamplesOp(
            self,
            name="input",
            spinner=spinner,
            example=self.kwargs("example"),
        )
        chat_op = OpenAIOperator(
            self,
            name="chat",
            spinner=spinner,
            **self.kwargs("nim"),
        )

        self.add_flow(input_op, chat_op)


if __name__ == "__main__":
    config_file = os.path.join(os.path.dirname(__file__), "nvidia_nim.yaml")
    app = NVClipNIMApp()
    app.config(config_file)

    try:
        app.run()
    except Exception as e:
        logger.error("Error:", str(e))
