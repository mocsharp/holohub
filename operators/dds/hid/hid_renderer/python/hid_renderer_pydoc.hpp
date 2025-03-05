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

#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc::HIDRendererOp {

PYDOC(HIDRendererOp, R"doc(
HID Renderer operator.
)doc")

// PyDDSHIDRendererOp Constructor
PYDOC(HIDRendererOp_python, R"doc(
HID Renderer operator.

Parameters
----------
allocator : holoscan.core.Allocator, optional
    The allocator to use for the operator.
tensors : list, optional
    The Holoviz input tensors to use for the operator.
width : int, optional
    The width of the Holoviz window.
height : int, optional
    The height of the Holoviz window.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::HIDRendererOp
