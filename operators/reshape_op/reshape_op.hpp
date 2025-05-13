/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef RESHAPE_OP_HPP
#define RESHAPE_OP_HPP

#include <string>
#include <vector>

#include "holoscan/core/operator.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator to reshape (copy) an input tensor to an output tensor.
 *
 * This operator receives a message containing a tensor, creates a new message with a new tensor,
 * copies the data from the input tensor to the output tensor, and transmits the new message.
 * It mimics the behavior of writing a tensor to a file and reading it back, providing decoupling
 * and potential memory type conversion. This is useful for connecting components that might have
 * incompatible output/input tensor memory types or require separate message entities.
 *
 * == Inputs ==
 *
 * - **input_tensor** : `holoscan::gxf::Entity`
 *   - Expects an entity containing at least one `nvidia::gxf::Tensor`.
 *
 * == Outputs ==
 *
 * - **output_tensor** : `holoscan::gxf::Entity`
 *   - Produces an entity containing a new `nvidia::gxf::Tensor` with the copied data.
 *
 * == Parameters ==
 *
 * - **allocator**: `holoscan::Allocator`
 *   - Allocator used to allocate the buffer for the output tensor.
 * - **out_storage_type**: `int` (Default: 0)
 *   - Desired storage type for the output tensor buffer. 0 for host memory (`kHost`), 1 for
 *     device memory (`kDevice`).
 * - **input_tensor_name**: `std::string` (Optional)
 *   - Name of the tensor component in the input message. If empty, the operator will use the first
 *     tensor found in the input entity.
 * - **output_tensor_name**: `std::string` (Optional)
 *   - Name to assign to the tensor component in the output message. If empty, it uses the name of
 *     the input tensor component.
 */
class ReshapeOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ReshapeOp)

  ReshapeOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<int32_t> out_storage_type_;
  Parameter<std::string> input_tensor_name_;
  Parameter<std::string> output_tensor_name_;

  // Internal buffer for NAL unit processing
  std::vector<uint8_t> buffer_;
  size_t search_offset_ = 0; // Offset in the buffer to start searching for NAL units
  // Parameter to know if AUD NAL units are present, similar to VideoReadBitStream
  Parameter<bool> aud_nal_present_;
};

}  // namespace holoscan::ops

#endif /* RESHAPE_OP_HPP */
