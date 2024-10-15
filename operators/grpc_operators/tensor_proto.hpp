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

#ifndef TENSOR_PROTO_HPP
#define TENSOR_PROTO_HPP

#include <memory>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/holoscan.hpp>

#include "generated/holoscan.pb.h"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

class TensorProto {
 public:
  static void tensor_to_entity_request(const nvidia::gxf::Entity& gxf_entity,
                                       std::shared_ptr<EntityRequest> request);
  static void tensor_to_entity_response(const nvidia::gxf::Entity& gxf_entity,
                                        std::shared_ptr<EntityResponse> response);
  static void entity_request_to_tensor(const EntityRequest* entity_request,
                                       nvidia::gxf::Entity& gxf_entity,
                                       nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator);
  static void entity_response_to_tensor(const EntityResponse& entity_request,
                                        nvidia::gxf::Entity& gxf_entity,
                                        nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator);

 private:
  template <typename T>
  static void copy_data_to_proto(const nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor,
                                 ::holoscan::entity::Tensor& tensor_proto);
  template <typename T>
  static void copy_data_to_tensor(const ::holoscan::entity::Tensor& tensor_proto,
                                  nvidia::gxf::Handle<nvidia::gxf::Tensor>& tensor);
  static void gxf_time_to_proto(const nvidia::gxf::Entity& gxf_entity,
                                ::holoscan::entity::Timestamp* timestamp);
  static void gxf_tensor_to_proto(
      const nvidia::gxf::Entity& gxf_entity,
      google::protobuf::Map<std::string, ::holoscan::entity::Tensor>* tensor_map);
  static void proto_to_gxf_time(nvidia::gxf::Entity& gxf_entity,
                                const ::holoscan::entity::Timestamp& timestamp);
  static void proto_to_gxf_tensor(
      nvidia::gxf::Entity& gxf_entity,
      const google::protobuf::Map<std::string, ::holoscan::entity::Tensor>& tensor_map,
      nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator);
};

}  // namespace holoscan::ops

#endif /* TENSOR_PROTO_HPP */
