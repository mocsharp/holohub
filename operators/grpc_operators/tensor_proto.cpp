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

#include <fmt/format.h>

#include <gxf/std/tensor.hpp>

#include "tensor_proto.hpp"

namespace holoscan::ops {

void TensorProto::gxf_time_to_proto(const nvidia::gxf::Entity& gxf_entity,
                                    ::holoscan::entity::Timestamp* timestamp) {
  auto gxf_timestamp = gxf_entity.get<nvidia::gxf::Timestamp>();
  if (gxf_timestamp) {
    timestamp->set_acqtime((*gxf_timestamp)->acqtime);
    timestamp->set_pubtime((*gxf_timestamp)->pubtime);
  }
}

void TensorProto::proto_to_gxf_time(nvidia::gxf::Entity& gxf_entity,
                                    const ::holoscan::entity::Timestamp& timestamp) {
  auto gxf_timestamp = gxf_entity.add<nvidia::gxf::Timestamp>("timestamp");
  (*gxf_timestamp)->acqtime = timestamp.acqtime();
  (*gxf_timestamp)->pubtime = timestamp.pubtime();
}

void TensorProto::gxf_tensor_to_proto(
    const nvidia::gxf::Entity& gxf_entity,
    google::protobuf::Map<std::string, ::holoscan::entity::Tensor>* tensor_map) {
  auto tensors = gxf_entity.findAll<nvidia::gxf::Tensor, 4>();
  if (!tensors) { throw std::runtime_error("Tensor not found"); }

  for (auto tensor : tensors.value()) {
    HOLOSCAN_LOG_INFO("Tensor name: {}", tensor->name());
    holoscan::entity::Tensor& tensor_proto = (*tensor_map)[tensor->name()];
    for (uint32_t i = 0; i < (*tensor)->shape().rank(); i++) {
      tensor_proto.add_dimensions((*tensor)->shape().dimension(i));
    }
    switch ((*tensor)->element_type()) {
      case nvidia::gxf::PrimitiveType::kUnsigned8:
        tensor_proto.set_primitive_type(holoscan::entity::Tensor::kUnsigned8);
        break;
      case nvidia::gxf::PrimitiveType::kUnsigned16:
        tensor_proto.set_primitive_type(holoscan::entity::Tensor::kUnsigned16);
        break;
      case nvidia::gxf::PrimitiveType::kFloat32:
        tensor_proto.set_primitive_type(holoscan::entity::Tensor::kFloat32);
        break;
      default:
        throw std::runtime_error(fmt::format("Unsupported primitive type: {}",
                                             static_cast<int>((*tensor)->element_type())));
    }
    tensor_proto.set_data((*tensor)->pointer(), (*tensor)->size());
  }
}

void TensorProto::proto_to_gxf_tensor(
    nvidia::gxf::Entity& gxf_entity,
    const google::protobuf::Map<std::string, ::holoscan::entity::Tensor>& tensor_map,
    nvidia::gxf::Handle<nvidia::gxf::Allocator>& allocator) {
  for (auto tensor_entry : tensor_map) {
    const holoscan::entity::Tensor& tensor_proto = tensor_entry.second;
    auto tensor = gxf_entity.add<nvidia::gxf::Tensor>(tensor_entry.first.c_str());
    if (!tensor) { throw std::runtime_error("Failed to create tensor"); }

    nvidia::gxf::Shape shape({tensor_proto.dimensions().begin(), tensor_proto.dimensions().end()});
    switch (tensor_proto.primitive_type()) {
      case holoscan::entity::Tensor::kUnsigned8:
        tensor.value()->reshape<uint8_t>(shape, nvidia::gxf::MemoryStorageType::kHost, allocator);
        break;
      case holoscan::entity::Tensor::kUnsigned16:
        tensor.value()->reshape<uint16_t>(shape, nvidia::gxf::MemoryStorageType::kHost, allocator);
        break;
      case holoscan::entity::Tensor::kFloat32:
        tensor.value()->reshape<float>(shape, nvidia::gxf::MemoryStorageType::kHost, allocator);
        break;
      default:
        throw std::runtime_error(
            fmt::format("Unsupported primitive type: {}", tensor_proto.primitive_type()));
    }
    std::copy(tensor_proto.data().begin(), tensor_proto.data().end(), tensor.value()->pointer());
  }
}

void TensorProto::tensor_to_entity_request(const nvidia::gxf::Entity& gxf_entity,
                                           std::shared_ptr<EntityRequest> request) {
  TensorProto::gxf_time_to_proto(gxf_entity, request->mutable_timestamp());
  TensorProto::gxf_tensor_to_proto(gxf_entity, request->mutable_tensors());
}

void TensorProto::tensor_to_entity_response(const nvidia::gxf::Entity& gxf_entity,
                                            std::shared_ptr<EntityResponse> response) {
  TensorProto::gxf_time_to_proto(gxf_entity, response->mutable_timestamp());
  TensorProto::gxf_tensor_to_proto(gxf_entity, response->mutable_tensors());
}

void TensorProto::entity_request_to_tensor(
    const EntityRequest* entity_request, nvidia::gxf::Entity& gxf_entity,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator) {
  TensorProto::proto_to_gxf_time(gxf_entity, entity_request->timestamp());
  TensorProto::proto_to_gxf_tensor(gxf_entity, entity_request->tensors(), gxf_allocator);
}

void TensorProto::entity_response_to_tensor(
    const EntityResponse& entity_response, nvidia::gxf::Entity& gxf_entity,
    nvidia::gxf::Handle<nvidia::gxf::Allocator> gxf_allocator) {
  TensorProto::proto_to_gxf_time(gxf_entity, entity_response.timestamp());
  TensorProto::proto_to_gxf_tensor(gxf_entity, entity_response.tensors(), gxf_allocator);
}
}  // namespace holoscan::ops