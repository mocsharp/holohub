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

EntityRequest TensorProto::tensor_to_entity_request(const nvidia::gxf::Entity& gxf_entity) {
  EntityRequest request;
  auto timestamp = gxf_entity.get<nvidia::gxf::Timestamp>();
  if (timestamp) {
    request.mutable_timestamp()->set_acqtime((*timestamp)->acqtime);
    request.mutable_timestamp()->set_pubtime((*timestamp)->pubtime);
  }

  auto tensors = gxf_entity.findAll<nvidia::gxf::Tensor, 4>();
  if (!tensors) { throw std::runtime_error("Tensor not found"); }
  for (auto tensor : tensors.value()) {
    holoscan::entity::Tensor& tensor_proto = (*request.mutable_tensors())[tensor->name()];
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
  return request;
}

void TensorProto::tensor_to_entity_response(const nvidia::gxf::Entity& gxf_entity,
                                            EntityResponse* response) {
  auto timestamp = gxf_entity.get<nvidia::gxf::Timestamp>();
  if (timestamp) {
    response->mutable_timestamp()->set_acqtime((*timestamp)->acqtime);
    response->mutable_timestamp()->set_pubtime((*timestamp)->pubtime);
  }

  auto tensors = gxf_entity.findAll<nvidia::gxf::Tensor, 4>();
  if (!tensors) { throw std::runtime_error("Tensor not found"); }
  for (auto tensor : tensors.value()) {
    holoscan::entity::Tensor& tensor_proto = (*response->mutable_tensors())[tensor->name()];
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
nvidia::gxf::Entity TensorProto::entity_request_to_tensor(const EntityRequest* entity_request,
                                                          ExecutionContext& context,
                                                          std::shared_ptr<Allocator> allocator) {
  auto gxf_allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator->gxf_cid());
  auto entity = nvidia::gxf::Entity::New(context.context());
  if (!entity) { throw std::runtime_error("Error creating new entity"); }

  for (auto tensor_entry : entity_request->tensors()) {
    const holoscan::entity::Tensor& tensor_proto = tensor_entry.second;
    auto tensor = entity.value().add<nvidia::gxf::Tensor>(tensor_entry.first.c_str());
    if (!tensor) { throw std::runtime_error("Failed to create tensor"); }
    nvidia::gxf::Shape shape({tensor_proto.dimensions().begin(), tensor_proto.dimensions().end()});
    switch (tensor_proto.primitive_type()) {
      case holoscan::entity::Tensor::kUnsigned8:
        tensor.value()->reshape<uint8_t>(
            shape, nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());
        break;
      case holoscan::entity::Tensor::kUnsigned16:
        tensor.value()->reshape<uint16_t>(
            shape, nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());
        break;
      case holoscan::entity::Tensor::kFloat32:
        tensor.value()->reshape<float>(
            shape, nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());
        break;
      default:
        throw std::runtime_error(
            fmt::format("Unsupported primitive type: {}", tensor_proto.primitive_type()));
    }
    std::copy(tensor_proto.data().begin(), tensor_proto.data().end(), tensor.value()->pointer());
  }

  return entity.value();
}

nvidia::gxf::Entity TensorProto::entity_response_to_tensor(const EntityResponse& entity_response,
                                              ExecutionContext& context,
                                              std::shared_ptr<Allocator> allocator) {
  auto gxf_allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator->gxf_cid());
  auto entity = nvidia::gxf::Entity::New(context.context());
  if (!entity) { throw std::runtime_error("Error creating new entity"); }

  for (auto tensor_entry : entity_response.tensors()) {
    const holoscan::entity::Tensor& tensor_proto = tensor_entry.second;
    auto tensor = entity.value().add<nvidia::gxf::Tensor>(tensor_entry.first.c_str());
    if (!tensor) { throw std::runtime_error("Failed to create tensor"); }
    nvidia::gxf::Shape shape({tensor_proto.dimensions().begin(), tensor_proto.dimensions().end()});
    switch (tensor_proto.primitive_type()) {
      case holoscan::entity::Tensor::kUnsigned8:
        tensor.value()->reshape<uint8_t>(
            shape, nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());
        break;
      case holoscan::entity::Tensor::kUnsigned16:
        tensor.value()->reshape<uint16_t>(
            shape, nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());
        break;
      case holoscan::entity::Tensor::kFloat32:
        tensor.value()->reshape<float>(
            shape, nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());
        break;
      default:
        throw std::runtime_error(
            fmt::format("Unsupported primitive type: {}", tensor_proto.primitive_type()));
    }
    std::copy(tensor_proto.data().begin(), tensor_proto.data().end(), tensor.value()->pointer());
  }

  return entity.value();
}
}  // namespace holoscan::ops