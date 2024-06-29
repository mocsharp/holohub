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

cmake_minimum_required(VERSION 3.20)
project(grpc_h264_endoscopy_tool_tracking CXX)

find_package(PkgConfig REQUIRED)
find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

include(../protos/common.cmake)

set(CORE_PROTO_FILES
    ${CMAKE_CURRENT_LIST_DIR}/../protos/helloworld.proto
)
grpc_generate_cpp(CORE_GRPC_SRCS CORE_GRPC_HDRS INCLUDE_DIRS ${CORE_PROTO_FILES})

add_library(grpc_entity_service_protos ${CORE_GRPC_HDRS} ${CORE_GRPC_SRCS})

target_link_libraries(grpc_entity_service_protos
  ${GRPC_GRPCPP}
  ${GRPCPP_REFLECTION}
  ${PROTOBUF_LIBPROTOBUF})

add_executable(grpc_h264_endoscopy_tool_tracking main.cpp)
target_include_directories(grpc_h264_endoscopy_tool_tracking PUBLIC ${INCLUDE_DIRS} ${PROTOBUF_INCLUDE_DIRS})


target_link_libraries(grpc_h264_endoscopy_tool_tracking
    grpc_entity_service_protos
    absl::flags
    absl::flags_parse
    ${GRPC_GRPCPP}
    ${GRPCPP_REFLECTION}
    ${PROTOBUF_LIBPROTOBUF}
    holoscan::core
    holoscan::ops::gxf_codelet
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::video_encoder
    lstm_tensor_rt_inference
    tool_tracking_postprocessor)


# Download the associated dataset if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(grpc_h264_endoscopy_tool_tracking endoscopy_data)
endif()

# Copy the config to the binary directory
add_custom_target(grpc_h264_endoscopy_tool_tracking_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "endoscopy_tool_tracking.yaml"
  BYPRODUCTS "endoscopy_tool_tracking.yaml"
)
add_dependencies(grpc_h264_endoscopy_tool_tracking grpc_h264_endoscopy_tool_tracking_deps)
