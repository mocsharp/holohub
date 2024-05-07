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

# # Proto file
# get_filename_component(hw_proto "../proto/helloworld.proto" ABSOLUTE)
# get_filename_component(hw_proto_path "${hw_proto}" PATH)

# # Generated sources
# set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
# set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
# set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
# set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")
# message(STATUS "before add_custom_command")
# add_custom_command(
#       OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
#       COMMAND ${_PROTOBUF_PROTOC}
#       ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
#         --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
#         -I "${hw_proto_path}"
#         --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
#         "${hw_proto}"
#       DEPENDS "${hw_proto}")

# # Include generated *.pb.h files
# message(STATUS "after add_custom_command")
# include_directories(${CMAKE_CURRENT_BINARY_DIR} ${PROTOBUF_INCLUDE_DIR})
# message(STATUS "after include_directories")

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
  PUBLIC
    grpc_entity_service_protos
  PRIVATE
    absl::flags
    absl::flags_parse
    ${GRPC_GRPCPP}
    ${GRPCPP_REFLECTION}
    ${PROTOBUF_LIBPROTOBUF}
    holoscan::core
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::video_decoder
    holoscan::ops::video_encoder
    holoscan::ops::video_read_bitstream
    holoscan::ops::video_write_bitstream
    lstm_tensor_rt_inference
    tool_tracking_postprocessor)

message(STATUS "===========>PROTOBUF_INCLUDE_DIRS=${PROTOBUF_INCLUDE_DIRS}")
message(STATUS "===========>PROTOBUF_LIBPROTOBUF=${PROTOBUF_LIBPROTOBUF}")
message(STATUS "===========>GRPCPP_REFLECTION=${GRPCPP_REFLECTION}")
message(STATUS "===========>GRPC_GRPCPP=${GRPC_GRPCPP}")
# get_cmake_property(_variableNames VARIABLES)
# list (SORT _variableNames)
# foreach (_variableName ${_variableNames})
#     message(STATUS "===========>${_variableName}=${${_variableName}}")
# endforeach()

# target_link_libraries(grpc_h264_endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::videomaster>)
# target_link_libraries(grpc_h264_endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::qcap_source>)
# target_link_libraries(grpc_h264_endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::vtk_renderer>)

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
