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

cmake_minimum_required(VERSION 3.24)
project(telesurgery)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(telesurgery telesurgery.cpp)
target_link_libraries(telesurgery
  holoscan::core
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  holoscan::ops::video_stream_recorder
  holoscan::ops::v4l2
  holoscan::ops::dds_hid_publisher
  holoscan::ops::dds_hid_subscriber
  holoscan::ops::dds_video_publisher
  holoscan::ops::dds_video_subscriber
  holoscan::ops::dds_camera_info_subscriber
  holoscan::ops::hid_renderer
  dds_hid_command
)

# Copy qos_profiles.xml to the binary directory
add_custom_target(telesurgery_qos_profiles_xml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/qos_profiles.xml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/qos_profiles.xml"
)
add_dependencies(telesurgery telesurgery_qos_profiles_xml)

# copy the config file to the binary directory
add_custom_target(telesurgery_config_file
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/telesurgery.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/telesurgery.yaml"
)
add_dependencies(telesurgery telesurgery_config_file)
