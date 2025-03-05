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

#include "dds_hid_subscriber.hpp"

#include "dds/topic/find.hpp"

namespace holoscan::ops {

void DDSHIDSubscriberOp::setup(OperatorSpec& spec) {
  DDSOperatorBase::setup(spec);

  // Change the output type to directly use std::vector<InputCommand>
  spec.output<std::vector<InputCommand>>("output");

  spec.param(reader_qos_, "reader_qos", "Reader QoS", "Data Reader QoS Profile", std::string());
  spec.param(hid_device_filters_,
             "hid_device_filters",
             "HID Device Filters",
             "HID Device Filters to capture HID events from",
             std::vector<std::string>());
}

void DDSHIDSubscriberOp::initialize() {
  DDSOperatorBase::initialize();

  // Create the subscriber
  dds::sub::Subscriber subscriber(participant_);

  // Create the InputCommand topic
  auto topic = dds::topic::find<dds::topic::Topic<InputCommand>>(participant_, INPUT_COMMAND_TOPIC);
  if (topic == dds::core::null) {
    topic = dds::topic::Topic<InputCommand>(participant_, INPUT_COMMAND_TOPIC);
  }

  // Join the sanitized device filters with commas for the SQL-like IN clause
  std::string device_filter_string;
  for (const auto& device_filter : hid_device_filters_.get()) {
    if (!device_filter_string.empty()) device_filter_string += ",";
    device_filter_string += device_filter;
  }
  device_filter_string = "'" + device_filter_string + "'";

  HOLOSCAN_LOG_INFO("Device filters: {}", device_filter_string);

  dds::topic::ContentFilteredTopic<InputCommand> filtered_topic(
      topic, "FilteredInputCommand", dds::topic::Filter("device_path MATCH %0", {device_filter_string}));

  // Create the reader for the InputCommand
  reader_ = dds::sub::DataReader<InputCommand>(
      subscriber, filtered_topic, qos_provider_.datareader_qos(reader_qos_.get()));

  // Obtain the reader's status condition
  status_condition_ = dds::core::cond::StatusCondition(reader_);

  // Enable the 'data available' status
  status_condition_.enabled_statuses(dds::core::status::StatusMask::data_available());

  // Attach the status condition to the waitset
  waitset_ += status_condition_;
}

void DDSHIDSubscriberOp::compute(InputContext& op_input, OutputContext& op_output,
                                 ExecutionContext& context) {
  // Configure the wait timeout parameter
  const auto wait_timeout = dds::core::Duration::from_millisecs(100);

  // Wait for new data with timeout
  dds::core::cond::WaitSet::ConditionSeq active_conditions = waitset_.wait(wait_timeout);

  std::vector<InputCommand> valid_commands;
  for (const auto& cond : active_conditions) {
    if (cond == status_condition_) {
      // Take all available commands at once
      dds::sub::LoanedSamples<InputCommand> commands = reader_.take();

      if (commands.length() > 0) {
        // Create a vector to store valid commands
        valid_commands.reserve(commands.length());

        // Filter valid commands
        for (size_t i = 0; i < commands.length(); i++) {
          if (commands[i].info().valid()) { 
            valid_commands.push_back(commands[i].data());
          }
        }
      }
    }
  }

  op_output.emit(valid_commands, "output");
}

}  // namespace holoscan::ops
