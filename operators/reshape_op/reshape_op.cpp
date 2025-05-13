#include "reshape_op.hpp"

#include <cuda_runtime.h>
#include <chrono>
#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"  // To get nvidia::gxf::Allocator
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

#include "gxf/core/entity.hpp"    // nvidia::gxf::Entity::Shared
#include "gxf/std/allocator.hpp"  // nvidia::gxf::Allocator, nvidia::gxf::MemoryStorageType
#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.
#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp

namespace holoscan::ops {

#define MIN_CHUNK_SIZE 5
#define NAL_UNIT_START_CODE 0x00000001
#define AUD_NAL_UNIT_START_CODE 0x00000109

namespace gxf = nvidia::gxf;  // Define alias for convenience

void ReshapeOp::setup(OperatorSpec& spec) {
  // Need to save the spec ports to member variables if used outside setup
  // For now, just defining them is okay if only used here.
  spec.input<holoscan::gxf::Entity>("in");
  spec.output<holoscan::gxf::Entity>("out");

  spec.param(allocator_, "allocator", "Allocator", "Allocator for the output tensor buffer");
  spec.param(out_storage_type_,
             "out_storage_type",
             "Output Storage Type",
             "Memory storage type for the output tensor buffer. 0:kHost, 1:kDevice",
             0);
  spec.param(input_tensor_name_,
             "input_tensor_name",
             "Input Tensor Name",
             "Optional name of the tensor component in the input message. "
             "If empty, finds the first available tensor.",
             std::string(""));
  spec.param(output_tensor_name_,
             "output_tensor_name",
             "Output Tensor Name",
             "Optional name of the tensor component to add to the output message. "
             "If empty, uses the same name as the input tensor.",
             std::string("h264_video"));
  spec.param(aud_nal_present_,
             "aud_nal_present",
             "AUD NAL Present",
             "Whether AUD NAL units are present in the bitstream.",
             false);
}

void ReshapeOp::compute(InputContext& op_input, OutputContext& op_output,
                        ExecutionContext& context) {
  const uint64_t acqtime = std::chrono::system_clock::now().time_since_epoch().count();
  // Try to receive an input entity
  auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("in");

  if (!maybe_entity) {
    throw std::runtime_error("Failed to receive input entity");
  }
  auto tensor = maybe_entity.value().get<Tensor>(input_tensor_name_.get().c_str());

  if (!tensor) {
    throw std::runtime_error("Failed to get tensor with name: " + input_tensor_name_.get());
  }

  if (tensor && tensor->size() > 0) {
    // Ensure input is uint8_t for raw byte access
    DLDataType dtype = tensor->dtype();
    if (dtype.code != kDLUInt || dtype.bits != 8 || dtype.lanes != 1) {
      HOLOSCAN_LOG_WARN(
          "ReshapeOp received tensor with non-uint8_t element type. Attempting to copy raw "
          "bytes.");
    }
    const size_t prev_buffer_size = buffer_.size();
    buffer_.resize(prev_buffer_size + tensor->size());

    // Determine memcpy kind
    cudaMemcpyKind copy_kind_to_buffer;
    if (tensor->device().device_type == kDLCUDA) {
      copy_kind_to_buffer = cudaMemcpyDeviceToHost;
    } else {
      copy_kind_to_buffer = cudaMemcpyHostToHost;
    }

    cudaError_t cuda_result = cudaMemcpy(
        buffer_.data() + prev_buffer_size, tensor->data(), tensor->size(), copy_kind_to_buffer);
    if (cuda_result != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("Failed to cudaMemcpy input tensor to internal buffer: {}",
                         cudaGetErrorString(cuda_result));
      buffer_.resize(prev_buffer_size);  // Revert resize
      // Decide if we should drop this tensor or stop
      return;
    }
    HOLOSCAN_LOG_INFO("Appended {} bytes to internal buffer. Total buffer size: {}",
                      tensor->size(),
                      buffer_.size());
  }

  if (buffer_.empty() || search_offset_ >= buffer_.size()) {
    if (buffer_.empty()) {
      HOLOSCAN_LOG_INFO("ReshapeOp: No input and buffer empty. Waiting.");
    }
    return;
  }

  // NAL Unit Search Logic (simplified from videodecoder_input.cpp)
  const uint32_t target_start_code =
      aud_nal_present_.get() ? AUD_NAL_UNIT_START_CODE : NAL_UNIT_START_CODE;

  size_t nal_start_pos = search_offset_;
  size_t nal_end_pos = search_offset_;
  bool first_nal_found = false;

  // Find the start of the first NAL unit
  for (size_t i = search_offset_; i + 3 < buffer_.size(); ++i) {
    uint32_t val =
        (buffer_[i] << 24) | (buffer_[i + 1] << 16) | (buffer_[i + 2] << 8) | buffer_[i + 3];
    if (val == target_start_code) {
      nal_start_pos = i;
      first_nal_found = true;
      break;
    }
  }

  if (!first_nal_found) {
    // No start code found in the remaining buffer.
    // If we received new data this tick, we might need more data to form a NAL unit.
    // If no new data came, and still no NAL, maybe clear buffer or wait.
    // For now, assume data might be incomplete, so keep it and wait.
    HOLOSCAN_LOG_INFO("No NAL start found after offset {}. Buffer size: {}. Waiting for more data.",
                      search_offset_,
                      buffer_.size());
    return;
  }

  // Find the start of the NEXT NAL unit (or end of buffer)
  nal_end_pos = nal_start_pos + 4;  // Start searching after the current start code
  for (size_t i = nal_start_pos + 4; i + 3 < buffer_.size(); ++i) {
    uint32_t val =
        (buffer_[i] << 24) | (buffer_[i + 1] << 16) | (buffer_[i + 2] << 8) | buffer_[i + 3];
    if (val == target_start_code) {
      nal_end_pos = i;
      break;
    }
    nal_end_pos = buffer_.size();  // If no next NAL, consume till end
  }
  if (nal_start_pos + 4 > buffer_.size()) {  // Not enough data even for one start code
    nal_end_pos = buffer_.size();
  }

  size_t chunk_size = nal_end_pos - nal_start_pos;

  if (chunk_size == 0) {
    // This might happen if we only have a start code at the very end
    // or some other edge case. Advance search_offset_ to avoid infinite loop.
    HOLOSCAN_LOG_INFO("NAL parsing resulted in zero chunk size. Advancing search offset.");
    search_offset_ = nal_end_pos > search_offset_ ? nal_end_pos : search_offset_ + 1;
    if (search_offset_ >= buffer_.size() && !buffer_.empty()) {
      HOLOSCAN_LOG_INFO("Processed entire buffer, clearing.");
      buffer_.clear();
      search_offset_ = 0;
    }
    return;  // Try again next tick
  }

  HOLOSCAN_LOG_INFO("NAL Chunk: start={}, end={}, size={}", nal_start_pos, nal_end_pos, chunk_size);

  // --- Create and send output tensor ---
  gxf_context_t gxf_context = context.context();
  auto allocator_handle_expected =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(gxf_context, allocator_.get()->gxf_cid());
  if (!allocator_handle_expected) {
    HOLOSCAN_LOG_ERROR("Failed to create allocator handle. Code: {}",
                       GxfResultStr(allocator_handle_expected.error()));
    throw std::runtime_error("Failed to create allocator handle");
  }
  auto allocator_handle = allocator_handle_expected.value();

  auto holoscan_output_entity_wrapper = holoscan::gxf::Entity::New(&context);
  auto output_gxf_entity =
      nvidia::gxf::Entity::Shared(gxf_context, holoscan_output_entity_wrapper.eid());
  if (!output_gxf_entity) {
    HOLOSCAN_LOG_ERROR("Failed to create output GXF entity. Code: {}",
                       GxfResultStr(output_gxf_entity.error()));
    return;
  }

  const char* out_name =
      output_tensor_name_.get().empty() ? nullptr : output_tensor_name_.get().c_str();
  auto maybe_output_tensor_handle = output_gxf_entity->add<nvidia::gxf::Tensor>(out_name);
  if (!maybe_output_tensor_handle) {
    HOLOSCAN_LOG_ERROR("Failed to add tensor component \'{}\' to output message. Code: {}",
                       out_name ? out_name : "<unnamed>",
                       GxfResultStr(maybe_output_tensor_handle.error()));
    return;
  }
  auto output_tensor_handle = maybe_output_tensor_handle.value();
  auto output_tensor = output_tensor_handle.get();
  if (!output_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to get tensor data from output handle \'{}\'",
                       output_tensor_handle.name());
    return;
  }

  nvidia::gxf::MemoryStorageType output_storage_type_val;
  if (out_storage_type_.get() == 0) {
    output_storage_type_val = nvidia::gxf::MemoryStorageType::kHost;
  } else {
    output_storage_type_val = nvidia::gxf::MemoryStorageType::kDevice;
  }

  nvidia::gxf::Shape output_shape{static_cast<int32_t>(chunk_size)};
  auto reshape_result =
      output_tensor->reshape<uint8_t>(output_shape, output_storage_type_val, allocator_handle);

  if (!reshape_result) {
    HOLOSCAN_LOG_ERROR("Failed to reshape output tensor. Code: {}",
                       GxfResultStr(reshape_result.error()));
    return;
  }

  cudaMemcpyKind copy_to_output_kind;
  if (output_storage_type_val == nvidia::gxf::MemoryStorageType::kDevice) {
    copy_to_output_kind = cudaMemcpyHostToDevice;
  } else {
    copy_to_output_kind = cudaMemcpyHostToHost;
  }

  cudaError_t cuda_copy_result =
      cudaMemcpy(output_tensor->pointer(),
                 buffer_.data() + nal_start_pos,  // Copy from the start of the NAL unit
                 chunk_size,
                 copy_to_output_kind);  // Corrected cudaMemcpyKind

  if (cuda_copy_result != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("cudaMemcpy to output_tensor failed: {}",
                       cudaGetErrorString(cuda_copy_result));
    return;
  }

  // Update search offset for the next tick
  search_offset_ = nal_end_pos;

  // If we've processed the whole buffer, clear it.
  if (search_offset_ >= buffer_.size()) {
    HOLOSCAN_LOG_INFO("Processed entire buffer, clearing.");
    buffer_.clear();
    search_offset_ = 0;
  } else {
    HOLOSCAN_LOG_INFO("Remaining data in buffer: {}, next search_offset_: {}",
                      buffer_.size() - search_offset_,
                      search_offset_);
    // We have more data, should we try to process more NAL units in this same tick?
    // For simplicity, let's process one NAL chunk per tick for now.
    // The operator will be scheduled again.
  }

  const uint64_t pubtime = std::chrono::system_clock::now().time_since_epoch().count();
  auto timestamp = output_gxf_entity->add<nvidia::gxf::Timestamp>("timestamp");
  if (!timestamp) {
    HOLOSCAN_LOG_ERROR("Failed to add timestamp component to output message. Code: {}",
                       GxfResultStr(timestamp.error()));
    return;
  }
  timestamp.value()->acqtime = acqtime;
  timestamp.value()->pubtime = pubtime;

  op_output.emit(holoscan_output_entity_wrapper, "out");  // User changed to "out"
  HOLOSCAN_LOG_INFO("ReshapeOp emitted NAL unit of size {}", chunk_size);
}

}  // namespace holoscan::ops