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
  static EntityRequest tensor_to_entity_request(const nvidia::gxf::Entity& gxf_entity);
  static void tensor_to_entity_response(const nvidia::gxf::Entity& gxf_entity,
                                        EntityResponse* response);
  static nvidia::gxf::Entity entity_request_to_tensor(const EntityRequest* entity_request,
                                                      ExecutionContext& context,
                                                      std::shared_ptr<Allocator> allocator);
  static nvidia::gxf::Entity entity_response_to_tensor(const EntityResponse& entity_request,
                                                       ExecutionContext& context,
                                                       std::shared_ptr<Allocator> allocator);
};

}  // namespace holoscan::ops

#endif /* TENSOR_PROTO_HPP */
