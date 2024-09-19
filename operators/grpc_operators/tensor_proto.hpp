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
