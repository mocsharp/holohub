/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../../operator_util.hpp"

#include "../reshape_op.hpp"
#include "./reshape_op_pydoc.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for unordered_map -> dict, etc.

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/fragment.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */
class PyReshapeOp : public ReshapeOp {
 public:
  /* Inherit the constructors */
  using ReshapeOp::ReshapeOp;

  // Define a constructor that fully initializes the object.
  PyReshapeOp(Fragment* fragment, const py::args& args,
                   const std::shared_ptr<Allocator>& allocator,
                   int32_t out_storage_type,
                   const std::string& input_tensor_name,
                   const std::string& output_tensor_name,
                   bool aud_nal_present,
                   const std::string& name = "reshape_op")
      : ReshapeOp(ArgList{Arg{"allocator", allocator},
                         Arg{"out_storage_type", out_storage_type}, 
                         Arg{"input_tensor_name", input_tensor_name},
                         Arg{"output_tensor_name", output_tensor_name},
                         Arg{"aud_nal_present", aud_nal_present}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

PYBIND11_MODULE(_reshape_op, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _reshape_op
        .. autosummary::
           :toctree: _generate
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<ReshapeOp, PyReshapeOp, Operator, std::shared_ptr<ReshapeOp>>(
      m, "ReshapeOp", doc::ReshapeOp::doc_ReshapeOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::shared_ptr<Allocator>&,
                    int32_t,
                    const std::string&,
                    const std::string&,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a,
           "out_storage_type"_a,
           "input_tensor_name"_a = std::string(""),
           "output_tensor_name"_a = std::string(""),
           "aud_nal_present"_a = false,
           "name"_a = "reshape_op"s,
           doc::ReshapeOp::doc_ReshapeOp_python)
      .def("setup", &ReshapeOp::setup, "spec"_a, doc::ReshapeOp::doc_setup);
}  // PYBIND11_MODULE

}  // namespace holoscan::ops
