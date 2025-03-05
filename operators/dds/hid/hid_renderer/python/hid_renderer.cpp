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

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include "../hid_renderer.hpp"
#include "./hid_renderer_pydoc.hpp"

#include "../../../../operator_util.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/operator_spec.hpp"

using std::string_literals::operator""s;
using pybind11::literals::operator""_a;

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan::ops {

/* Trampoline class for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the operator.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the operator's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_operator<OperatorT>
 */

class PyHIDRendererOp : public HIDRendererOp {
 public:
  /* Inherit the constructors */
  using HIDRendererOp::HIDRendererOp;

  // Define a constructor that fully initializes the object.
  PyHIDRendererOp(Fragment* fragment, const py::args& args,
                         const std::string& allocator = "",
                         const std::vector<ops::HolovizOp::InputSpec>& tensors = {},
                         uint32_t width = 1024,
                         uint32_t height = 576,
                         const std::string& name = "hid_renderer")
      : HIDRendererOp(ArgList{
                                     Arg{"allocator", allocator},
                                     Arg{"tensors", tensors},
                                     Arg{"width", width},
                                     Arg{"height", height}}) {
    add_positional_condition_and_resource_args(this, args);
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<OperatorSpec>(fragment);
    setup(*spec_.get());
  }
};

/* The python module */

PYBIND11_MODULE(_hid_renderer, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Python Bindings
        ---------------------------------------
        .. currentmodule:: _hid_renderer
        .. autosummary::
           :toctree: _generate
           add
           subtract
    )pbdoc";

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif

  py::class_<HIDRendererOp,
             PyHIDRendererOp,
             Operator,
             std::shared_ptr<HIDRendererOp>>(
      m, "HIDRendererOp", doc::HIDRendererOp::doc_HIDRendererOp)
      .def(py::init<Fragment*,
                    const py::args&,
                    const std::string&,
                    const std::vector<ops::HolovizOp::InputSpec>&,
                    uint32_t,
                    uint32_t,
                    const std::string&>(),
           "fragment"_a,
           "allocator"_a = ""s,
           "tensors"_a = std::vector<ops::HolovizOp::InputSpec>{},
           "width"_a = 1024,
           "height"_a = 576,
           "name"_a = "hid_renderer"s,
           doc::HIDRendererOp::doc_HIDRendererOp)
      .def("initialize", &HIDRendererOp::initialize,
           doc::HIDRendererOp::doc_initialize)
      .def("setup", &HIDRendererOp::setup, "spec"_a,
           doc::HIDRendererOp::doc_setup);
}  // PYBIND11_MODULE NOLINT
}  // namespace holoscan::ops
