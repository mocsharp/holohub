cmake_minimum_required(VERSION 3.20)
project(reshape_op
  LANGUAGES CXX
)

find_package(holoscan 3.0 REQUIRED CONFIG)

add_library(reshape_op SHARED
  reshape_op.cpp
)

target_link_libraries(reshape_op
  GXF::std
  holoscan::core
  )
add_library(holoscan::ops::reshape_op ALIAS reshape_op)

target_include_directories(reshape_op INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_subdirectory(python)