#ifndef CONDITIONAL_VARIABLE_QUEUE_HPP
#define CONDITIONAL_VARIABLE_QUEUE_HPP

#include <queue>

#include <holoscan.pb.h>

namespace holoscan::ops {

template <typename DataT>
class ConditionVariableQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(ConditionVariableQueue, Resource)

  ConditionVariableQueue() : queue_() {}

  void push(DataT value) {
    std::lock_guard<std::mutex> lock(response_available_mutex_);
    queue_.push(value);
    data_available_condition_.notify_one();
  }

  DataT pop() {
    std::unique_lock<std::mutex> lock(response_available_mutex_);
    data_available_condition_.wait(lock, [this]() { return !queue_.empty(); });
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  bool empty() {
    std::lock_guard<std::mutex> lock(response_available_mutex_);
    return queue_.empty();
  }

 private:
  std::queue<DataT> queue_;
  std::condition_variable data_available_condition_;
  std::mutex response_available_mutex_;
};

}  // namespace holoscan::ops

#endif /* CONDITIONAL_VARIABLE_QUEUE_HPP */
