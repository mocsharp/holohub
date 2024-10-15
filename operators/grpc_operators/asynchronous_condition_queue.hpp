#ifndef ASYNCHRONOUS_CONDITION_QUEUE_HPP
#define ASYNCHRONOUS_CONDITION_QUEUE_HPP

#include <queue>

#include <holoscan.pb.h>

namespace holoscan::ops {

template <typename DataT>
class AsynchronousConditionQueue : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(AsynchronousConditionQueue, Resource)

  explicit AsynchronousConditionQueue(
      std::shared_ptr<AsynchronousCondition> request_available_condition)
      : queue_(), data_available_condition_(request_available_condition) {}

  void push(DataT entity) {
    queue_.push(entity);
    if (data_available_condition_->event_state() == AsynchronousEventState::EVENT_WAITING) {
      data_available_condition_->event_state(AsynchronousEventState::EVENT_DONE);
    }
  }

  DataT pop() {
    if (empty()) { return nullptr; }
    auto item = queue_.front();
    queue_.pop();
    return item;
  }

  bool empty() { return queue_.empty(); }

 private:
  std::shared_ptr<AsynchronousCondition> data_available_condition_;
  std::queue<DataT> queue_;
};

}  // namespace holoscan::ops

#endif /* ASYNCHRONOUS_CONDITION_QUEUE_HPP */
