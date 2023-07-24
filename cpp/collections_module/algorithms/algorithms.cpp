#include <list>

#include "algorithms.hpp"

void Collections::ContainsSorted(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    bool contains{false};
    const auto &list = arguments[0].ValueList();
    const auto &element = arguments[1];

    int left{0};
    int right{list.Size() - 1};
    int check;

    while (left <= right) {
      check = (left + right) / 2;
      if (list[check] == element) {
        contains = true;
        break;
      } else if (element < list[check]) {
        right = check - 1;
      } else {
        left = check + 1;
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultCS).c_str(), contains);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
