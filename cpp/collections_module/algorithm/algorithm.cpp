#include "algorithm.hpp"
#include <unordered_set>

void Collections::toSet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List list = arguments[0].ValueList();
    std::unordered_set<mgp::Value> set(list.begin(), list.end());
    mgp::List return_list;
    for (auto elem : set) {
      return_list.AppendExtend(std::move(elem));
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnToSet).c_str(), std::move(return_list));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
