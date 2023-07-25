#include "algorithm.hpp"

#include <unordered_set>
#include "mgp.hpp"

void Collections::ContainsAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto &list1{arguments[0].ValueList()};
    std::unordered_set<mgp::Value> set{list1.begin(), list1.end()};
    const auto &list2{arguments[1].ValueList()};
    std::unordered_set<mgp::Value> values{list2.begin(), list2.end()};

    bool contained{true};
    for (const auto &value : values)
      if (set.find(value) == set.end()) {
        contained = false;
        break;
      }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultContainsAll).c_str(), contained);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
