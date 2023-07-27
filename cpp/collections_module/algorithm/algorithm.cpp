#include "algorithm.hpp"

#include <unordered_set>
#include "mgp.hpp"

void Collections::ContainsAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto list1{arguments[0].ValueList()};
    std::unordered_set<mgp::Value> set{list1.begin(), list1.end()};
    const auto list2{arguments[1].ValueList()};
    std::unordered_set<mgp::Value> values{list2.begin(), list2.end()};

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultContainsAll).c_str(), 
      std::all_of(values.begin(), values.end(), [&](const auto &x){return set.contains(x);}));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
