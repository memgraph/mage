#include "algorithm.hpp"

#include <unordered_set>
#include "mgp.hpp"

void Collections::Intersection(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto list1{arguments[0].ValueList()};
    std::unordered_set<mgp::Value> set1{list1.begin(), list1.end()};
    const auto list2{arguments[1].ValueList()};
    std::unordered_set<mgp::Value> set2{list2.begin(), list2.end()};

    if (set1.size() > set2.size()) {
      std::swap(set1, set2);
    }

    mgp::List intersection{};
    for (const auto &element : set1) {
      if (set2.contains(element)) {
        intersection.AppendExtend(std::move(element));
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultIntersection).c_str(), intersection);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
