#include <algorithm>
#include <list>
#include <vector>

#include <mgp.hpp>

constexpr std::string_view kResultSort = "sorted";

void Sort(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &list = arguments[0].ValueList();
    std::vector<mgp::Value> sorted;

    for (const auto value : list) {
      sorted.push_back(value);
    }

    std::sort(sorted.begin(), sorted.end());

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSort).c_str(), mgp::List(sorted));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
