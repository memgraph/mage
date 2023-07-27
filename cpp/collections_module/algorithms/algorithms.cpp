#include "algorithms.hpp"

void Collections::RemoveAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto input_list = arguments[0].ValueList();
    const auto to_remove_list = arguments[1].ValueList();

    std::multiset<mgp::Value> searchable;

    for (const auto value : input_list) {
      searchable.insert(std::move(value));
    }

    for (const auto key : to_remove_list) {
      while (true) {
        auto itr = searchable.find(key);
        if (itr == searchable.end()) {
          break;
        }
        searchable.erase(itr);
      }
    }

    mgp::List final_list = mgp::List();
    for (const auto element : searchable) {
      final_list.AppendExtend(std::move(element));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRemoveAll).c_str(), final_list);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
