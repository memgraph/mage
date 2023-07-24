#include "algorithms.hpp"

void Collections::RemoveAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto &input_list = arguments[0].ValueList();
    const auto &to_remove_list = arguments[1].ValueList();
    std::vector<mgp::Value> searchable;

    for (const auto value : input_list) {
      searchable.push_back(std::move(value));
    }

    std::vector<mgp::Value>::iterator itr;

    for (const auto key : to_remove_list) {
      while (true) {
        itr = std::find(searchable.begin(), searchable.end(), key);
        if (itr != searchable.cend()) {
          searchable.erase(itr);
        } else {
          break;
        }
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRemoveAll).c_str(), mgp::List(std::move(searchable)));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
