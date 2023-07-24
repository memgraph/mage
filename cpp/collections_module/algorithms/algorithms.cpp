#include <list>

#include "algorithms.hpp"

void Collections::Split(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;

  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &inputList = arguments[0].ValueList();
    const auto &delimiter = arguments[1];

    if (inputList.Empty()) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(Collections::kResultSplit).c_str(), inputList);
      return;
    }

    mgp::List part = mgp::List();
    for (const auto value : inputList) {
      if (value != delimiter) {
        part.AppendExtend(value);
        continue;
      }
      if (part.Empty()) {
        continue;
      }
      auto record = record_factory.NewRecord();
      record.Insert(std::string(Collections::kResultSplit).c_str(), part);
      part = mgp::List();
    }
    if (part.Empty()) {
      return;
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Collections::kResultSplit).c_str(), std::move(part));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
