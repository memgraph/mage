#include <list>

#include <mgp.hpp>

constexpr std::string_view kResultSplit = "splitted";

void Split(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;

  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &inputList = arguments[0].ValueList();
    const auto &delimiter = arguments[1];

    mgp::List part = mgp::List();
    for (const auto value : inputList) {
      if (value == delimiter) {
        if (!part.Empty()) {
          auto record = record_factory.NewRecord();
          record.Insert(std::string(kResultSplit).c_str(), part);
          part = mgp::List();
        }
      } else {
        part.AppendExtend(value);
      }
    }
    if (!part.Empty()) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kResultSplit).c_str(), std::move(part));
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
