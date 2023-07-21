#include <list>

#include <mgp.hpp>

constexpr std::string_view kResultMax = "max";

void Max(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &list = arguments[0].ValueList();
    mgp::Value max = mgp::Value(list[0]);
    mgp::Type type = max.Type();

    for (const auto value : list) {
      if (max < value) {
        max = value;
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultMax).c_str(), max);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
