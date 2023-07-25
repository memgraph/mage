#include "map.hpp"

void Map::Merge(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &map1 = arguments[0].ValueMap();
    const auto &map2 = arguments[1].ValueMap();

    mgp::Map merged = mgp::Map(std::move(map2));

    for (const auto element : map1) {
      if (merged.At(element.key).IsNull()) {
        merged.Insert(element.key, element.value);
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultMerge).c_str(), std::move(merged));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
