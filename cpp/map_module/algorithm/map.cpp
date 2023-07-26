#include "map.hpp"

void Map::SetKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto map = arguments[0].ValueMap();
    const auto &key{arguments[1].ValueString()};
    const auto &value = arguments[2];
    map.Update(key, std::move(value));

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSetKey).c_str(), std::move(map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
