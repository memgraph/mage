#include "map.hpp"


void Map::RemoveKeys(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Map map = arguments[0].ValueMap();
    const mgp::List &list = arguments[1].ValueList();
    for(auto &elem: list){
        //map.Erase(elem.ValueString());
    }
    auto record = record_factory.NewRecord();
    record.Insert("result", std::move(map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}