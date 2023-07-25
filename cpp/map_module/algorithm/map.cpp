#include <list>
#include <sstream>

#include "map.hpp"

void Map::FromPairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &list = arguments[0].ValueList();

    std::map<std::string_view, mgp::Value> pairsMap;
    mgp::Value key;

    for (const auto inside_list : list) {
      key = inside_list.ValueList()[0] if (key.IsNumeric)
                pairsMap.insert({inside_list.ValueList()[0], inside_list.ValueList()[1]});
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFromPairs).c_str(), mgp::Map(std::move(pairsMap)));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
