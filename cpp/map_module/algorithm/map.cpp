#include <list>
#include <sstream>

#include "map.hpp"

void Map::FromPairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &list = arguments[0].ValueList();

    mgp::Map pairsMap;

    for (const auto inside_list : list) {
      if (inside_list.ValueList().Size() != 2) {
        throw mgp::ValueException("Pairs must consist of 2 elements exactly.");
      }
      if (!inside_list.ValueList()[0].IsString()) {
        throw mgp::ValueException("All keys have to be type string.");
      }
      pairsMap.Insert(inside_list.ValueList()[0].ValueString(), std::move(inside_list.ValueList()[1]));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFromPairs).c_str(), std::move(pairsMap));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
