#include <list>
#include <sstream>

#include "map.hpp"

const auto number_of_elements_in_pair = 2;

void Map::FromPairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto list = arguments[0].ValueList();

    mgp::Map pairs_map;

    for (const auto inside_list : list) {
      if (inside_list.ValueList().Size() != number_of_elements_in_pair) {
        throw mgp::IndexException("Number of elements in a pair is not right.");
      }
      if (!inside_list.ValueList()[0].IsString()) {
        throw mgp::ValueException("All keys have to be type string.");
      }
      pairs_map.Update(inside_list.ValueList()[0].ValueString(), std::move(inside_list.ValueList()[1]));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultFromPairs).c_str(), std::move(pairs_map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
