#include <fmt/format.h>
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
        throw mgp::IndexException(
            fmt::format("Pairs must consist of {} elements exactly.", number_of_elements_in_pair));
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

void Map::Merge(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto map1 = arguments[0].ValueMap();
    const auto map2 = arguments[1].ValueMap();

    mgp::Map merged_map = mgp::Map(std::move(map2));

    for (const auto element : map1) {
      if (merged_map.At(element.key).IsNull()) {
        merged_map.Insert(element.key, element.value);
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultMerge).c_str(), std::move(merged_map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
