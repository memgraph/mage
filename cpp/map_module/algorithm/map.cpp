#include <fmt/format.h>
#include <list>
#include <sstream>

#include "map.hpp"

const auto number_of_elements_in_pair = 2;

void Map::RemoveRecursion(mgp::Map &result, bool recursive, std::string_view key) {
  for (auto element : result) {
    if (element.key == key) {
      result.Erase(element.key);
      continue;
    }
    if (element.value.IsMap() && recursive) {
      // TO-DO no need for non_const_value_map in new version of memgraph
      mgp::Map non_const_value_map = mgp::Map(std::move(element.value.ValueMap()));
      RemoveRecursion(non_const_value_map, recursive, key);
      if (non_const_value_map.Empty()) {
        result.Erase(element.key);
        continue;
      }
      result.Update(element.key, mgp::Value(std::move(non_const_value_map)));
    }
  }
}

void Map::RemoveKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto map = arguments[0].ValueMap();
    const auto key = arguments[1].ValueString();
    const auto recursive = arguments[2].ValueBool();

    mgp::Map map_removed = mgp::Map(std::move(map));

    RemoveRecursion(map_removed, recursive, key);

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRemoveKey).c_str(), std::move(map_removed));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

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
