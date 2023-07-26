#include "map.hpp"

void Map::removeRecursion(mgp::Map &result, bool recursive, std::string_view key) {
  for (auto element : result) {
    if (element.key == key) {
      result.Erase(element.key);
      continue;
    }
    if (element.value.IsMap() && recursive) {
      mgp::Map non_const_value_map = mgp::Map(std::move(element.value.ValueMap()));
      removeRecursion(non_const_value_map, recursive, key);
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
    const auto recursive_map = arguments[2].ValueMap();

    mgp::Map map_removed = mgp::Map(std::move(map));
    bool recursive = false;

    if (!recursive_map.Empty() && !recursive_map.At("recursive").IsBool()) {
      throw mgp::ValueException(
          "Third argument must be a map with key recursive and value true or false.");  // neo4j ne baci exception samo
                                                                                        // nastavi kao da je dobio false
    }
    if (!recursive_map.Empty() && recursive_map.At("recursive").ValueBool() == true) {
      recursive = true;
    }

    removeRecursion(map_removed, recursive, key);

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRemoveKey).c_str(), std::move(map_removed));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}