#include "map.hpp"

void Map::removeRecursion(mgp::Map &removed, bool recursive, std::string_view key) {
  for (auto element : removed) {
    if (element.key == key) {
      removed.Erase(key);
    } else if (element.value.IsMap() && recursive) {
      // removeRecursion(, recursive, key);
    }
  }
}

void Map::RemoveKey(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto map = arguments[0].ValueMap();
    const auto key = arguments[1].ValueString();
    const auto recursive_map = arguments[2].ValueMap();

    bool recursive = false;

    if (!recursive_map.Empty() && !recursive_map.At("recursive").IsBool()) {
      throw mgp::ValueException(
          "Third argument must be a map with key recursive and value true or false.");  // neo4j ne baci exception samo
                                                                                        // nastavi kao da je dobio false
    }
    if (!recursive_map.Empty() && recursive_map.At("recursive").ValueBool() == true) {
      recursive = true;
    }

    removeRecursion(map, recursive, key);

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultRemoveKey).c_str(), std::move(map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}