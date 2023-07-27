#include "map.hpp"

void Map::removeRecursionSet(mgp::Map &result, bool recursive, std::unordered_set<std::string_view> &set) {
  for (auto element : result) {
    bool inSet = false;
    if (set.find(element.key) != set.end()) {
      inSet = true;
    }
    if (inSet) {
      result.Erase(element.key);
      continue;
    }
    if (element.value.IsMap() && recursive) {
      mgp::Map non_const_value_map = mgp::Map(std::move(element.value.ValueMap()));
      removeRecursionSet(non_const_value_map, recursive, set);
      if (non_const_value_map.Empty()) {
        result.Erase(element.key);
        continue;
      }
      result.Update(element.key, mgp::Value(std::move(non_const_value_map)));
    }
  }
}

void Map::RemoveKeys(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Map map = arguments[0].ValueMap();
    const mgp::List list = arguments[1].ValueList();
    bool recursive = arguments[2].ValueBool();
    std::unordered_set<std::string_view> set;
    for (auto elem : list) {
      set.insert(std::move(elem.ValueString()));
    }
    removeRecursionSet(map, recursive, set);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Map::kReturnRemoveKeys).c_str(), std::move(map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
