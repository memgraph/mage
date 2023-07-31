#include "map.hpp"

void Map::flattenRecursion(mgp::Map &result, const mgp::Map &input, const std::string &key,
                           const std::string &delimiter) {
  for (auto element : input) {
    std::string el_key(element.key);
    if (element.value.IsMap()) {
      flattenRecursion(result, element.value.ValueMap(), key + el_key + delimiter, delimiter);
    } else {
      result.Insert(key + el_key, element.value);
    }
  }
}

void Map::Flatten(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::Map map = arguments[0].ValueMap();
    const std::string delimiter(arguments[1].ValueString());
    mgp::Map result_map = mgp::Map();
    flattenRecursion(result_map, map, "", delimiter);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Map::kReturnValueFlatten).c_str(), std::move(result_map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Map::FromLists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List list1 = arguments[0].ValueList(); 
    mgp::List list2 = arguments[1].ValueList();

    const auto expected_list_size = list1.Size();
    if (expected_list_size  != list2.Size() || expected_list_size  == 0) {
      throw mgp::ValueException("Lists must be of same size and not empty");
    }
    mgp::Map result = mgp::Map();
    for (size_t i = 0; i < expected_list_size ; i++) {
      result.Update(std::move(list1[i].ValueString()), std::move(list2[i]));
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Map::kReturnListFromLists).c_str(), std::move(result));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Map::RemoveRecursionSet(mgp::Map &result, bool recursive, std::unordered_set<std::string_view> &set) {
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
      RemoveRecursionSet(non_const_value_map, recursive, set);
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
    RemoveRecursionSet(map, recursive, set);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Map::kReturnRemoveKeys).c_str(), std::move(map));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
