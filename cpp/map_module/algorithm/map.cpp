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
    const mgp::Map &map = arguments[0].ValueMap();
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
