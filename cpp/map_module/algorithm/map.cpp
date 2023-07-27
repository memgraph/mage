#include "map.hpp"

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
