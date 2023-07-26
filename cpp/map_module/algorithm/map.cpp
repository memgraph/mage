#include "map.hpp"

void Map::FromLists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List list1 = arguments[0].ValueList();  // lef this non const, because of the fix
    mgp::List list2 = arguments[1].ValueList();

    const size_t n = list1.Size();
    if (n != list2.Size() || n == 0) {
      throw mgp::ValueException("Lists must be of same size and not null");
    }
    mgp::Map result = mgp::Map();
    for (size_t i = 0; i < n; i++) {
      result.Update(std::move(list1[i].ValueString()), std::move(list2[i]));
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(Map::kReturnListFromLists).c_str(), std::move(result));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
