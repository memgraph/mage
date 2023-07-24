#include <list>
#include <unordered_set>

#include <mgp.hpp>

constexpr std::string_view kResultUnion = "union";

void Union(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto &list1 = arguments[0].ValueList();
    const auto &list2 = arguments[1].ValueList();

    std::unordered_set<mgp::Value, mgp::Value::Hash> unionSet;

    for (const auto value : list1) {
      unionSet.insert(value);
    }
    for (const auto value : list2) {
      unionSet.insert(value);
    }

    mgp::List unionList;

    for (const auto value : unionSet) {
      unionList.AppendExtend(value);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultUnion).c_str(), unionList);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
