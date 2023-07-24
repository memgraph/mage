#include <mgp.hpp>

#include "algorithm.hpp"

void Collections::UnionAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List list1 = arguments[0].ValueList();
    mgp::List list2 = arguments[1].ValueList();

    for (size_t i = 0; i < list2.Size(); i++) {
      list1.AppendExtend(std::move(list2[i]));
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnValueUnionAll).c_str(), list1);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
