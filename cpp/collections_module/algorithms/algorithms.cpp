#include <vector>

#include "algorithms.hpp"

void Collections::Pairs(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();
  try {
    mgp::List pairsList = mgp::List();

    const auto &inputList = arguments[0].ValueList();

    if (inputList.Size() == 0) {
      record.Insert(std::string(kResultPairs).c_str(), pairsList);
      return;
    }
    for (size_t i = 0; i < inputList.Size() - 1; i++) {
      mgp::List helper = mgp::List();
      helper.AppendExtend(inputList[i]);
      helper.AppendExtend(inputList[i + 1]);
      pairsList.AppendExtend(mgp::Value(std::move(helper)));
    }
    mgp::List helper = mgp::List();
    helper.AppendExtend(inputList[inputList.Size() - 1]);
    helper.AppendExtend(mgp::Value());
    pairsList.AppendExtend(mgp::Value(std::move(helper)));

    record.Insert(std::string(kResultPairs).c_str(), pairsList);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
