#include "date.hpp"

void Date::Parse(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto graph = mgp::Graph(memgraph_graph);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    int64_t parsed{0};
    parsed = 5;
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultParsed).c_str(), std::move(parsed));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
