#include "path.hpp"

#include <list>

void Path::SubgraphAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List nodes = mgp::List();
    mgp::List rels = mgp::List();

    const auto config = arguments[1].ValueMap();

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultSubgraphAll).c_str(), std::move(nodes));
    record.Insert(std::string(kResultSubgraphAll).c_str(), std::move(rels));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
