#include "algo.hpp"

void Algo::Cover(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto list_nodes = arguments[0].ValueList();
    std::unordered_set<mgp::Node> nodes;
    for (const auto elem : list_nodes) {
      auto node = elem.ValueNode();
      nodes.insert(node);
    }

    for (auto node : nodes) {
      for (auto rel : node.OutRelationships()) {
        if (nodes.find(rel.To()) != nodes.end()) {
          auto record = record_factory.NewRecord();
          record.Insert(std::string(kCoverRet1).c_str(), rel);
        }
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}