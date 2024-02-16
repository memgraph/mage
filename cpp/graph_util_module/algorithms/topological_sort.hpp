#include <list>

#include <mgp.hpp>

const char *kResultSortedNodes = "sorted_nodes";

void TopologicalSort(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto record_factory = mgp::RecordFactory(result);
  const auto graph = mgp::Graph(memgraph_graph);
  mgp::List topological_ordering = mgp::List();

  std::map<mgp::Node, uint64_t> in_degrees;

  for (const auto node : graph.Nodes()) {
    in_degrees[node] = 0;
  }

  for (const auto relationship : graph.Relationships()) {
    const auto to_node = relationship.To();
    in_degrees[to_node] += 1;
  }

  std::list<mgp::Node> nodes_with_no_incoming_edges;
  for (const auto node : graph.Nodes()) {
    if (in_degrees[node] == 0) {
      nodes_with_no_incoming_edges.emplace_back(node);
    }
  }

  while (nodes_with_no_incoming_edges.size() > 0) {
    const auto node = nodes_with_no_incoming_edges.back();
    nodes_with_no_incoming_edges.pop_back();
    topological_ordering.AppendExtend(mgp::Value(node));

    for (const auto relationship : node.OutRelationships()) {
      const auto to_node = relationship.To();
      in_degrees[to_node] -= 1;

      if (in_degrees[to_node] == 0) {
        nodes_with_no_incoming_edges.emplace_back(to_node);
      }
    }
  }

  if (topological_ordering.Size() == graph.Order()) {
    auto record = record_factory.NewRecord();
    record.Insert(kResultSortedNodes, topological_ordering);
    return;
  }

  record_factory.SetErrorMessage("The graph is cyclic and therefore no topological ordering exists.");
}
