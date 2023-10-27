#include <list>

#include <mgp.hpp>

const char *kResultAncestors = "ancestors";

void Ancestors(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const auto given_node = arguments[0].ValueNode();

    mgp::List result = mgp::List();

    std::set<uint64_t> visited_node_ids;
    std::list<mgp::Node> next_nodes;

    for (const auto relationship : given_node.InRelationships()) {
      const auto source_node = relationship.From();

      if (visited_node_ids.find(source_node.Id().AsUint()) == visited_node_ids.end()) {
        visited_node_ids.insert(source_node.Id().AsUint());
        next_nodes.emplace_back(source_node);
      }
    }

    while (next_nodes.size() > 0) {
      const auto current_node = next_nodes.front();
      result.AppendExtend(mgp::Value(current_node));
      next_nodes.pop_front();

      for (const auto relationship : current_node.InRelationships()) {
        const auto source_node = relationship.From();

        if (visited_node_ids.find(source_node.Id().AsUint()) == visited_node_ids.end()) {
          visited_node_ids.insert(source_node.Id().AsUint());
          next_nodes.emplace_back(source_node);
        }
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert(kResultAncestors, result);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
