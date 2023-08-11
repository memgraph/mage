#include "nodes.hpp"
#include "mgp.hpp"

namespace {
void ThrowException(const mgp::Value &value) {
  std::ostringstream oss;
  oss << value.Type();
  throw mgp::ValueException("Unsuppported type for this operation, received type: " + oss.str());
};

void DetachDeleteNode(const mgp::Value &node, mgp::Graph &graph) {
  if (node.IsInt()) {
    graph.DetachDeleteNode(graph.GetNodeById(mgp::Id::FromInt(node.ValueInt())));
  } else if (node.IsNode()) {
    graph.DetachDeleteNode(node.ValueNode());
  } else {
    ThrowException(node);
  }
}

}  // namespace

void Nodes::Delete(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph{memgraph_graph};
    auto nodes{arguments[0]};

    if (!nodes.IsList()) {
      DetachDeleteNode(nodes, graph);
      return;
    }

    for (const auto &list_item : nodes.ValueList()) {
      DetachDeleteNode(list_item, graph);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
