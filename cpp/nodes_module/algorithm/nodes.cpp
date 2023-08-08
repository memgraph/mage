#include "nodes.hpp"
#include "mgp.hpp"

namespace {
void throw_exception(const mgp::Value &value) {
  std::ostringstream oss;
  oss << value.Type();
  throw mgp::ValueException("Unsuppported type for this operation, received type: " + oss.str());
};

}  // namespace

void Nodes::Delete(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph{memgraph_graph};
    auto nodes{arguments[0]};

    if (nodes.IsInt()) {
      graph.DetachDeleteNode(graph.GetNodeById(mgp::Id::FromInt(nodes.ValueInt())));
    } else if (nodes.IsNode()) {
      graph.DetachDeleteNode(nodes.ValueNode());
    } else if (nodes.IsList()) {
      for (const auto &list_item : nodes.ValueList()) {
        if (list_item.IsInt()) {
          graph.DetachDeleteNode(graph.GetNodeById(mgp::Id::FromInt(list_item.ValueInt())));
        } else if (list_item.IsNode()) {
          graph.DetachDeleteNode(list_item.ValueNode());
        } else {
          throw_exception(list_item);
        }
      }
    } else {
      throw_exception(nodes);
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
