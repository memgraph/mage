#include "refactor.hpp"

void Refactor::TransferProperties(const mgp::Node &node, mgp::Relationship &rel) {
  for (auto &[key, value] : node.Properties()) {
    rel.SetProperty(key, value);
  }
}

void Refactor::Collapse(mgp::Graph &graph, const mgp::Node &node, const std::string &type,
                        const mgp::RecordFactory &record_factory) {
  if (node.InDegree() != 1 || node.OutDegree() != 1) {
    throw mgp::ValueException("Out and in degree of the nodes both must be 1!");
  }

  const mgp::Node from_node = (*node.InRelationships().begin()).From();
  const mgp::Node to_node = (*node.OutRelationships().begin()).To();
  if (from_node == node && to_node == node) {
    throw mgp::ValueException("Nodes with self relationships are non collapsible!");
  }
  mgp::Relationship new_rel = graph.CreateRelationship(from_node, to_node, type);
  TransferProperties(node, new_rel);

  auto record = record_factory.NewRecord();
  record.Insert(std::string(kReturnIdCollapseNode).c_str(), node.Id().AsInt());
  record.Insert(std::string(kReturnRelationshipCollapseNode).c_str(), new_rel);
  graph.DetachDeleteNode(node);
}

void Refactor::CollapseNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Value input = arguments[0];
    const std::string type{arguments[1].ValueString()};

    if(!input.IsNode() && !input.IsInt() && !input.IsList()){
      record_factory.SetErrorMessage("Input can only be node, node ID, or list of nodes/IDs");
      return;
    }

    if (input.IsNode()) {
      const mgp::Node node = input.ValueNode();
      Collapse(graph, node, type, record_factory);
    } else if (input.IsInt()) {
      const mgp::Node node = graph.GetNodeById(mgp::Id::FromInt(input.ValueInt()));
      Collapse(graph, node, type, record_factory);
    } else if (input.IsList()) {
      for (auto elem : input.ValueList()) {
        if (elem.IsNode()) {
          const mgp::Node node = elem.ValueNode();
          Collapse(graph, node, type, record_factory);
        } else if (elem.IsInt()) {
          const mgp::Node node = graph.GetNodeById(mgp::Id::FromInt(elem.ValueInt()));
          Collapse(graph, node, type, record_factory);
        } else {
          record_factory.SetErrorMessage("Elements in the list can only be Node or ID");
          return;
        }
      }
    } 
      
    

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
