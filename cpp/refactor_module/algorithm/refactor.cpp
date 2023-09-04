#include "refactor.hpp"

void TransferProperties(const mgp::Node &node, mgp::Relationship &rel){
  for(auto &[key, value]: node.Properties()){
    rel.SetProperty(key, value);
  }

}


void Refactor::CollapseNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Node node = arguments[0].ValueNode();
    
    if(node.InDegree() != 1 && node.OutDegree() != 1){
        throw mgp::ValueException("Out and in degree of the nodes both must be 1!");
    } 

    
    const mgp::Node from_node = (*node.InRelationships().begin()).From();
    const mgp::Node to_node = (*node.OutRelationships().begin()).To();

    
    if(from_node == node && to_node == node){
      throw mgp::ValueException("Nodes with self relationships are non collapsible!");
    }
    

    mgp::Relationship new_rel = graph.CreateRelationship(from_node, to_node, "aa");
    TransferProperties(node, new_rel);
    graph.DetachDeleteNode(node);
 

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}