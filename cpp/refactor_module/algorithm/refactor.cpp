#include "refactor.hpp"




void Refactor::CollapseNode(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::Graph graph = mgp::Graph(memgraph_graph);
    const mgp::Node node = arguments[0].ValueNode();
    
    auto in_rel = node.InRelationships();

    /*
    auto inDegree = std::distance(node.InRelationships().begin(), node.InRelationships().end());
    auto outDegree = std::distance(node.OutRelationships().begin(), node.OutRelationships().end());

    if(inDegree != 1 && outDegree != 1){
        throw mgp::ValueException("Out and in degree of the nodes both must be 1!");
    } 


    //discuss if exception throw if from_node == node == to_node

    mgp::Node from_node = (*node.InRelationships().begin()).From();
    mgp::Node to_node = (*node.OutRelationships().begin()).To();
    mgp::Relationship new_rel = graph.CreateRelationship(from_node, to_node, "aa");
    graph.DetachDeleteNode(node);
    */
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}