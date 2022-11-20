#include <set>

#include <mgp.hpp>

void ConnectNodes(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;
    const auto arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);
    const auto list_of_nodes = arguments[0].ValueList();

    mgp::List result = mgp::List();

    std::set<uint64_t> graph_nodes;
    for (const auto node_element : list_of_nodes) {
      const auto node = node_element.ValueNode();
      graph_nodes.insert(node.Id().AsUint());
      result.AppendExtend(node_element);
    }

    for (const auto node_element : list_of_nodes) {
      const auto node = node_element.ValueNode();
      for (const auto relationship : node.OutRelationships()) {
        const auto in_node_id = relationship.To().Id().AsUint();
        const auto out_node_id = node.Id().AsUint();

        if (graph_nodes.find(in_node_id) != graph_nodes.end() && graph_nodes.find(out_node_id) != graph_nodes.end()) {
          result.AppendExtend(mgp::Value(relationship));
        }
      }
    }

    for (auto value : result) {
      auto record = record_factory.NewRecord();
      if (value.IsNode()) {
        record.Insert("connected_graph", value.ValueNode());
      } else {
        record.Insert("connected_graph", value.ValueRelationship());
      }
    }

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto nodes_parameter = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(ConnectNodes, "connect_nodes", mgp::ProdecureType::Read, {mgp::Parameter("nodes", nodes_parameter)},
                 {mgp::Return("connected_graph", mgp::Type::Any)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
