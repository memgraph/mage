#include <list>
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

    auto record = record_factory.NewRecord();
    record.Insert("connections", result);

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void Descendants(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;
    const auto arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);
    const auto given_node = arguments[0].ValueNode();

    mgp::List result = mgp::List();

    std::set<uint64_t> visited_node_ids;
    std::list<mgp::Node> next_nodes;

    for (const auto relationship : given_node.OutRelationships()) {
      const auto target_node = relationship.To();

      if (visited_node_ids.find(target_node.Id().AsUint()) == visited_node_ids.end()) {
        visited_node_ids.insert(target_node.Id().AsUint());
        next_nodes.emplace_back(target_node);
      }
    }

    while (next_nodes.size()) {
      const auto current_node = next_nodes.front();
      result.AppendExtend(mgp::Value(current_node));
      next_nodes.pop_front();

      for (const auto relationship : current_node.OutRelationships()) {
        const auto target_node = relationship.To();

        if (visited_node_ids.find(target_node.Id().AsUint()) == visited_node_ids.end()) {
          visited_node_ids.insert(target_node.Id().AsUint());
          next_nodes.emplace_back(target_node);
        }
      }
    }

    auto record = record_factory.NewRecord();
    record.Insert("descendants", result);

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void Ancestors(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;
    const auto arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);
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
    record.Insert("ancestors", result);

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void TopologicalSort(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;
    const auto record_factory = mgp::RecordFactory(result);
    const auto graph = mgp::Graph(memgraph_graph);

    std::map<mgp::Node, uint64_t> in_degrees;

    for (const auto node : graph.Nodes()) {
      in_degrees[node] = 0;
    }

    for (const auto node : graph.Nodes()) {
      for (const auto relationship : node.OutRelationships()) {
        const auto to_node = relationship.To();
        in_degrees[to_node] += 1;
      }
    }

    std::list<mgp::Node> nodes_with_no_incoming_edges;
    for (const auto node : graph.Nodes()) {
      if (in_degrees[node] == 0) {
        nodes_with_no_incoming_edges.emplace_back(node);
      }
    }

    mgp::List topological_ordering = mgp::List();

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
      record.Insert("sorted_nodes", topological_ordering);
      return;
    }

    throw std::runtime_error("Graph has a cycle! No topological ordering exists.");

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    // Register ancestors procedure
    const auto ancestors_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(Ancestors, "ancestors", mgp::ProdecureType::Read, {mgp::Parameter("node", mgp::Type::Node)},
                 {mgp::Return("ancestors", ancestors_return)}, module, memory);

    // Register connect nodes procedure
    const auto connect_nodes_input = std::make_pair(mgp::Type::List, mgp::Type::Node);
    const auto connect_nodes_return = std::make_pair(mgp::Type::List, mgp::Type::Relationship);

    AddProcedure(ConnectNodes, "connect_nodes", mgp::ProdecureType::Read,
                 {mgp::Parameter("nodes", connect_nodes_input)}, {mgp::Return("connections", connect_nodes_return)},
                 module, memory);

    // Register descendants procedure
    const auto descendants_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(Descendants, "descendants", mgp::ProdecureType::Read, {mgp::Parameter("node", mgp::Type::Node)},
                 {mgp::Return("descendants", descendants_return)}, module, memory);

    // Register topological sort procedure
    const auto topological_sort_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(TopologicalSort, "topological_sort", mgp::ProdecureType::Read, {},
                 {mgp::Return("sorted_nodes", topological_sort_return)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
