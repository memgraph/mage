#include "algorithm_online/community_detection.hpp"

LabelRankT::LabelRankT algorithm = LabelRankT::LabelRankT();
bool initialized = false;

auto saved_directedness = false;
auto saved_weightedness = false;
std::string saved_weight_property = "weight";

void Set(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto graph = mgp::Graph(memgraph_graph);
    const auto arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);

    const auto directed = arguments[0].ValueBool();
    const auto weighted = arguments[1].ValueBool();
    const auto similarity_threshold = arguments[2].ValueDouble();
    const auto exponent = arguments[3].ValueDouble();
    const auto min_value = arguments[4].ValueDouble();
    const auto weight_property = arguments[5].ValueString();
    const auto w_selfloop = weighted ? arguments[6].ValueDouble() : 1.0;
    const auto max_iterations = arguments[7].ValueInt();
    const auto max_updates = arguments[8].ValueInt();

    ::saved_directedness = directed;
    ::saved_weightedness = weighted;
    ::saved_weight_property = weight_property;

    const auto labels = algorithm.SetLabels(graph, directed, weighted, similarity_threshold, exponent, min_value,
                                            weight_property.data(), w_selfloop, max_iterations, max_updates);
    ::initialized = true;

    for (const auto [node_id, label] : labels) {
      auto record = record_factory.NewRecord();
      record.Insert("node", graph.GetNodeById(mgp::Id::FromUint(node_id)));
      record.Insert("community_id", label);
    }
  } catch (const std::exception &e) {
    return;
  }
  return;
}

void Get(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto graph = mgp::Graph(memgraph_graph);
    const auto record_factory = mgp::RecordFactory(result);

    const auto labels = initialized ? algorithm.GetLabels(graph) : algorithm.SetLabels(graph);

    for (const auto [node_id, label] : labels) {
      // Previously-computed communities may contain nodes since deleted
      try {
        // If so, throw an exception:
        const auto maybe_node = graph.GetNodeById(mgp::Id::FromUint(node_id));

        // Otherwise:
        auto record = record_factory.NewRecord();
        record.Insert("node", graph.GetNodeById(mgp::Id::FromUint(node_id)));
        record.Insert("community_id", label);
      } catch (const std::exception &e) {
        continue;
      }
    }
  } catch (const std::exception &e) {
    return;
  }
  return;
}

void Update(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto graph = mgp::Graph(memgraph_graph);
    const auto arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);

    const auto created_nodes = arguments[0].ValueList();
    const auto created_relationships = arguments[1].ValueList();
    const auto updated_nodes = arguments[2].ValueList();
    const auto updated_relationships = arguments[3].ValueList();
    const auto deleted_nodes_ = arguments[4].ValueList();
    const auto deleted_relationships_ = arguments[5].ValueList();

    if (initialized) {
      // compute modified_nodes & modified_relationships

      std::unordered_set<std::uint64_t> modified_nodes;
      std::vector<std::pair<std::uint64_t, std::uint64_t>> modified_relationships;
      std::unordered_set<std::uint64_t> deleted_nodes;
      std::vector<std::pair<std::uint64_t, std::uint64_t>> deleted_relationships;

      for (const auto &node : created_nodes) {
        modified_nodes.insert(node.ValueNode().Id().AsUint());
      }
      for (const auto &relationship : created_relationships) {
        modified_relationships.push_back({relationship.ValueRelationship().From().Id().AsUint(),
                                          relationship.ValueRelationship().To().Id().AsUint()});
      }

      for (const auto &node : updated_nodes) {
        modified_nodes.insert(node.ValueNode().Id().AsUint());
      }
      for (const auto &relationship : updated_relationships) {
        modified_relationships.push_back({relationship.ValueRelationship().From().Id().AsUint(),
                                          relationship.ValueRelationship().To().Id().AsUint()});
      }

      for (const auto &node : deleted_nodes_) {
        deleted_nodes.insert(node.ValueNode().Id().AsUint());
      }
      for (const auto &relationship : deleted_relationships_) {
        deleted_relationships.push_back({relationship.ValueRelationship().From().Id().AsUint(),
                                         relationship.ValueRelationship().To().Id().AsUint()});
      }

      const auto labels =
          algorithm.UpdateLabels(graph, modified_nodes, modified_relationships, deleted_nodes, deleted_relationships);

      for (const auto [node_id, label] : labels) {
        auto record = record_factory.NewRecord();
        record.Insert("node", graph.GetNodeById(mgp::Id::FromUint(node_id)));
        record.Insert("community_id", label);
      }
    } else {
      const auto labels = algorithm.SetLabels(graph);

      for (const auto [node_id, label] : labels) {
        auto record = record_factory.NewRecord();
        record.Insert("node", graph.GetNodeById(mgp::Id::FromUint(node_id)));
        record.Insert("community_id", label);
      }
    }

  } catch (const std::exception &e) {
    return;
  }
  return;
}

void Reset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto record_factory = mgp::RecordFactory(result);

    ::algorithm = LabelRankT::LabelRankT();
    ::initialized = false;

    ::saved_directedness = false;
    ::saved_weightedness = false;
    ::saved_weight_property = "weight";

    auto record = record_factory.NewRecord();
    record.Insert("message", "The algorithm has been successfully reset!");
  } catch (const std::exception &e) {
    mgp::memory = memory;

    const auto record_factory = mgp::RecordFactory(result);
    auto record = record_factory.NewRecord();
    record.Insert("message", "Reset failed: An exception occurred, please check your module!");

    return;
  }
  return;
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto node_list = std::make_pair(mgp::Type::List, mgp::Type::Node);
    const auto relationship_list = std::make_pair(mgp::Type::List, mgp::Type::Relationship);
    const auto empty_list = mgp::Value(mgp::List());

    mgp::AddProcedure(
        Set, "set", mgp::ProcedureType::Read,
        {mgp::Parameter("directed", mgp::Type::Bool, false), mgp::Parameter("weighted", mgp::Type::Bool, false),
         mgp::Parameter("similarity_threshold", mgp::Type::Double, 0.7),
         mgp::Parameter("exponent", mgp::Type::Double, 4.0), mgp::Parameter("min_value", mgp::Type::Double, 0.1),
         mgp::Parameter("weight_property", mgp::Type::String, "weight"),
         mgp::Parameter("w_selfloop", mgp::Type::Double, 1.0),
         mgp::Parameter("max_iterations", mgp::Type::Int, (int64_t)100),
         mgp::Parameter("max_updates", mgp::Type::Int, (int64_t)5)},
        {mgp::Return("node", mgp::Type::Node), mgp::Return("community_id", mgp::Type::Int)}, module, memory);

    mgp::AddProcedure(Get, "get", mgp::ProcedureType::Read, {},
                      {mgp::Return("node", mgp::Type::Node), mgp::Return("community_id", mgp::Type::Int)}, module,
                      memory);

    mgp::AddProcedure(Update, "update", mgp::ProcedureType::Read,
                      {mgp::Parameter("createdVertices", node_list, empty_list),
                       mgp::Parameter("createdEdges", relationship_list, empty_list),
                       mgp::Parameter("updatedVertices", node_list, empty_list),
                       mgp::Parameter("updatedEdges", relationship_list, empty_list),
                       mgp::Parameter("deletedVertices", node_list, empty_list),
                       mgp::Parameter("deletedEdges", relationship_list, empty_list)},
                      {mgp::Return("node", mgp::Type::Node), mgp::Return("community_id", mgp::Type::Int)}, module,
                      memory);

    mgp::AddProcedure(Reset, "reset", mgp::ProcedureType::Read, {}, {mgp::Return("message", mgp::Type::String)}, module,
                      memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
