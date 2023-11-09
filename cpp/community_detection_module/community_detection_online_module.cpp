#include "algorithm_online/community_detection.hpp"

// Procedures
constexpr std::string_view kProcSet{"set"};
constexpr std::string_view kProcGet{"get"};
constexpr std::string_view kProcUpdate{"update"};
constexpr std::string_view kProcReset{"reset"};

// Arguments
constexpr std::string_view kDirected{"directed"};
constexpr std::string_view kWeighted{"weighted"};
constexpr std::string_view kSimilarityThreshold{"similarity_threshold"};
constexpr std::string_view kExponent{"exponent"};
constexpr std::string_view kMinValue{"min_value"};
constexpr std::string_view kWeightProperty{"weight_property"};
constexpr std::string_view kWSelfloop{"w_selfloop"};
constexpr std::string_view kMaxIterations{"max_iterations"};
constexpr std::string_view kMaxUpdates{"max_updates"};

// The following are predefined variables
// (https://memgraph.com/docs/memgraph/reference-guide/triggers#predefined-variables) and thus not renamed to
// nodes/relationships
constexpr std::string_view kCreatedVertices{"createdVertices"};
constexpr std::string_view kCreatedEdges{"createdEdges"};
constexpr std::string_view kUpdatedVertices{"updatedVertices"};
constexpr std::string_view kUpdatedEdges{"updatedEdges"};
constexpr std::string_view kDeletedVertices{"deletedVertices"};
constexpr std::string_view kDeletedEdges{"deletedEdges"};

// Default values
constexpr bool kDefaultDirected = false;
constexpr bool kDefaultWeighted = false;
constexpr double kDefaultSimilarityThreshold = 0.7;
constexpr double kDefaultExponent = 4.0;
constexpr double kDefaultMinValue = 0.1;
constexpr std::string_view kDefaultWeightProperty{"weight"};
constexpr double kDefaultWSelfloop = 1.0;
constexpr int64_t kDefaultMaxIterations = 5;
constexpr int64_t kDefaultMaxUpdates = 100;

// Returns
constexpr std::string_view kFieldNode{"node"};
constexpr std::string_view kFieldCommunityId{"community_id"};

constexpr std::string_view kFieldMessage{"message"};

LabelRankT::LabelRankT algorithm = LabelRankT::LabelRankT();
bool initialized = false;

auto saved_directedness = kDefaultDirected;
auto saved_weightedness = kDefaultWeighted;
std::string saved_weight_property = kDefaultWeightProperty.data();

void Set(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

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
      // As IN_MEMORY_ANALYTICAL doesn’t offer ACID guarantees, check if the graph elements in the result exist
      try {
        // If so, throw an exception:
        const auto maybe_node = graph.GetNodeById(mgp::Id::FromUint(node_id));

        // Otherwise:
        auto record = record_factory.NewRecord();
        record.Insert(kFieldNode.data(), graph.GetNodeById(mgp::Id::FromUint(node_id)));
        record.Insert(kFieldCommunityId.data(), label);
      } catch (const std::exception &e) {
        continue;
      }
    }
  } catch (const std::exception &e) {
    return;
  }
  return;
}

void Get(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

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
        record.Insert(kFieldNode.data(), graph.GetNodeById(mgp::Id::FromUint(node_id)));
        record.Insert(kFieldCommunityId.data(), label);
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
    mgp::MemoryDispatcherGuard guard{memory};

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
      // Compute modified_nodes & modified_relationships
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
        // As IN_MEMORY_ANALYTICAL doesn’t offer ACID guarantees, check if the graph elements in the result exist
        try {
          // If so, throw an exception:
          const auto maybe_node = graph.GetNodeById(mgp::Id::FromUint(node_id));

          // Otherwise:
          auto record = record_factory.NewRecord();
          record.Insert(kFieldNode.data(), graph.GetNodeById(mgp::Id::FromUint(node_id)));
          record.Insert(kFieldCommunityId.data(), label);
        } catch (const std::exception &e) {
          continue;
        }
      }
    } else {
      const auto labels = algorithm.SetLabels(graph);

      for (const auto [node_id, label] : labels) {
        // As IN_MEMORY_ANALYTICAL doesn’t offer ACID guarantees, check if the graph elements in the result exist
        try {
          // If so, throw an exception:
          const auto maybe_node = graph.GetNodeById(mgp::Id::FromUint(node_id));

          // Otherwise:
          auto record = record_factory.NewRecord();
          record.Insert(kFieldNode.data(), graph.GetNodeById(mgp::Id::FromUint(node_id)));
          record.Insert(kFieldCommunityId.data(), label);
        } catch (const std::exception &e) {
          continue;
        }
      }
    }

  } catch (const std::exception &e) {
    return;
  }
  return;
}

void Reset(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    const auto record_factory = mgp::RecordFactory(result);

    ::algorithm = LabelRankT::LabelRankT();
    ::initialized = false;

    ::saved_directedness = kDefaultDirected;
    ::saved_weightedness = kDefaultWeighted;
    ::saved_weight_property = kDefaultWeightProperty.data();

    auto record = record_factory.NewRecord();
    record.Insert(kFieldMessage.data(), "The algorithm has been successfully reset!");
  } catch (const std::exception &e) {
    mgp::MemoryDispatcherGuard guard{memory};

    const auto record_factory = mgp::RecordFactory(result);
    auto record = record_factory.NewRecord();
    record.Insert(kFieldMessage.data(), "Reset failed: An exception occurred, please check the module!");

    return;
  }
  return;
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    const auto node_list = std::make_pair(mgp::Type::List, mgp::Type::Node);
    const auto relationship_list = std::make_pair(mgp::Type::List, mgp::Type::Relationship);
    const auto empty_list = mgp::Value(mgp::List());

    mgp::AddProcedure(
        Set, kProcSet.data(), mgp::ProcedureType::Read,
        {mgp::Parameter(kDirected.data(), mgp::Type::Bool, kDefaultDirected),
         mgp::Parameter(kWeighted.data(), mgp::Type::Bool, kDefaultWeighted),
         mgp::Parameter(kSimilarityThreshold.data(), mgp::Type::Double, kDefaultSimilarityThreshold),
         mgp::Parameter(kExponent.data(), mgp::Type::Double, kDefaultExponent),
         mgp::Parameter(kMinValue.data(), mgp::Type::Double, kDefaultMinValue),
         mgp::Parameter(kWeightProperty.data(), mgp::Type::String, kDefaultWeightProperty.data()),
         mgp::Parameter(kWSelfloop.data(), mgp::Type::Double, kDefaultWSelfloop),
         mgp::Parameter(kMaxIterations.data(), mgp::Type::Int, (int64_t)kDefaultMaxIterations),
         mgp::Parameter(kMaxUpdates.data(), mgp::Type::Int, (int64_t)kDefaultMaxUpdates)},
        {mgp::Return(kFieldNode.data(), mgp::Type::Node), mgp::Return(kFieldCommunityId.data(), mgp::Type::Int)},
        module, memory);

    mgp::AddProcedure(
        Get, kProcGet.data(), mgp::ProcedureType::Read, {},
        {mgp::Return(kFieldNode.data(), mgp::Type::Node), mgp::Return(kFieldCommunityId.data(), mgp::Type::Int)},
        module, memory);

    mgp::AddProcedure(
        Update, kProcUpdate.data(), mgp::ProcedureType::Read,
        {mgp::Parameter(kCreatedVertices.data(), node_list, empty_list),
         mgp::Parameter(kCreatedEdges.data(), relationship_list, empty_list),
         mgp::Parameter(kUpdatedVertices.data(), node_list, empty_list),
         mgp::Parameter(kUpdatedEdges.data(), relationship_list, empty_list),
         mgp::Parameter(kDeletedVertices.data(), node_list, empty_list),
         mgp::Parameter(kDeletedEdges.data(), relationship_list, empty_list)},
        {mgp::Return(kFieldNode.data(), mgp::Type::Node), mgp::Return(kFieldCommunityId.data(), mgp::Type::Int)},
        module, memory);

    mgp::AddProcedure(Reset, kProcReset.data(), mgp::ProcedureType::Read, {},
                      {mgp::Return(kFieldMessage, mgp::Type::String)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
