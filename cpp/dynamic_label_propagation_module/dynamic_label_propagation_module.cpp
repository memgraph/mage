#include <mg_graph.hpp>
#include <mg_utils.hpp>

#include "algorithm/dynamic_label_propagation.hpp"

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldCommunityId = "community_id";

constexpr char const *kDirected = "directed";
constexpr char const *kWeightProperty = "weight_property";
constexpr char const *kWSelfloop = "w_selfloop";
constexpr char const *kSimilarityThreshold = "similarity_threshold";
constexpr char const *kExponent = "exponent";
constexpr char const *kMinValue = "min_value";
constexpr char const *kMaxIterations = "max_iterations";
constexpr char const *kMaxUpdates = "max_updates";

constexpr char const *kCreatedVertices = "createdVertices";
constexpr char const *kCreatedEdges = "createdEdges";
constexpr char const *kUpdatedVertices = "updatedVertices";
constexpr char const *kUpdatedEdges = "updatedEdges";
constexpr char const *kDeletedVertices = "deletedVertices";
constexpr char const *kDeletedEdges = "deletedEdges";

std::unique_ptr<mg_graph::Graph<>> graph;
LabelRankT::LabelRankT algorithm = LabelRankT::LabelRankT(graph);
bool initialized = false;
auto direction_parameter = mg_graph::GraphType::kDirectedGraph;

void InsertCommunityDetectionRecord(mgp_graph *graph, mgp_result *result,
                                    mgp_memory *memory,
                                    const std::uint64_t node_id,
                                    std::uint64_t community_id) {
  auto *record = mgp::result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
  mg_utility::InsertIntValueResult(record, kFieldCommunityId, community_id,
                                   memory);
}

void DetectWrapper(mgp_list *args, mgp_graph *memgraph_graph,
                         mgp_result *result, mgp_memory *memory) {
  try {
    auto directed = mgp::value_get_bool(mgp::list_at(args, 0));
    std::string weight_property =
        kWeightProperty;  // add after get_weight() is implemented
    auto w_selfloop = mgp::value_get_double(mgp::list_at(args, 1));
    auto similarity_threshold = mgp::value_get_double(mgp::list_at(args, 2));
    auto exponent = mgp::value_get_double(mgp::list_at(args, 3));
    auto min_value = mgp::value_get_double(mgp::list_at(args, 4));
    auto max_iterations = mgp::value_get_int(mgp::list_at(args, 5));
    auto max_updates = mgp::value_get_int(mgp::list_at(args, 6));

    if (directed) {
      direction_parameter = mg_graph::GraphType::kDirectedGraph;
    } else {
      direction_parameter = mg_graph::GraphType::kUndirectedGraph;
    }

    graph = mg_utility::GetGraphView(memgraph_graph, result, memory,
                                     direction_parameter);
    algorithm.set_parameters(weight_property, w_selfloop, similarity_threshold,
                             exponent, min_value);
    initialized = true;

    auto labels = algorithm.calculate_labels(max_iterations, max_updates);

    for (const auto [node_id, label] : labels) {
      InsertCommunityDetectionRecord(memgraph_graph, result, memory, node_id,
                                     label);
    }

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void GetWrapper(mgp_list *args, mgp_graph *memgraph_graph,
                mgp_result *result, mgp_memory *memory) {
  try {
    if (initialized) {
      auto labels = algorithm.get_labels();

      for (const auto [node_id, label] : labels) {
        InsertCommunityDetectionRecord(memgraph_graph, result, memory, node_id,
                                       label);
      }
    } else {
      graph = mg_utility::GetGraphView(memgraph_graph, result, memory,
                                       direction_parameter);

      auto labels = algorithm.calculate_labels();

      for (const auto [node_id, label] : labels) {
        InsertCommunityDetectionRecord(memgraph_graph, result, memory, node_id,
                                       label);
      }
    }

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void UpdateWrapper(mgp_list *args, mgp_graph *memgraph_graph,
                   mgp_result *result, mgp_memory *memory) {
  try {
    auto created_nodes = mgp::value_get_list(mgp::list_at(args, 0));
    auto created_edges = mgp::value_get_list(mgp::list_at(args, 1));
    auto updated_nodes = mgp::value_get_list(mgp::list_at(args, 2));
    auto updated_edges = mgp::value_get_list(mgp::list_at(args, 3));
    auto deleted_nodes = mgp::value_get_list(mgp::list_at(args, 4));
    auto deleted_edges = mgp::value_get_list(mgp::list_at(args, 5));

    if (initialized) {
      auto modified_node_ids = mg_utility::get_node_ids(created_nodes);
      auto modified_edge_endpoint_ids =
          mg_utility::get_edge_endpoint_ids(created_edges);

      auto updated_node_ids = mg_utility::get_node_ids(updated_nodes);
      modified_node_ids.insert(modified_node_ids.end(),
                               updated_node_ids.begin(),
                               updated_node_ids.end());
      auto updated_edge_endpoint_ids =
          mg_utility::get_edge_endpoint_ids(updated_edges);
      modified_edge_endpoint_ids.insert(modified_edge_endpoint_ids.end(),
                                        updated_edge_endpoint_ids.begin(),
                                        updated_edge_endpoint_ids.end());

      auto deleted_node_ids = mg_utility::get_node_ids(deleted_nodes);
      auto deleted_edge_endpoint_ids =
          mg_utility::get_edge_endpoint_ids(deleted_edges);

      graph = mg_utility::GetGraphView(memgraph_graph, result, memory,
                                       direction_parameter);

      auto labels =
          algorithm.update_labels(modified_node_ids, modified_edge_endpoint_ids,
                                  deleted_node_ids, deleted_edge_endpoint_ids);
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module,
                               struct mgp_memory *memory) {
  auto default_directed = mgp::value_make_bool(1, memory);
  // auto default_weight_property = mgp::value_make_string("weight", memory);
  auto default_w_selfloop = mgp::value_make_double(1.0, memory);
  auto default_similarity_threshold = mgp::value_make_double(0.7, memory);
  auto default_exponent = mgp::value_make_double(4.0, memory);
  auto default_min_value = mgp::value_make_double(0.1, memory);
  auto default_max_iterations = mgp::value_make_int(100, memory);
  auto default_max_updates = mgp::value_make_int(5, memory);

  try {
    struct mgp_proc *get_proc =
        mgp::module_add_read_procedure(module, "get", GetWrapper);
    struct mgp_proc *detect_proc =
        mgp::module_add_read_procedure(module, "detect", DetectWrapper);
    struct mgp_proc *update_proc =
        mgp::module_add_read_procedure(module, "update", UpdateWrapper);

    mgp::proc_add_opt_arg(detect_proc, kDirected, mgp::type_bool(),
                          default_directed);
    // mgp::proc_add_opt_arg(detect_proc, kWeightProperty, mgp::type_string(),
    //                       default_weight_property);
    mgp::proc_add_opt_arg(detect_proc, kWSelfloop, mgp::type_float(),
                          default_w_selfloop);
    mgp::proc_add_opt_arg(detect_proc, kSimilarityThreshold, mgp::type_float(),
                          default_similarity_threshold);
    mgp::proc_add_opt_arg(detect_proc, kExponent, mgp::type_float(),
                          default_exponent);
    mgp::proc_add_opt_arg(detect_proc, kMinValue, mgp::type_float(),
                          default_min_value);
    mgp::proc_add_opt_arg(detect_proc, kMaxIterations, mgp::type_int(),
                          default_max_iterations);
    mgp::proc_add_opt_arg(detect_proc, kMaxUpdates, mgp::type_int(),
                          default_max_updates);

    mgp::proc_add_arg(update_proc, kCreatedVertices,
                      mgp::type_list(mgp::type_node()));
    mgp::proc_add_arg(update_proc, kCreatedEdges,
                      mgp::type_list(mgp::type_relationship()));
    mgp::proc_add_arg(update_proc, kUpdatedVertices,
                      mgp::type_list(mgp::type_node()));
    mgp::proc_add_arg(update_proc, kUpdatedEdges,
                      mgp::type_list(mgp::type_relationship()));
    mgp::proc_add_arg(update_proc, kDeletedVertices,
                      mgp::type_list(mgp::type_node()));
    mgp::proc_add_arg(update_proc, kDeletedEdges,
                      mgp::type_list(mgp::type_relationship()));
    // Query module output record
    mgp::proc_add_result(get_proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(get_proc, kFieldCommunityId, mgp::type_int());

    mgp::proc_add_result(detect_proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(detect_proc, kFieldCommunityId, mgp::type_int());
  } catch (const std::exception &e) {
    mgp::value_destroy(default_directed);
    // mgp::value_destroy(default_weight_property);
    mgp::value_destroy(default_w_selfloop);
    mgp::value_destroy(default_similarity_threshold);
    mgp::value_destroy(default_exponent);
    mgp::value_destroy(default_min_value);
    mgp::value_destroy(default_max_iterations);
    mgp::value_destroy(default_max_updates);
    return 1;
  }

  mgp::value_destroy(default_directed);
  // mgp::value_destroy(default_weight_property);
  mgp::value_destroy(default_w_selfloop);
  mgp::value_destroy(default_similarity_threshold);
  mgp::value_destroy(default_exponent);
  mgp::value_destroy(default_min_value);
  mgp::value_destroy(default_max_iterations);
  mgp::value_destroy(default_max_updates);

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
