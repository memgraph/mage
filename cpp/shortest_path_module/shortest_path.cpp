#include <omp.h>

#include <chrono>
#include <queue>
#include <set>
#include <unordered_map>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

struct PriorityPathItem {
  std::int32_t distance;
  std::uint64_t vertex;
  std::vector<std::uint64_t> path;

  bool operator<(const PriorityPathItem &other) const { return distance > other.distance; }
};

constexpr char const *kProcedureGet = "get";

constexpr char const *kArgumentSources = "sources";
constexpr char const *kArgumentTargets = "targets";

constexpr char const *kFieldSource = "source";
constexpr char const *kFieldTarget = "target";
constexpr char const *kFieldPath = "path";

void InsertPathResult(mgp_graph *graph, mgp_result *result, mgp_memory *memory, std::uint64_t source_id,
                      std::uint64_t target_id, std::vector<std::uint64_t> &edge_ids, mg_utility::EdgeStore &store) {
  auto *record = mgp::result_new_record(result);

  // Construct the graph out of reversed edge list
  auto edges_size = edge_ids.size();
  auto path = mgp::path_make_with_start(mgp::edge_get_from(store.Get(edge_ids[edges_size - 1])), memory);
  for (std::int32_t i = edges_size - 1; i >= 0; --i) {
    auto edge = store.Get(edge_ids[i]);
    mgp::path_expand(path, edge);
  }

  // Insert records in Memgraph
  mg_utility::InsertNodeValueResult(graph, record, kFieldSource, source_id, memory);
  mg_utility::InsertNodeValueResult(graph, record, kFieldTarget, target_id, memory);
  mg_utility::InsertPathValueResult(record, kFieldPath, path, memory);
}

void ShortestPath(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    // Fetch the target IDs
    auto targets_arg = mg_utility::GetNodeIDs(mgp::value_get_list(mgp::list_at(args, 0)));
    auto targets_size = targets_arg.size();

    // Construct the graph view
    auto res = mg_utility::GetGraphViewWithEdge(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    const auto &graph = res.first;
    const auto &edge_store = res.second;

    // Get targets as graph view IDs
    std::vector<std::uint64_t> targets;
    targets.reserve(targets_size);
    std::transform(
        targets_arg.begin(), targets_arg.end(), std::back_inserter(targets),
        [&graph](const std::uint64_t node_id) -> std::uint64_t { return graph.get()->GetInnerNodeId(node_id); });

    // Reversed Dijsktra with priority queue. Parallel for each target
#pragma omp parallel for
    for (std::size_t i = 0; i < targets_size; ++i) {
      auto target = targets[i];

      std::priority_queue<PriorityPathItem> priority_queue;
      std::unordered_map<std::uint64_t, std::int64_t> shortest_path_length;
      std::set<std::uint64_t> visited;

      priority_queue.push({0, target, std::vector<std::uint64_t>{}});

      while (!priority_queue.empty()) {
        auto [distance, node_id, path] = priority_queue.top();
        priority_queue.pop();

        // No expansion if distance is higher
        if (visited.find(node_id) != visited.end() && distance > shortest_path_length[node_id]) {
          continue;
        }

        visited.emplace(node_id);
        shortest_path_length[node_id] = distance;

        // If path is found, insert in Memgraph result
        if (!path.empty()) {
#pragma omp critical
          InsertPathResult(memgraph_graph, result, memory, graph.get()->GetMemgraphNodeId(node_id),
                           graph.get()->GetMemgraphNodeId(target), path, *edge_store.get());
        }

        // Traverse in-neighbors and append to priority queue
        for (auto [nxt_vertex_id, nxt_edge_id] : graph.get()->InNeighbours(node_id)) {
          auto nxt_distance = distance + 1;

          std::vector<std::uint64_t> nxt_path(path.begin(), path.end());
          nxt_path.emplace_back(nxt_edge_id);

          priority_queue.push({nxt_distance, nxt_vertex_id, nxt_path});
        }
      }
    }
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    auto *wcc_proc = mgp::module_add_read_procedure(module, kProcedureGet, ShortestPath);

    mgp::proc_add_arg(wcc_proc, kArgumentTargets, mgp::type_list(mgp::type_node()));

    mgp::proc_add_result(wcc_proc, kFieldSource, mgp::type_node());
    mgp::proc_add_result(wcc_proc, kFieldTarget, mgp::type_node());
    mgp::proc_add_result(wcc_proc, kFieldPath, mgp::type_path());
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
