#include <omp.h>

#include <chrono>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>

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

inline std::chrono::time_point<std::chrono::high_resolution_clock> StartTimer() {
  return std::chrono::high_resolution_clock::now();
}
inline double EndTimer(std::chrono::time_point<std::chrono::high_resolution_clock> start) {
  return ((std::chrono::duration<double, std::milli>)(std::chrono::high_resolution_clock::now() - start)).count();
}

std::vector<std::uint64_t> TransformNodeIDs(const mg_graph::GraphView<> &mg_graph,
                                            std::vector<std::uint64_t> &mg_nodes) {
  std::vector<std::uint64_t> nodes;
  nodes.reserve(mg_nodes.size());
  std::transform(
      mg_nodes.begin(), mg_nodes.end(), std::back_inserter(nodes),
      [&mg_graph](const std::uint64_t node_id) -> std::uint64_t { return mg_graph.GetInnerNodeId(node_id); });
  return nodes;
}

std::vector<std::uint64_t> FetchAllNodesIDs(const mg_graph::GraphView<> &mg_graph) {
  std::vector<uint64_t> nodes(mg_graph.Nodes().size());
  std::iota(nodes.begin(), nodes.end(), 0);
  return nodes;
}

std::vector<std::uint64_t> FetchNodeIDs(const mg_graph::GraphView<> &mg_graph, mgp_list *mg_nodes) {
  std::vector<uint64_t> nodes;
  if (mg_nodes != nullptr) {
    auto sources_arg = mg_utility::GetNodeIDs(mg_nodes);
    nodes = TransformNodeIDs(mg_graph, sources_arg);
  } else {
    nodes = FetchAllNodesIDs(mg_graph);
  }
  return nodes;
}

void ShortestPath(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    // Fetch the target & source IDs

    double t1, t2, t3, t4, t5, t6, t7, t8, t9 = 0;
    double module_time = 0;

    auto start_module = StartTimer();

    auto st1 = StartTimer();

    auto sources_arg =
        !mgp::value_is_null(mgp::list_at(args, 0)) ? mgp::value_get_list(mgp::list_at(args, 0)) : nullptr;

    auto targets_arg =
        !mgp::value_is_null(mgp::list_at(args, 1)) ? mgp::value_get_list(mgp::list_at(args, 1)) : nullptr;

    auto res = mg_utility::GetGraphViewWithEdge(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);
    const auto &graph = res.first;
    const auto &edge_store = res.second;

    // Fetch target inner IDs. If not provided, fetch all.
    auto targets = FetchNodeIDs(*graph.get(), targets_arg);
    auto targets_size = targets.size();

    // Fetch sources inner IDs. If not provided, fetch all.
    auto sources_tmp = FetchNodeIDs(*graph.get(), sources_arg);
    std::unordered_set<std::uint64_t> sources(sources_tmp.begin(), sources_tmp.end());

    t1 += EndTimer(st1);
    omp_set_dynamic(0);
    omp_set_num_threads(8);

    // Reversed Dijsktra with priority queue. Parallel for each target
#pragma omp parallel for
    for (std::size_t i = 0; i < targets_size; ++i) {
      auto target = targets[i];

      std::priority_queue<PriorityPathItem> priority_queue;
      std::unordered_map<std::uint64_t, std::int64_t> shortest_path_length;
      std::unordered_set<std::uint64_t> visited;

      priority_queue.push({0, target, std::vector<std::uint64_t>{}});

      auto st2 = StartTimer();
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
        auto st3 = StartTimer();
        if (!path.empty() && (sources.find(node_id) != sources.end())) {
#pragma omp critical
          InsertPathResult(memgraph_graph, result, memory, graph.get()->GetMemgraphNodeId(node_id),
                           graph.get()->GetMemgraphNodeId(target), path, *edge_store.get());
        }
        t3 += EndTimer(st3);

        auto st4 = StartTimer();
        // Traverse in-neighbors and append to priority queue
        for (auto [nxt_vertex_id, nxt_edge_id] : graph.get()->InNeighbours(node_id)) {
          auto nxt_distance = distance + 1;

          std::vector<std::uint64_t> nxt_path(path.begin(), path.end());
          nxt_path.emplace_back(nxt_edge_id);

          priority_queue.push({nxt_distance, nxt_vertex_id, nxt_path});
        }
        t4 += EndTimer(st4);
      }
      t2 += EndTimer(st2);
    }

    module_time = EndTimer(start_module);

    std::cout << "#----------------------------------------------------------#" << std::endl;
    std::cout << "Fetch Arguments: " << std::to_string(t1) << " ms"
              << " " << std::to_string((t1 / module_time) * 100) << "%" << std::endl;
    std::cout << "while loop total: " << std::to_string(t2) << " ms"
              << " " << std::to_string((t2 / module_time) * 100) << "%" << std::endl;
    std::cout << "insert results critical: " << std::to_string(t3) << " ms"
              << " " << std::to_string((t3 / module_time) * 100) << "%" << std::endl;
    std::cout << "neighbour traversal: " << std::to_string(t4) << " ms"
              << " " << std::to_string((t4 / module_time) * 100) << "%" << std::endl;
    std::cout << "Full module time: " << std::to_string(module_time) << " ms" << std::endl;
    std::cout << "#----------------------------------------------------------#" << std::endl;

  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    auto *wcc_proc = mgp::module_add_read_procedure(module, kProcedureGet, ShortestPath);

    auto default_null = mgp::value_make_null(memory);
    mgp::proc_add_opt_arg(wcc_proc, kArgumentSources, mgp::type_nullable(mgp::type_list(mgp::type_node())),
                          default_null);
    mgp::proc_add_opt_arg(wcc_proc, kArgumentTargets, mgp::type_nullable(mgp::type_list(mgp::type_node())),
                          default_null);

    mgp::proc_add_result(wcc_proc, kFieldSource, mgp::type_node());
    mgp::proc_add_result(wcc_proc, kFieldTarget, mgp::type_node());
    mgp::proc_add_result(wcc_proc, kFieldPath, mgp::type_path());

    mgp::value_destroy(default_null);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
