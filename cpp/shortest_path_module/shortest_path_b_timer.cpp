#include <omp.h>

#include <chrono>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include "fibonacci_heap.hpp"

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

constexpr char const *kProcedureGet = "get";

constexpr char const *kArgumentSources = "sources";
constexpr char const *kArgumentTargets = "targets";

constexpr char const *kFieldSource = "source";
constexpr char const *kFieldTarget = "target";
constexpr char const *kFieldPath = "path";

struct MgConnectorData {
  mgp_graph *memgraph_graph;
  mgp_result *result;
  mgp_memory *memory;
  mg_utility::EdgeStore &store;
  mg_graph::Graph<uint64_t> &graph;
};

void InsertPathResult(struct MgConnectorData &conn, std::uint64_t source_id, std::uint64_t target_id,
                      std::vector<std::uint64_t> &edge_ids) {
  // void InsertPathResult(mgp_graph *graph, mgp_result *result, mgp_memory *memory, std::uint64_t source_id,
  //                       std::uint64_t target_id, std::vector<std::uint64_t> &edge_ids, mg_utility::EdgeStore &store)
  //                       {
  auto *record = mgp::result_new_record(conn.result);

  auto edges_size = edge_ids.size();
  auto path = mgp::path_make_with_start(mgp::edge_get_from(conn.store.Get(edge_ids[0])), conn.memory);
  for (std::int32_t i = 0; i < edges_size; ++i) {
    auto edge = conn.store.Get(edge_ids[i]);
    mgp::path_expand(path, edge);
  }

  // Insert records in Memgraph
  mg_utility::InsertNodeValueResult(conn.memgraph_graph, record, kFieldSource, source_id, conn.memory);
  mg_utility::InsertNodeValueResult(conn.memgraph_graph, record, kFieldTarget, target_id, conn.memory);
  mg_utility::InsertPathValueResult(record, kFieldPath, path, conn.memory);
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

void DFS_get_paths(std::unordered_map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> &prev,
                   std::uint64_t source_v, std::uint64_t current_v, std::vector<std::uint64_t> &path,
                   struct MgConnectorData &conn) {
  // check if target is source
  if (prev[current_v][0].first == -1) {
#pragma omp critical
    InsertPathResult(conn, source_v, conn.graph.GetMemgraphNodeId(current_v), path);
    // InsertPathResult(memgraph_graph, result, memory, source_v, graph.GetMemgraphNodeId(current_v), path, store);
    return;
  }

  for (std::pair<uint64_t, uint64_t> par : prev[current_v]) {
    // could the push_back and pop_back be placed outside of for?
    path.push_back(par.second);

    DFS_get_paths(prev, source_v, par.first, path, conn);

    path.pop_back();
  }
}

typedef std::pair<std::uint32_t, std::uint64_t> iPair;

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
    auto targets_tmp = FetchNodeIDs(*graph.get(), targets_arg);
    std::unordered_set<std::uint64_t> targets(targets_tmp.begin(), targets_tmp.end());

    // Fetch sources inner IDs. If not provided, fetch all.
    auto sources = FetchNodeIDs(*graph.get(), sources_arg);
    auto sources_size = sources.size();

    t1 += EndTimer(st1);
    omp_set_dynamic(0);
    omp_set_num_threads(8);

    // Dijsktra with priority queue. Parallel for each source
#pragma omp parallel for
    for (std::size_t i = 0; i < sources_size; ++i) {
      auto st2 = StartTimer();

      auto source = sources[i];

      fibonacci_heap<std::int32_t, std::uint64_t> *priority_queue;
      priority_queue = new fibonacci_heap<std::int32_t, std::uint64_t>([](int k1, int k2) { return k1 < k2; });
      std::unordered_map<std::uint64_t, std::vector<std::pair<std::uint64_t, std::uint64_t>>> prev;
      std::unordered_map<std::uint64_t, std::uint64_t> dist;
      std::unordered_set<std::uint64_t> visited;

      // std::priority_queue<iPair, std::vector<iPair>, std::greater<iPair>> priority_queue;

      // warning: assinging -1 to uint
      prev[source].push_back(std::make_pair(-1, -1));
      // priority_queue.push(std::make_pair(0, source));
      priority_queue->insert(0, source);

      t2 += EndTimer(st2);
      auto st3 = StartTimer();

      // while (!priority_queue.empty()) {
      while (!priority_queue->empty()) {
        auto st4 = StartTimer();

        auto [distance, node_id] = priority_queue->get();
        priority_queue->remove();
        // auto [distance, node_id] = priority_queue.top();
        // priority_queue.pop();

        if (visited.find(node_id) != visited.end()) continue;

        visited.emplace(node_id);

        t4 += EndTimer(st4);

        auto st5 = StartTimer();

        // Traverse in-neighbors and append to priority queue
        for (auto [nxt_vertex_id, nxt_edge_id] : graph.get()->InNeighbours(node_id)) {
          auto nxt_distance = distance + 1;

          if (dist.find(nxt_vertex_id) == dist.end()) {
            // hasn't been visited yet
            dist[nxt_vertex_id] = nxt_distance;
            prev[nxt_vertex_id].push_back(std::make_pair(node_id, nxt_edge_id));
            // priority_queue.push(std::make_pair(nxt_distance, nxt_vertex_id));
            priority_queue->insert(nxt_distance, nxt_vertex_id);
          } else if (nxt_distance < dist[nxt_vertex_id]) {
            // has been visited, but found a shorter path
            prev[nxt_vertex_id] = std::vector<std::pair<std::uint64_t, std::uint64_t>>();
            prev[nxt_vertex_id].push_back(std::make_pair(node_id, nxt_edge_id));
            dist[nxt_vertex_id] = nxt_distance;
            priority_queue->update_key(nxt_distance, nxt_vertex_id);
            // priority_queue.push(std::make_pair(nxt_distance, nxt_vertex_id));
          } else if (nxt_distance == dist[nxt_vertex_id]) {
            // found a path of same length
            prev[nxt_vertex_id].push_back(std::make_pair(node_id, nxt_edge_id));
          }
        }

        t5 += EndTimer(st5);
      }

      t3 += EndTimer(st3);

      auto st6 = StartTimer();

      // so there is no path of length 0 given
      visited.erase(source);

      MgConnectorData conn = {memgraph_graph, result, memory, *edge_store.get(), *graph.get()};

      std::vector<std::uint64_t> path = std::vector<std::uint64_t>();
      uint64_t source_v = graph.get()->GetMemgraphNodeId(source);
      for (auto target : visited) {
        DFS_get_paths(prev, source_v, target, path, conn);
      }

      t6 += EndTimer(st6);
    }

    module_time = EndTimer(start_module);

    std::cout << "#----------------------------------------------------------#" << std::endl;
    std::cout << "Fetch Arguments: " << std::to_string(t1) << " ms"
              << " " << std::to_string((t1 / module_time) * 100) << "%" << std::endl;
    std::cout << "for loop setup: " << std::to_string(t2) << " ms"
              << " " << std::to_string((t2 / module_time) * 100) << "%" << std::endl;
    std::cout << "while loop total: " << std::to_string(t3) << " ms"
              << " " << std::to_string((t3 / module_time) * 100) << "%" << std::endl;
    std::cout << "Check visited: " << std::to_string(t4) << " ms"
              << " " << std::to_string((t4 / module_time) * 100) << "%" << std::endl;
    std::cout << "Neighbour traversal total: " << std::to_string(t5) << " ms"
              << " " << std::to_string((t5 / module_time) * 100) << "%" << std::endl;
    std::cout << "Recursive DFS: " << std::to_string(t6) << " ms"
              << " " << std::to_string((t6 / module_time) * 100) << "%" << std::endl;
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
