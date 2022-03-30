#include <omp.h>

#include <chrono>
#include <queue>
#include <set>
#include <unordered_map>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

struct EdgeDelete {
  void operator()(mgp_edge *e) {
    // if (e) mgp::edge_destroy(e);
  }
};

struct EdgeStore {
  std::unordered_map<std::uint64_t, std::unique_ptr<mgp_edge, EdgeDelete>> _edge_map;

  void put(mgp_edge *edge) {
    _edge_map.emplace((std::uint64_t)mgp::edge_get_id(edge).as_int, std::unique_ptr<mgp_edge, EdgeDelete>(edge));
  }

  mgp_edge *get(std::uint64_t edge_id) { return _edge_map.at(edge_id).get(); }
};

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
                      std::uint64_t target_id, const std::vector<std::uint64_t> &edge_ids, EdgeStore &store) {
  auto *record = mgp::result_new_record(result);

  auto edges_size = edge_ids.size();
  auto path = mgp::path_make_with_start(mgp::edge_get_from(store.get(edge_ids[edges_size - 1])), memory);
  for (std::int32_t i = edges_size - 1; i >= 0; --i) {
    auto edge = store.get(edge_ids[i]);
    mgp::path_expand(path, edge);
  }

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

void ShortestPath(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    double t1, t2, t3, t4, t5, t6, t7, t8, t9 = 0;
    double module_time = 0;

    auto start_module = StartTimer();

    auto st1 = StartTimer();
    auto targets = mg_utility::GetNodeIDs(mgp::value_get_list(mgp::list_at(args, 0)));
    auto targets_size = targets.size();
    t1 += EndTimer(st1);
    

    omp_set_dynamic(0);
    omp_set_num_threads(8);
#pragma omp parallel for
    for (std::size_t i = 0; i < targets_size; ++i) {
      EdgeStore store;
      auto target = targets[i];

      std::priority_queue<PriorityPathItem> priority_queue;
      std::unordered_map<std::uint64_t, std::int64_t> shortest_path_length;
      std::set<std::uint64_t> visited;

      priority_queue.push({0, target, std::vector<std::uint64_t>{}});

      auto st2 = StartTimer();
      while (!priority_queue.empty()) {
        auto [distance, node_id, path] = priority_queue.top();
        priority_queue.pop();

        auto st3 = StartTimer();
        auto stop = visited.find(node_id) != visited.end() && distance > shortest_path_length[node_id];
        t3 += EndTimer(st3);
        if (stop) {
          continue;
        }

        visited.emplace(node_id);
        shortest_path_length[node_id] = distance;

        if (!path.empty()) {
          auto st9 = StartTimer();
          InsertPathResult(memgraph_graph, result, memory, node_id, target, path, store);
          t9 += EndTimer(st9);
        }

        /// Directed graph
        auto st4 = StartTimer();
        auto node = mgp::graph_get_vertex_by_id(memgraph_graph, mgp_vertex_id{.as_int = (int)node_id}, memory);
        t4 += EndTimer(st4);

        auto st5 = StartTimer();
        auto *edges_it = mgp::vertex_iter_in_edges(node, memory);
        // mg_utility::OnScopeExit delete_edges_it([&edges_it] { mgp::edges_iterator_destroy(edges_it); });
        t5 += EndTimer(st5);

        for (auto *in_edge = mgp::edges_iterator_get(edges_it); in_edge; in_edge = mgp::edges_iterator_next(edges_it)) {
          auto st6 = StartTimer();
          auto nxt_vertex = mgp::edge_get_from(in_edge);
          auto nxt_edge_id = (std::uint64_t)mgp::edge_get_id(in_edge).as_int;
          auto nxt_vertex_id = (std::uint64_t)mgp::vertex_get_id(nxt_vertex).as_int;
          auto nxt_distance = distance + 1;

// #pragma omp atomic
          store.put(in_edge);
          t6 += EndTimer(st6);

          auto st7 = StartTimer();
          std::vector<std::uint64_t> nxt_path(path.begin(), path.end());
          nxt_path.emplace_back(nxt_edge_id);
          t7 += EndTimer(st7);

          auto st8 = StartTimer();
          priority_queue.push({nxt_distance, nxt_vertex_id, nxt_path});
          t8 += EndTimer(st8);
        }
      }
      t2 += EndTimer(st2);
    }

    module_time = EndTimer(start_module);

    std::cout << "#----------------------------------------------------------#" << std::endl;
    std::cout << "Fetch Arguments: " << std::to_string(t1) << " ms"
              << " " << std::to_string((t1 / module_time) * 100) << "%" << std::endl;
    std::cout << "Check stopping: " << std::to_string(t3) << " ms"
              << " " << std::to_string((t3 / module_time) * 100) << "%" << std::endl;
    std::cout << "Fetch next by ID: " << std::to_string(t4) << " ms"
              << " " << std::to_string((t4 / module_time) * 100) << "%" << std::endl;
    std::cout << "Create edge iterator: " << std::to_string(t5) << " ms"
              << " " << std::to_string((t5 / module_time) * 100) << "%" << std::endl;
    std::cout << "Fetch edge and vertex ID: " << std::to_string(t6) << " ms"
              << " " << std::to_string((t6 / module_time) * 100) << "%" << std::endl;
    std::cout << "Copy path vector: " << std::to_string(t7) << " ms"
              << " " << std::to_string((t7 / module_time) * 100) << "%" << std::endl;
    std::cout << "Push to priority queue: " << std::to_string(t8) << " ms"
              << " " << std::to_string((t8 / module_time) * 100) << "%" << std::endl;
    std::cout << "Record saving: " << std::to_string(t9) << " ms"
              << " " << std::to_string((t9 / module_time) * 100) << "%" << std::endl;
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
