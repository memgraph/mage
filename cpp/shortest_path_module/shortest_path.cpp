#include <queue>
#include <set>
#include <unordered_map>

#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

struct VertexDelete {
  void operator()(mgp_vertex *v) {
    if (v) mgp::vertex_destroy(v);
  }
};

struct EdgeDelete {
  void operator()(mgp_edge *e) {
    if (e) mgp::edge_destroy(e);
  }
};

static std::shared_ptr<mgp_vertex> CreateVertexPointer(mgp_vertex *v, mgp_memory *memory) {
  return std::shared_ptr<mgp_vertex>(mgp::vertex_copy(v, memory), VertexDelete());
}

static std::shared_ptr<mgp_edge> CreateEdgePointer(mgp_edge *e, mgp_memory *memory) {
  return std::shared_ptr<mgp_edge>(mgp::edge_copy(e, memory), EdgeDelete());
}

struct PriorityPathItem {
  std::int32_t distance;
  std::shared_ptr<mgp_vertex> vertex;
  std::vector<std::shared_ptr<mgp_edge>> path;

  bool operator<(const PriorityPathItem &other) const { return distance > other.distance; }
};

constexpr char const *kProcedureGet = "get";

constexpr char const *kArgumentSources = "sources";
constexpr char const *kArgumentTargets = "targets";

constexpr char const *kFieldSource = "source";
constexpr char const *kFieldTarget = "target";
constexpr char const *kFieldPath = "path";

std::vector<mgp_vertex *> ListToVector(mgp_list *list) {
  auto size = mgp::list_size(list);

  std::vector<mgp_vertex *> container;
  container.reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    auto vertex = mgp::value_get_vertex(mgp::list_at(list, i));
    container.emplace_back(vertex);
  }

  return container;
}

void InsertPathResult(mgp_graph *graph, mgp_result *result, mgp_memory *memory, mgp_vertex *source, mgp_vertex *target,
                      std::vector<std::shared_ptr<mgp_edge>> &edges) {
  auto *record = mgp::result_new_record(result);

  auto edges_size = edges.size();
  auto path = mgp::path_make_with_start(mgp::edge_get_from(edges[edges_size - 1].get()), memory);
  for (std::int32_t i = edges_size - 1; i >= 0; --i) {
    auto edge = edges[i].get();
    mgp::path_expand(path, edge);
  }

  mg_utility::InsertNodeValueResult(record, kFieldSource, mgp::vertex_copy(source, memory), memory);
  mg_utility::InsertNodeValueResult(record, kFieldTarget, mgp::vertex_copy(target, memory), memory);
  mg_utility::InsertPathValueResult(record, kFieldPath, path, memory);
}

void ShortestPath(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    auto targets = ListToVector(mgp::value_get_list(mgp::list_at(args, 0)));

    for (auto target : targets) {
      std::priority_queue<PriorityPathItem> priority_queue;
      std::unordered_map<std::uint64_t, std::int64_t> shortest_path_length;
      std::set<std::uint64_t> visited;

      priority_queue.push({0, CreateVertexPointer(target, memory), std::vector<std::shared_ptr<mgp_edge>>{}});

      while (!priority_queue.empty()) {
        auto [distance, node_ptr, path] = priority_queue.top();
        auto node = node_ptr.get();

        priority_queue.pop();

        auto node_id = mgp::vertex_get_id(node).as_int;
        if (visited.find(node_id) != visited.end() && distance > shortest_path_length[node_id]) {
          continue;
        }
        visited.emplace(node_id);
        shortest_path_length[node_id] = distance;

        if (!path.empty()) {
          InsertPathResult(memgraph_graph, result, memory, node, target, path);
        }

        /// Directed graph
        auto *edges_it = mgp::vertex_iter_in_edges(node, memory);
        mg_utility::OnScopeExit delete_edges_it([&edges_it] { mgp::edges_iterator_destroy(edges_it); });

        for (auto *in_edge = mgp::edges_iterator_get(edges_it); in_edge; in_edge = mgp::edges_iterator_next(edges_it)) {
          auto vertex_in = mgp::edge_get_from(in_edge);

          std::vector<std::shared_ptr<mgp_edge>> new_path(path.begin(), path.end());
          new_path.emplace_back(CreateEdgePointer(in_edge, memory));

          priority_queue.push({distance + 1, CreateVertexPointer(vertex_in, memory), new_path});
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
