#include <memory>
#include <unordered_map>

#include <mg_graph.hpp>
#include "pagerank.hpp"

namespace pagerank_online_alg {
namespace {

class PageRankData {
  ///
  ///@brief Context for storing data for dynamic pagerank
  ///
  ///
 public:
  void Init() {
    walks.clear();
    walks_counter.clear();
    walks_table.clear();
  }

  bool IsEmpty() const { return walks.empty(); }

  /// Keeping the information about walks on the graph
  std::vector<std::vector<std::uint64_t>> walks;

  // Keeping the information of walk appearance in algorithm for faster calculation
  std::unordered_map<std::uint64_t, uint64_t> walks_counter;

  /// Table that keeps the node appearance and walk ID
  std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> walks_table;
};

// Dynamic global context
// This is unsafe in the multithreaded environment, the workaround would be building a thread-safe dynamic storage
// implementation
PageRankData context;
std::uint64_t global_R;
double global_epsilon;

int GetRandInt(int from, int to) {
  static std::minstd_rand eng{std::random_device{}()};
  std::uniform_int_distribution<int> dist{from, to - 1};
  return dist(eng);
}

float GetRandFloat() {
  static std::minstd_rand eng{std::random_device{}()};
  static std::uniform_real_distribution<float> dist{};
  return dist(eng);
}

///
///@brief Function for vector normalization
///
///@param rank Vector that needs to be normalized
///
void NormalizeRank(std::vector<std::pair<std::uint64_t, double>> &rank) {
  const double sum =
      std::accumulate(rank.begin(), rank.end(), 0.0, [](auto sum, const auto &p) { return sum + p.second; });
  for (auto &[node_id, value] : rank) {
    value /= sum;
  }
}

///
///@brief Calculates pagerank based on current information stored in global context
///
///@return std::vector<std::pair<std::uint64_t, double>>
///
std::vector<std::pair<std::uint64_t, double>> CalculatePageRank() {
  std::vector<std::pair<std::uint64_t, double>> pageranks;

  auto R = pagerank_online_alg::global_R;
  auto eps = pagerank_online_alg::global_epsilon;

  auto n = pagerank_online_alg::context.walks_counter.size();
  pageranks.reserve(n);
  for (auto const [node_id, total] : pagerank_online_alg::context.walks_counter) {
    auto rank = total / ((n * R) / eps);
    pageranks.emplace_back(node_id, rank);
  }

  NormalizeRank(pageranks);
  return pageranks;
}

///
///@brief Creates a route starting from start_id, stores it in walk and updates the walk_index. Route is created via
/// random walk depending on random number genrator.
///
///@param graph Graph for route creation
///@param start_id Starting node in graph creation
///@param walk Walk vector that stores a route
///@param walk_index Index of a walk for context storing
///@param epsilon Probability of stopping the route creation
///
void CreateRoute(const mg_graph::GraphView<> &graph, std::uint64_t start_id, std::vector<std::uint64_t> &walk,
                 std::uint64_t walk_index, double epsilon) {
  std::uint64_t current_id = start_id;
  while (true) {
    auto neighbors = graph.Neighbours(current_id);
    if (neighbors.empty()) break;

    // Pick and add the random outer relationship
    auto number_of_neighbors = neighbors.size();
    auto next_id = neighbors[GetRandInt(0, number_of_neighbors)].node_id;
    next_id = graph.GetMemgraphNodeId(next_id);

    walk.emplace_back(next_id);
    pagerank_online_alg::context.walks_table[next_id].insert(walk_index);
    pagerank_online_alg::context.walks_counter[next_id]++;

    // Finish walk when random number is smaller than epsilon
    // Average length of walk is 1/epsilon
    if (GetRandFloat() < epsilon) {
      break;
    }

    current_id = graph.GetInnerNodeId(next_id);
  }
}

///
///@brief Creates a route starting from start_id, stores it in walk and updates the walk_index. Route is created via
/// random walk depending on random number genrator.
///
///@param graph Graph for route creation
///@param start_id Starting node in graph creation
///@param walk Walk vector that stores a route
///@param walk_index Index of a walk for context storing
///@param epsilon Probability of stopping the route creation
///
void CreateRoute(const mgp::Graph &graph, std::uint64_t start_id, std::vector<std::uint64_t> &walk,
                 std::uint64_t walk_index, double epsilon) {
  std::uint64_t current_id = start_id;
  while (true) {
    std::vector<std::uint64_t> neighbor_ids;
    for (const auto out_relationship : graph.GetNodeById(mgp::Id::FromUint(current_id)).OutRelationships()) {
      neighbor_ids.push_back(out_relationship.To().Id().AsUint());
    }
    // auto neighbors = graph.Neighbours(current_id);

    if (neighbor_ids.empty()) break;

    // Pick and add a random outer relationship
    auto next_id = neighbor_ids[GetRandInt(0, neighbor_ids.size())];

    walk.emplace_back(next_id);
    pagerank_online_alg::context.walks_table[next_id].insert(walk_index);
    pagerank_online_alg::context.walks_counter[next_id]++;

    // Finish walk when random number is smaller than epsilon
    // Average length of walk is 1/epsilon
    if (GetRandFloat() < epsilon) {
      break;
    }

    // current_id = graph.GetInnerNodeId(next_id);
  }
}

///
///@brief Updates the context based on new relationship addition. Reverts previous walks made from starting node and
/// updates
/// them.
///
///@param graph Graph for updating
///@param new_relationship New relationship
///
void UpdateCreate(const mgp::Graph &graph, const std::pair<std::uint64_t, std::uint64_t> &new_relationship) {
  auto &[from, to] = new_relationship;

  std::unordered_set<std::uint64_t> walk_table_copy(pagerank_online_alg::context.walks_table[from]);
  for (auto walk_index : walk_table_copy) {
    auto &walk = pagerank_online_alg::context.walks[walk_index];

    auto position = std::find(walk.begin(), walk.end(), from) + 1;
    while (position != walk.end()) {
      auto node_id = *position;
      pagerank_online_alg::context.walks_table[node_id].erase(walk_index);
      pagerank_online_alg::context.walks_counter[node_id]--;
      position++;
    }
    walk.erase(std::find(walk.begin(), walk.end(), from) + 1, walk.end());

    auto current_id = from;
    auto half_eps = pagerank_online_alg::global_epsilon / 2.0;
    CreateRoute(graph, current_id, walk, walk_index, half_eps);
  }
}

///
///@brief Updates the context based on adding the new node. This means adding it to a context tables and creating
/// walks from it.
///
///@param graph Graph for updating
///@param new_node New node
///
void UpdateCreate(const mgp::Graph &graph, std::uint64_t new_node) {
  auto R = pagerank_online_alg::global_R;
  auto eps = pagerank_online_alg::global_epsilon;

  auto walk_index = pagerank_online_alg::context.walks.size();
  for (std::uint64_t i = 0; i < R; ++i) {
    std::vector<std::uint64_t> walk{new_node};

    pagerank_online_alg::context.walks_table[new_node].insert(walk_index);
    pagerank_online_alg::context.walks_counter[new_node]++;

    CreateRoute(graph, new_node, walk, walk_index, eps);

    pagerank_online_alg::context.walks.emplace_back(std::move(walk));
    walk_index++;
  }
}

///
///@brief Removes the relationship from the context and updates walks. This method works by updating walks that contain
/// starting
/// node because they no longer exist.
///
///@param graph Graph for updating
///@param removed_relationship Deleted relationship
///
void UpdateDelete(const mgp::Graph &graph, const std::pair<std::uint64_t, std::uint64_t> &removed_relationship) {
  auto &[from, to] = removed_relationship;

  std::unordered_set<std::uint64_t> walk_table_copy(pagerank_online_alg::context.walks_table[from]);
  for (auto walk_index : walk_table_copy) {
    auto &walk = pagerank_online_alg::context.walks[walk_index];

    auto position = std::find(walk.begin(), walk.end(), from) + 1;

    if (position == walk.end()) {
      continue;
    }

    while (position != walk.end()) {
      auto node_id = *position;
      pagerank_online_alg::context.walks_table[node_id].erase(walk_index);
      pagerank_online_alg::context.walks_counter[node_id]--;
      position++;
    }
    walk.erase(std::find(walk.begin(), walk.end(), from) + 1, walk.end());

    auto current_id = from;

    // Skip creating routes if node does not exist anymore
    if (!graph.ContainsNode(mgp::Id::FromUint(current_id))) {
      continue;
    }

    auto half_eps = pagerank_online_alg::global_epsilon / 2.0;
    CreateRoute(graph, current_id, walk, walk_index, half_eps);
  }
}

///
///@brief Deletes node from context. This is trivial because we are sure that no relationship exists around that node.
///
///@param graph Graph for updating
///@param removed_node Removed node
///
void UpdateDelete(std::uint64_t removed_node) {
  pagerank_online_alg::context.walks_table.erase(removed_node);
  pagerank_online_alg::context.walks_counter.erase(removed_node);
}

bool IsInconsistent(const mg_graph::GraphView<> &graph) {
  for (auto const [node_id] : graph.Nodes()) {
    auto external_id = graph.GetMemgraphNodeId(node_id);
    if (pagerank_online_alg::context.walks_counter.find(external_id) ==
        pagerank_online_alg::context.walks_counter.end()) {
      return true;
    }
  }
  return false;
}
}  // namespace

std::vector<std::pair<std::uint64_t, double>> SetPageRank(const mg_graph::GraphView<> &graph, std::uint64_t R,
                                                          double epsilon) {
  pagerank_online_alg::global_R = R;
  pagerank_online_alg::global_epsilon = epsilon;
  pagerank_online_alg::context.Init();

  auto walk_index = 0;
  for (auto [node_id] : graph.Nodes()) {
    // We have R random walks for each node in the graph
    for (std::uint64_t i = 0; i < R; i++) {
      std::vector<std::uint64_t> walk;

      auto current_id = graph.GetMemgraphNodeId(node_id);
      walk.emplace_back(current_id);
      pagerank_online_alg::context.walks_table[current_id].insert(walk_index);
      pagerank_online_alg::context.walks_counter[current_id]++;

      CreateRoute(graph, graph.GetInnerNodeId(current_id), walk, walk_index, epsilon);

      pagerank_online_alg::context.walks.emplace_back(std::move(walk));
      walk_index++;
    }
  }

  return CalculatePageRank();
}

std::vector<std::pair<std::uint64_t, double>> GetPageRank(const mg_graph::GraphView<> &graph) {
  if (pagerank_online_alg::context.IsEmpty()) {
    return SetPageRank(graph);
  }
  if (IsInconsistent(graph)) {
    throw std::runtime_error(
        "Graph has been modified and is thus inconsistent with cached PageRank scores. To update them, please call "
        "set/reset!");
  }
  return CalculatePageRank();
}

std::vector<std::pair<std::uint64_t, double>> UpdatePageRank(
    const mgp::Graph &graph, const std::vector<std::uint64_t> &new_nodes,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &new_relationships,
    const std::vector<std::uint64_t> &deleted_nodes,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_relationships) {
  for (const auto &relationship : deleted_relationships) {
    UpdateDelete(graph, relationship);
  }
  for (const auto node : deleted_nodes) {
    UpdateDelete(node);
  }
  for (const auto node : new_nodes) {
    UpdateCreate(graph, node);
  }
  for (const auto &relationship : new_relationships) {
    UpdateCreate(graph, relationship);
  }

  return CalculatePageRank();
}

bool ContextEmpty() { return pagerank_online_alg::context.IsEmpty(); }

void Reset() { pagerank_online_alg::context.Init(); }
}  // namespace pagerank_online_alg
