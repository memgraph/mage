#include <chrono>
#include <queue>
#include <random>

#include <gtest/gtest.h>
#include <mg_graph.hpp>

#include "algorithm/bridges.hpp"

/// This class is threadsafe
class Timer {
 public:
  Timer() : start_time_(std::chrono::steady_clock::now()) {}

  template <typename TDuration = std::chrono::duration<double>>
  TDuration Elapsed() const {
    return std::chrono::duration_cast<TDuration>(std::chrono::steady_clock::now() - start_time_);
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

namespace bruteforce_alg {

uint32_t CountComponents(const mg_graph::Graph<> *G) {
  uint32_t n_components = 0;
  std::vector<bool> visited(G->Nodes().size());

  for (auto &node : G->Nodes()) {
    if (visited[node.id]) continue;

    ++n_components;

    std::queue<uint64_t> Q;
    visited[node.id] = true;
    Q.push(node.id);
    while (!Q.empty()) {
      auto curr_id = Q.front();
      Q.pop();
      for (const auto &neigh : G->Neighbours(curr_id)) {
        auto next_id = neigh.node_id;
        if (visited[next_id]) continue;
        visited[next_id] = true;
        Q.push(next_id);
      }
    }
  }
  return n_components;
}

std::vector<std::pair<uint64_t, uint64_t>> GetBridges(mg_graph::Graph<> *G) {
  int comp_cnt = CountComponents(G);
  std::vector<std::pair<uint64_t, uint64_t>> bridges;
  auto edges = G->ExistingEdges();
  for (const auto &edge : edges) {
    uint32_t u = edge.from, v = edge.to;
    G->EraseEdge(u, v);
    int new_comp_cnt = CountComponents(G);

    if (new_comp_cnt > comp_cnt) bridges.emplace_back(u, v);
    G->CreateEdge(u, v);
  }
  return bridges;
}
}  // namespace bruteforce_alg

/// Builds the graph from a given number of nodes and a list of edges.
/// Nodes should be 0-indexed and each edge should be provided in both
/// directions.
inline mg_graph::Graph<> *BuildGraph(uint64_t nodes, std::vector<std::pair<uint64_t, uint64_t>> edges) {
  auto *G = new mg_graph::Graph<>();
  for (uint32_t i = 0; i < nodes; ++i) G->CreateNode(i);
  for (auto &p : edges) G->CreateEdge(p.first, p.second);

  return G;
}

/// Generates random undirected graph with a given numer of nodes and edges.
/// The generated graph is not picked out of a uniform distribution.
inline mg_graph::Graph<> *GenRandomGraph(uint64_t nodes, uint64_t edges) {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<uint64_t> dist(0, nodes - 1);
  std::set<std::pair<uint64_t, uint64_t>> E;
  for (uint64_t i = 0; i < edges; ++i) {
    int64_t u, v;
    do {
      u = dist(rng);
      v = dist(rng);
      if (u > v) std::swap(u, v);
    } while (u == v || E.find({u, v}) != E.end());
    E.insert({u, v});
  }
  return BuildGraph(nodes, std::vector<std::pair<uint64_t, uint64_t>>(E.begin(), E.end()));
}

/// Generates a random undirected tree with a given number of nodes.
/// The generated tree is not picked out of a uniform distribution.
inline mg_graph::Graph<> *GenRandomTree(uint64_t nodes) {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::vector<std::pair<uint64_t, uint64_t>> edges;
  for (uint64_t i = 1; i < nodes; ++i) {
    std::uniform_int_distribution<uint64_t> dist(0, i - 1);
    uint64_t dad = dist(rng);
    edges.emplace_back(dad, i);
  }
  return BuildGraph(nodes, edges);
}

/// Checks if obtained list of bridges is correct.
bool CheckBridges(std::vector<mg_graph::Edge<>> user, std::vector<std::pair<uint64_t, uint64_t>> correct) {
  std::set<std::pair<uint64_t, uint64_t>> user_bridge_set, correct_bridge_set;
  for (auto &p : correct) {
    correct_bridge_set.insert(p);
    correct_bridge_set.insert({p.second, p.first});
  }
  for (auto &edge : user) {
    user_bridge_set.insert({edge.from, edge.to});
    user_bridge_set.insert({edge.to, edge.from});
  }
  return user_bridge_set == correct_bridge_set;
}

TEST(Bridges, EmptyGraph) {
  auto *G = BuildGraph(0, {});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, SingleNode) {
  auto *G = BuildGraph(1, {});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, DisconnectedNodes) {
  auto *G = BuildGraph(100, {});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, Cycle) {
  uint64_t n = 100;
  std::vector<std::pair<uint64_t, uint64_t>> E;
  for (uint32_t i = 0; i < n; ++i) {
    E.emplace_back(i, (i + 1) % n);
  }
  auto *G = BuildGraph(n, E);
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, SmallTree) {
  //    (4)
  //   /   \
  // (2)   (1)
  //  |   /   \
  // (0)(3)   (5)
  auto *G = BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}}));
}

TEST(Bridges, RandomTree) {
  auto *G = GenRandomTree(10000);
  std::vector<std::pair<uint64_t, uint64_t>> edges;

  for (const auto &edge : G->Edges()) edges.emplace_back(edge.from, edge.to);

  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, edges));
}

TEST(Bridges, HandmadeConnectedGraph1) {
  //    (1)--(3)--(7)
  //   / |    |
  // (0) |   (4)--(5)
  //   \ |     \  /
  //    (2)     (6)
  auto *G = BuildGraph(8, {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 4}, {3, 7}, {4, 5}, {4, 6}, {5, 6}});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {{1, 3}, {3, 4}, {3, 7}}));
}

TEST(Bridges, HandmadeConnectedGraph2) {
  //    (1)--(3)--(7)     (10)     (14)
  //   / |    |    |     /    \     |
  // (0) |   (4)--(5)--(9)   (12)--(13)
  //   \ |     \  / \    \   /
  //    (2)     (6) (8)   (11)
  auto *G = BuildGraph(15, {{0, 1},
                            {0, 2},
                            {1, 2},
                            {1, 3},
                            {3, 4},
                            {3, 7},
                            {4, 5},
                            {4, 6},
                            {5, 6},
                            {5, 7},
                            {5, 8},
                            {5, 9},
                            {9, 10},
                            {9, 11},
                            {10, 12},
                            {11, 12},
                            {12, 13},
                            {13, 14}});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {{1, 3}, {5, 8}, {5, 9}, {12, 13}, {13, 14}}));
}

TEST(Bridges, HandmadeDisconnectedGraph) {
  //    (4)--(5)                   (12)         (19)         (23)
  //     |    |                   /    \        /           /   \
  //    (1)--(3)               (11)    (13)--(18)--(20)--(22)--(24)
  //   /                        |       |       \
  // (0)             (8)--(9)  (10)----(14)      (21)--(25)
  //   \                        |
  //    (2)--(6)               (15)----(16)
  //      \  /                    \    /
  //       (7)                     (17)
  auto *G = BuildGraph(
      26, {{0, 1},   {0, 2},   {1, 4},   {1, 3},   {3, 5},   {4, 5},   {2, 6},   {2, 7},   {6, 7},   {8, 9},
           {10, 11}, {11, 12}, {12, 13}, {13, 14}, {10, 14}, {10, 15}, {15, 16}, {16, 17}, {15, 17}, {13, 18},
           {18, 19}, {18, 21}, {18, 20}, {21, 25}, {20, 22}, {22, 23}, {23, 24}, {22, 24}});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_TRUE(CheckBridges(
      bridges, {{0, 1}, {0, 2}, {8, 9}, {10, 15}, {13, 18}, {18, 19}, {18, 20}, {18, 21}, {21, 25}, {20, 22}}));
}

TEST(Bridges, SimpleNeighborCycle) {
  auto *G = BuildGraph(2, {{0, 1}, {1, 0}});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_EQ(0, bridges.size());
}

TEST(Bridges, NeighborCycle) {
  auto *G = BuildGraph(6, {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {3, 4}, {3, 4}, {4, 5}});
  auto bridges = bridges_alg::GetBridges(G);
  ASSERT_EQ(1, bridges.size());
  ASSERT_EQ(bridges[0].id, 6);
}

TEST(Bridges, Random100) {
  for (int t = 0; t < 100; ++t) {
    auto G = GenRandomGraph(10, 20);
    auto algo_bridges = bridges_alg::GetBridges(G);
    auto bf_bridges = bruteforce_alg::GetBridges(G);
    ASSERT_TRUE(CheckBridges(algo_bridges, bf_bridges));
  }
}

TEST(Bridges, Performance) {
  auto G = GenRandomGraph(10000, 25000);
  Timer timer;
  auto bridges = bridges_alg::GetBridges(G);
  auto time_elapsed = timer.Elapsed();
  ASSERT_TRUE(timer.Elapsed() < std::chrono::seconds(100));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}