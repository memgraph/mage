#include <glog/logging.h>
#include <gtest/gtest.h>

#include <queue>
#include <random>

#include "algorithms/algorithms.hpp"
#include "utils.hpp"

using namespace graphdata;

namespace bruteforce {
int CountComponents(const Graph &G) {
  int ret = 0;
  std::vector<bool> visited(G.Nodes().size());

  for (auto &node : G.Nodes()) {
    if (visited[node.id]) continue;
    ++ret;
    std::queue<uint32_t> Q;
    visited[node.id] = true;
    Q.push(node.id);
    while (!Q.empty()) {
      uint32_t curr_id = Q.front();
      Q.pop();
      for (const auto &neigh : G.Neighbours(curr_id)) {
        uint32_t next_id = neigh.node_id;
        if (visited[next_id]) continue;
        visited[next_id] = true;
        Q.push(next_id);
      }
    }
  }
  return ret;
}

std::vector<std::pair<uint32_t, uint32_t>> GetBridges(Graph &G) {
  int comp_cnt = CountComponents(G);
  std::vector<std::pair<uint32_t, uint32_t>> bridges;
  std::vector<Edge> edges = G.ExistingEdges();
  for (const Edge &edge : edges) {
    uint32_t u = edge.from, v = edge.to;
    G.EraseEdge(u, v);
    int new_comp_cnt = CountComponents(G);
    CHECK(new_comp_cnt >= comp_cnt)
        << "Erasing an edge can't reduce the number of connected components.";
    if (new_comp_cnt > comp_cnt) bridges.emplace_back(u, v);
    G.CreateEdge(u, v);
  }
  return bridges;
}
}  // namespace bruteforce

/// Checks if obtained list of bridges is correct.
bool CheckBridges(std::vector<Edge> user,
                  std::vector<std::pair<uint32_t, uint32_t>> correct) {
  std::set<std::pair<uint32_t, uint32_t>> user_bridge_set, correct_bridge_set;
  for (auto &p : correct) {
    correct_bridge_set.insert(p);
    correct_bridge_set.insert({p.second, p.first});
  }
  for (auto &edge : user) {
    user_bridge_set.insert({edge.from, edge.to});
    user_bridge_set.insert({edge.to, edge.from});
  }
  if (correct_bridge_set.size() != user_bridge_set.size()) {
    LOG(WARNING) << "The algorithm found " << user_bridge_set.size()
                 << " bridges, but the correct value is "
                 << correct_bridge_set.size();
  }
  return user_bridge_set == correct_bridge_set;
}

TEST(Bridges, EmptyGraph) {
  Graph G = BuildGraph(0, {});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, SingleNode) {
  Graph G = BuildGraph(1, {});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, DisconnectedNodes) {
  Graph G = BuildGraph(100, {});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, Cycle) {
  uint32_t n = 100;
  std::vector<std::pair<uint32_t, uint32_t>> E;
  for (int i = 0; i < n; ++i) {
    E.emplace_back(i, (i + 1) % n);
  }
  Graph G = BuildGraph(n, E);
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {}));
}

TEST(Bridges, SmallTree) {
  //    (4)
  //   /   \
  // (2)   (1)
  //  |   /   \
  // (0)(3)   (5)
  Graph G = BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}}));
}

TEST(Bridges, RandomTree) {
  Graph G = GenRandomTree(10000);
  std::vector<std::pair<uint32_t, uint32_t>> edges;

  for (const Edge &edge : G.Edges()) edges.emplace_back(edge.from, edge.to);

  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, edges));
}

TEST(Bridges, HandmadeConnectedGraph1) {
  //    (1)--(3)--(7)
  //   / |    |
  // (0) |   (4)--(5)
  //   \ |     \  /
  //    (2)     (6)
  Graph G = BuildGraph(
      8,
      {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 4}, {3, 7}, {4, 5}, {4, 6}, {5, 6}});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {{1, 3}, {3, 4}, {3, 7}}));
}

TEST(Bridges, HandmadeConnectedGraph2) {
  //    (1)--(3)--(7)     (10)     (14)
  //   / |    |    |     /    \     |
  // (0) |   (4)--(5)--(9)   (12)--(13)
  //   \ |     \  / \    \   /
  //    (2)     (6) (8)   (11)
  Graph G = BuildGraph(15, {{0, 1},
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
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(
      CheckBridges(bridges, {{1, 3}, {5, 8}, {5, 9}, {12, 13}, {13, 14}}));
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
  Graph G = BuildGraph(
      26,
      {{0, 1},   {0, 2},   {1, 4},   {1, 3},   {3, 5},   {4, 5},   {2, 6},
       {2, 7},   {6, 7},   {8, 9},   {10, 11}, {11, 12}, {12, 13}, {13, 14},
       {10, 14}, {10, 15}, {15, 16}, {16, 17}, {15, 17}, {13, 18}, {18, 19},
       {18, 21}, {18, 20}, {21, 25}, {20, 22}, {22, 23}, {23, 24}, {22, 24}});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_TRUE(CheckBridges(bridges, {{0, 1},
                                     {0, 2},
                                     {8, 9},
                                     {10, 15},
                                     {13, 18},
                                     {18, 19},
                                     {18, 20},
                                     {18, 21},
                                     {21, 25},
                                     {20, 22}}));
}

TEST(Bridges, SimpleNeighborCycle) {
  Graph G = BuildGraph(2, {{0, 1}, {1, 0}});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_EQ(0, bridges.size());
}

TEST(Bridges, NeighborCycle) {
  Graph G =
      BuildGraph(6, {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {3, 4}, {3, 4}, {4, 5}});
  auto bridges = algorithms::GetBridges(G);
  ASSERT_EQ(1, bridges.size());
  ASSERT_EQ(bridges[0].id, 6);
}

TEST(Bridges, Random100) {
  for (int t = 0; t < 100; ++t) {
    auto G = GenRandomGraph(10, 20);
    auto algo_bridges = algorithms::GetBridges(G);
    auto bf_bridges = bruteforce::GetBridges(G);
    ASSERT_TRUE(CheckBridges(algo_bridges, bf_bridges));
  }
}

TEST(Bridges, Performance) {
  auto G = GenRandomGraph(100000, 250000);
  Timer timer;
  auto bridges = algorithms::GetBridges(G);
  auto time_elapsed = timer.Elapsed();
  ASSERT_TRUE(timer.Elapsed() < std::chrono::seconds(100));
}
