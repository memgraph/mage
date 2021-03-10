#include <glog/logging.h>
#include <gtest/gtest.h>

#include <algorithm>

#include "algorithms/algorithms.hpp"
#include "utils.hpp"

using namespace graphdata;

bool CheckCycles(std::vector<std::vector<Node>> user,
                 std::vector<std::vector<uint32_t>> correct) {
  // normalize cycles
  for (auto &cycle : correct) {
    std::rotate(cycle.begin(), std::min_element(cycle.begin(), cycle.end()),
                cycle.end());
  }

  std::vector<std::vector<uint32_t>> user_cycles;
  for (const auto &cycle : user) {
    std::vector<uint32_t> user_cycle;
    for (const auto &node : cycle) user_cycle.push_back(node.id);
    std::rotate(user_cycle.begin(),
                std::min_element(user_cycle.begin(), user_cycle.end()),
                user_cycle.end());
    user_cycles.push_back(user_cycle);
  }

  if (user_cycles.size() != correct.size()) {
    LOG(WARNING) << "The algorithm found " << user_cycles.size()
                 << " simple cycles, but the correct value is "
                 << correct.size();
    return false;
  }

  for (int i = 0; i < static_cast<int>(user_cycles.size()); ++i) {
    user_cycles[i].push_back(*user_cycles[i].begin());
    correct[i].push_back(*correct[i].begin());
    if (user_cycles[i][1] > user_cycles[i][user_cycles[i].size() - 2])
      std::reverse(user_cycles[i].begin(), user_cycles[i].end());
    if (correct[i][1] > correct[i][correct[i].size() - 2])
      std::reverse(correct[i].begin(), correct[i].end());
  }

  std::sort(correct.begin(), correct.end());
  std::sort(user_cycles.begin(), user_cycles.end());

  return user_cycles == correct;
}

TEST(Cycles, EmptyGraph) {
  Graph G = BuildGraph(0, {});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {}));
}

TEST(Cycles, SingleNode) {
  Graph G = BuildGraph(1, {});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {}));
}

TEST(Cycles, DisconnectedNodes) {
  Graph G = BuildGraph(100, {});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {}));
}

TEST(Cycles, SmallTree) {
  //    (4)
  //   /   \
  // (2)   (1)
  //  |   /   \
  // (0)(3)   (5)
  Graph G = BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {}));
}

TEST(Cycles, RandomTree) {
  Graph G = GenRandomTree(10000);
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {}));
}

TEST(Cycles, Triangle) {
  Graph G = BuildGraph(3, {{0, 1}, {1, 2}, {0, 2}});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {{0, 1, 2}}));
}

TEST(Cycles, BigCycle) {
  int nodes = 1000;
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (int i = 1; i < nodes; ++i) {
    edges.emplace_back(i - 1, i);
  }
  edges.emplace_back(0, nodes - 1);

  Graph G = BuildGraph(nodes, edges);

  auto cycles = algorithms::GetCycles(G);

  std::vector<std::vector<uint32_t>> correct;
  correct.push_back({});
  for (int i = 0; i < nodes; ++i) correct.back().push_back(i);

  ASSERT_TRUE(CheckCycles(cycles, correct));
}

TEST(Cycles, HandmadeConnectedGraph1) {
  //    (1)--(3)--(7)
  //   / |    |
  // (0) |   (4)--(5)
  //   \ |     \  /
  //    (2)     (6)
  Graph G = BuildGraph(
      8,
      {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 4}, {3, 7}, {4, 5}, {4, 6}, {5, 6}});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {{0, 1, 2}, {4, 5, 6}}));
}

TEST(Cycles, HandmadeConnectedGraph2) {
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
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(
      cycles,
      {{0, 1, 2}, {3, 7, 5, 4}, {4, 5, 6}, {3, 7, 5, 6, 4}, {9, 10, 12, 11}}));
}

TEST(Cycles, DisconnectedCycles) {
  //    (1)  (3)---(4)
  //   / |    |     |
  // (0) |    |     |
  //   \ |    |     |
  //    (2)  (5)---(6)
  Graph G =
      BuildGraph(7, {{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 6}, {5, 6}});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {{0, 1, 2}, {3, 4, 6, 5}}));
}

TEST(Cycles, HandmadeDisconnectedGraph) {
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
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {{1, 3, 5, 4},
                                   {2, 6, 7},
                                   {10, 11, 12, 13, 14},
                                   {15, 16, 17},
                                   {22, 23, 24}}));
}

TEST(Cycles, HandmadeArticulationPoint) {
  // (0)     (3)
  //  | \   / |
  //  |  (2)  |
  //  | /   \ |
  // (1)     (4)   (9)
  //        /   \ /  \
  //      (5)---(6)--(10)
  //        \   /
  //         (8)
  Graph G = BuildGraph(11, {{0, 1},
                            {0, 2},
                            {1, 2},
                            {2, 3},
                            {2, 4},
                            {3, 4},
                            {4, 5},
                            {5, 8},
                            {6, 8},
                            {4, 6},
                            {5, 6},
                            {6, 9},
                            {9, 10},
                            {6, 10}});
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(
      cycles,
      {{0, 1, 2}, {2, 3, 4}, {4, 5, 6}, {5, 6, 8}, {4, 5, 8, 6}, {6, 9, 10}}));
}

TEST(Cycles, HandmadeComplexCycle) {
  //       (2)--(3)
  //      /        \
  //    (1)---------(4)
  //   / |            \
  // (0)-+------------(5)
  //  |  |             |
  // (11)+------------(6)
  //   \ |            /
  //    (10)        (7)
  //      \        /
  //       (9)--(8)
  int nodes = 12;
  std::vector<std::pair<uint32_t, uint32_t>> edges;
  for (int i = 1; i < nodes; ++i) {
    edges.emplace_back(i - 1, i);
  }
  edges.emplace_back(0, nodes - 1);

  edges.emplace_back(1, 4);
  edges.emplace_back(0, 5);
  edges.emplace_back(6, 11);
  edges.emplace_back(1, 10);

  Graph G = BuildGraph(nodes, edges);
  auto cycles = algorithms::GetCycles(G);
  ASSERT_TRUE(CheckCycles(cycles, {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                   {1, 2, 3, 4},
                                   {0, 1, 4, 5},
                                   {11, 0, 5, 6},
                                   {11, 6, 7, 8, 9, 10},
                                   {0, 1, 2, 3, 4, 5},
                                   {11, 0, 1, 4, 5, 6},
                                   {9, 10, 11, 0, 5, 6, 7, 8},
                                   {11, 0, 1, 2, 3, 4, 5, 6},
                                   {9, 10, 11, 0, 1, 4, 5, 6, 7, 8},
                                   {0, 1, 10, 11},
                                   {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                   {10, 1, 4, 5, 0, 11},
                                   {10, 1, 4, 5, 6, 11},
                                   {10, 1, 0, 5, 6, 7, 8, 9},
                                   {10, 1, 0, 11, 6, 7, 8, 9},
                                   {10, 1, 4, 5, 0, 11, 6, 7, 8, 9},
                                   {10, 1, 0, 5, 6, 11},
                                   {10, 1, 2, 3, 4, 5, 0, 11},
                                   {10, 1, 2, 3, 4, 5, 0, 11, 6, 7, 8, 9},
                                   {10, 1, 2, 3, 4, 5, 6, 11},
                                   {10, 1, 4, 5, 6, 7, 8, 9}}));
}

TEST(Cycles, CompleteGraphCounts) {
  std::vector<size_t> oracle = {0, 0, 1, 7, 37, 197, 1172};
  int nodes = 1;
  for (int correct : oracle) {
    Graph G = GenCompleteGraph(nodes);
    auto cycles = algorithms::GetCycles(G);
    ASSERT_EQ(cycles.size(), correct);
    ++nodes;
  }
}

TEST(Cycles, NeighbouringCycles) {
  Graph G;
  uint32_t node_1 = G.CreateNode();
  uint32_t node_2 = G.CreateNode();
  uint32_t node_3 = G.CreateNode();
  uint32_t node_4 = G.CreateNode();

  G.CreateEdge(node_1, node_2);
  G.CreateEdge(node_1, node_2);

  G.CreateEdge(node_1, node_4);

  G.CreateEdge(node_2, node_3);
  G.CreateEdge(node_2, node_3);
  G.CreateEdge(node_2, node_3);

  const auto &pairs = algorithms::GetNeighbourCycles(G);

  std::set<std::pair<uint32_t, uint32_t>> sol = {{1, 2}, {2, 3}};
  ASSERT_EQ(sol.size(), pairs.size());
  for (const auto &node_pair : pairs) {
    std::pair<uint32_t, uint32_t> pair = {node_pair.first.id + 1,
                                          node_pair.second.id + 1};
    ASSERT_TRUE(sol.find(pair) != sol.end());
  }
}
