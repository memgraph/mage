#include <glog/logging.h>
#include <gtest/gtest.h>

#include <set>
#include <vector>

#include "algorithms/algorithms.hpp"
#include "utils.hpp"

bool CheckBCC(std::vector<std::vector<graphdata::Edge>> user,
              std::vector<std::vector<std::pair<uint32_t, uint32_t>>> correct) {
  std::vector<std::set<std::pair<uint32_t, uint32_t>>> user_BCC, correct_BCC;

  for (auto &bcc : correct) {
    correct_BCC.push_back({});
    for (auto &p : bcc) {
      correct_BCC.back().insert({p.first, p.second});
      correct_BCC.back().insert({p.second, p.first});
    }
  }

  for (auto &bcc : user) {
    user_BCC.push_back({});
    for (const graphdata::Edge &edge : bcc) {
      user_BCC.back().insert({edge.from, edge.to});
      user_BCC.back().insert({edge.to, edge.from});
    }
  }

  std::sort(correct_BCC.begin(), correct_BCC.end());
  std::sort(user_BCC.begin(), user_BCC.end());

  if (user_BCC.size() != correct_BCC.size()) {
    LOG(WARNING) << "The algorithm found " << user_BCC.size()
                 << " biconnected components, but the correct value is "
                 << correct_BCC.size();
  }

  return user_BCC == correct_BCC;
}

TEST(BCC, EmptyGraph) {
  graphdata::Graph G = BuildGraph(0, {});
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {}));
}

TEST(BCC, SingleNode) {
  graphdata::Graph G = BuildGraph(1, {});
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {}));
}

TEST(BCC, DisconnectedNodes) {
  graphdata::Graph G = BuildGraph(100, {});
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {}));
}

TEST(BCC, Cycle) {
  uint32_t n = 100;
  std::vector<std::pair<uint32_t, uint32_t>> E;
  for (int i = 0; i < n; ++i) {
    E.emplace_back(i, (i + 1) % n);
  }
  graphdata::Graph G = BuildGraph(n, E);
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {E}));
}

TEST(BCC, SmallTree) {
  //    (4)
  //   /   \
  // (2)   (1)
  //  |   /   \
  // (0)(3)   (5)
  graphdata::Graph G = BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}});
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(
      CheckBCC(BCC, {{{2, 4}}, {{1, 4}}, {{0, 2}}, {{1, 3}}, {{1, 5}}}));
}

TEST(BCC, RandomTree) {
  graphdata::Graph G = GenRandomTree(10000);
  std::vector<std::vector<std::pair<uint32_t, uint32_t>>> correct_bcc;
  for (const graphdata::Edge &edge : G.Edges()) {
    if (edge.from < edge.to) correct_bcc.push_back({{edge.from, edge.to}});
  }
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, correct_bcc));
}

TEST(BCC, HandmadeConnectedGraph1) {
  //    (1)--(3)--(7)
  //   / |    |
  // (0) |   (4)--(5)
  //   \ |     \  /
  //    (2)     (6)
  graphdata::Graph G = BuildGraph(
      8,
      {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 4}, {3, 7}, {4, 5}, {4, 6}, {5, 6}});
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}, {1, 2}, {2, 0}},
                             {{1, 3}},
                             {{3, 7}},
                             {{3, 4}},
                             {{4, 5}, {5, 6}, {4, 6}}}));
}

TEST(BCC, HandmadeConnectedGraph2) {
  //    (1)--(3)--(7)     (10)     (14)
  //   / |    |    |     /    \     |
  // (0) |   (4)--(5)--(9)   (12)--(13)
  //   \ |     \  / \    \   /
  //    (2)     (6) (8)   (11)
  graphdata::Graph G = BuildGraph(15, {{0, 1},
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
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}, {1, 2}, {2, 0}},
                             {{1, 3}},
                             {{3, 7}, {7, 5}, {5, 4}, {3, 4}, {4, 6}, {5, 6}},
                             {{5, 8}},
                             {{5, 9}},
                             {{9, 10}, {10, 12}, {11, 12}, {9, 11}},
                             {{12, 13}},
                             {{13, 14}}}));
}

TEST(BCC, HandmadeDisconnectedGraph) {
  //    (4)--(5)                   (12)         (19)         (23)
  //     |    |                   /    \        /           /   \
  //    (1)--(3)               (11)    (13)--(18)--(20)--(22)--(24)
  //   /                        |       |       \
  // (0)             (8)--(9)  (10)----(14)      (21)--(25)
  //   \                        |
  //    (2)--(6)               (15)----(16)
  //      \  /                    \    /
  //       (7)                     (17)
  graphdata::Graph G = BuildGraph(
      26,
      {{0, 1},   {0, 2},   {1, 4},   {1, 3},   {3, 5},   {4, 5},   {2, 6},
       {2, 7},   {6, 7},   {8, 9},   {10, 11}, {11, 12}, {12, 13}, {13, 14},
       {10, 14}, {10, 15}, {15, 16}, {16, 17}, {15, 17}, {13, 18}, {18, 19},
       {18, 21}, {18, 20}, {21, 25}, {20, 22}, {22, 23}, {23, 24}, {22, 24}});
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}},
                             {{0, 2}},
                             {{1, 4}, {1, 3}, {4, 5}, {3, 5}},
                             {{2, 6}, {6, 7}, {2, 7}},
                             {{8, 9}},
                             {{10, 11}, {11, 12}, {12, 13}, {13, 14}, {14, 10}},
                             {{10, 15}},
                             {{15, 16}, {15, 17}, {16, 17}},
                             {{13, 18}},
                             {{18, 19}},
                             {{18, 21}},
                             {{18, 20}},
                             {{21, 25}},
                             {{20, 22}},
                             {{22, 23}, {22, 24}, {23, 24}}}));
}

TEST(BCC, HandmadeCrossEdge) {
  //     (1)--(2)--(5)--(6)        (13)
  //    / |\  /|    |  / | \      /    \
  // (0)  | \/ |    | /  | (7)--(10)--(11)
  //    \ | /\ |    |/   |/       \    /
  //     (3)--(4)  (9)--(8)        (12)
  graphdata::Graph G =
      BuildGraph(14, {{0, 1},   {0, 3},   {1, 3},   {1, 4},  {1, 2},  {3, 4},
                      {2, 4},   {2, 3},   {2, 5},   {5, 9},  {9, 8},  {8, 7},
                      {6, 7},   {6, 8},   {9, 6},   {5, 6},  {7, 10}, {10, 13},
                      {10, 11}, {10, 12}, {13, 11}, {11, 12}});
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(
      BCC, {{{0, 1}, {1, 3}, {0, 3}, {1, 4}, {1, 2}, {2, 4}, {3, 4}, {2, 3}},
            {{2, 5}},
            {{5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 5}, {9, 6}, {6, 8}},
            {{7, 10}},
            {{10, 11}, {10, 12}, {10, 13}, {11, 12}, {11, 13}}}));
}

TEST(BCC, HandmadeArticulationPoint) {
  // (0)     (3)
  //  | \   / |
  //  |  (2)  |
  //  | /   \ |
  // (1)     (4)   (9)
  //        /   \ /  \
  //      (5)---(6)--(10)
  //        \   /
  //         (8)
  graphdata::Graph G = BuildGraph(11, {{0, 1},
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
  auto BCC = algorithms::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}, {1, 2}, {2, 0}},
                             {{2, 3}, {2, 4}, {3, 4}},
                             {{4, 5}, {4, 6}, {5, 6}, {5, 8}, {6, 8}},
                             {{6, 9}, {9, 10}, {6, 10}}}));
}

TEST(BCC, Performance) {
  auto G = GenRandomGraph(100000, 250000);
  Timer timer;
  auto BCC = algorithms::GetBiconnectedComponents(G);
  auto time_elapsed = timer.Elapsed();
  ASSERT_TRUE(timer.Elapsed() < std::chrono::seconds(1));
}
