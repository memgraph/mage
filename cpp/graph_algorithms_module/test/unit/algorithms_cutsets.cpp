#include <glog/logging.h>
#include <gtest/gtest.h>

#include "algorithms/algorithms.hpp"
#include "utils.hpp"

using namespace graphdata;

using CutsetType = std::vector<std::vector<std::pair<uint32_t, uint32_t>>>;

bool CheckCutsets(std::vector<std::vector<graphdata::Edge>> user,
                  CutsetType correct) {
  std::vector<std::set<std::pair<uint32_t, uint32_t>>> user_cutsets,
      correct_cutsets;

  for (auto &bcc : correct) {
    correct_cutsets.push_back({});
    for (auto &p : bcc) {
      correct_cutsets.back().insert({p.first, p.second});
      correct_cutsets.back().insert({p.second, p.first});
    }
  }

  for (auto &bcc : user) {
    user_cutsets.push_back({});
    for (const graphdata::Edge &edge : bcc) {
      user_cutsets.back().insert({edge.from, edge.to});
      user_cutsets.back().insert({edge.to, edge.from});
    }
  }

  std::sort(correct_cutsets.begin(), correct_cutsets.end());
  std::sort(user_cutsets.begin(), user_cutsets.end());

  if (user_cutsets.size() != correct_cutsets.size()) {
    LOG(WARNING) << "The algorithm found " << user_cutsets.size()
                 << " cutsets, but the correct value is "
                 << correct_cutsets.size();
  }

  return user_cutsets == correct_cutsets;
}

bool CheckCutsetsgraphdata(std::vector<std::vector<graphdata::Edge>> A,
                           std::vector<std::vector<graphdata::Edge>> B) {
  std::vector<std::set<std::pair<uint32_t, uint32_t>>> A_set, B_set;

  for (auto &bcc : A) {
    A_set.push_back({});
    for (const graphdata::Edge &edge : bcc) {
      A_set.back().insert({edge.from, edge.to});
      A_set.back().insert({edge.to, edge.from});
    }
  }

  for (auto &bcc : B) {
    B_set.push_back({});
    for (const graphdata::Edge &edge : bcc) {
      B_set.back().insert({edge.from, edge.to});
      B_set.back().insert({edge.to, edge.from});
    }
  }

  std::sort(A_set.begin(), A_set.end());
  std::sort(B_set.begin(), B_set.end());

  return A_set == B_set;
}

TEST(Cutsets, EmptyGraph) {
  Graph G = BuildGraph(0, {});
  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);
  ASSERT_TRUE(CheckCutsets(cutsets, {}));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, {}));
}

TEST(Cutsets, SingleNode) {
  Graph G = BuildGraph(1, {});
  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms::GetCutsets(G);
  ASSERT_TRUE(CheckCutsets(cutsets, {}));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, {}));
}

TEST(Cutsets, DisconnectedNodes) {
  Graph G = BuildGraph(100, {});
  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);
  ASSERT_TRUE(CheckCutsets(cutsets, {}));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, {}));
}

TEST(Cutsets, SmallTree) {
  //    (4)
  //   /   \
  // (2)   (1)
  //  |   /   \
  // (0)(3)   (5)
  Graph G = BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}});

  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);

  CutsetType correct = {{{2, 4}}, {{1, 4}}, {{0, 2}}, {{1, 3}}, {{1, 5}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, CompleteGraphs) {
  for (int nodes = 1; nodes <= 13; ++nodes) {
    Graph G = GenCompleteGraph(nodes);
    auto cutsets = algorithms::GetCutsets(G);
    auto cutsets_bf = algorithms_bf::GetCutsets(G);
    ASSERT_EQ(cutsets.size(), (1 << (nodes - 1)) - 1);
    ASSERT_EQ(cutsets_bf.size(), (1 << (nodes - 1)) - 1);
    ASSERT_TRUE(CheckCutsetsgraphdata(cutsets, cutsets_bf));
  }
}

TEST(Cutsets, Cycles) {
  for (int nodes = 1; nodes < 15; ++nodes) {
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (int i = 1; i < nodes; ++i) {
      edges.emplace_back(i - 1, i);
    }
    edges.emplace_back(0, nodes - 1);

    Graph G = BuildGraph(nodes, edges);
    auto cutsets = algorithms::GetCutsets(G);
    auto cutsets_bf = algorithms_bf::GetCutsets(G);

    CutsetType correct;
    for (int i = 0; i < nodes; ++i) {
      for (int j = i + 1; j < nodes; ++j) {
        std::vector<std::pair<uint32_t, uint32_t>> cutset;
        cutset.emplace_back(i, (i + 1) % nodes);
        cutset.emplace_back(j, (j + 1) % nodes);
        correct.push_back(cutset);
      }
    }

    ASSERT_TRUE(CheckCutsets(cutsets, correct));
    ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
  }
}

TEST(Cutsets, DisconnectedCycles) {
  //    (1)  (3)---(4)
  //   / |    |     |
  // (0) |    |     |
  //   \ |    |     |
  //    (2)  (5)---(6)
  Graph G =
      BuildGraph(7, {{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 6}, {5, 6}});

  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);

  CutsetType correct = {{{0, 1}, {1, 2}}, {{0, 1}, {0, 2}}, {{0, 2}, {2, 1}},
                        {{3, 4}, {4, 6}}, {{4, 6}, {6, 5}}, {{6, 5}, {5, 3}},
                        {{5, 3}, {3, 4}}, {{3, 4}, {5, 6}}, {{3, 5}, {4, 6}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, HandmadeConnectedGraph1) {
  //    (1)--(3)--(7)
  //   / |    |
  // (0) |   (4)--(5)
  //   \ |     \  /
  //    (2)     (6)
  Graph G = BuildGraph(
      8,
      {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 4}, {3, 7}, {4, 5}, {4, 6}, {5, 6}});

  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);

  CutsetType correct = {// bridges
                        {{1, 3}},
                        {{3, 4}},
                        {{3, 7}},

                        // two from first cycle
                        {{0, 1}, {0, 2}},
                        {{0, 1}, {1, 2}},
                        {{0, 2}, {1, 2}},

                        // two from second cycle
                        {{4, 5}, {5, 6}},
                        {{4, 6}, {4, 5}},
                        {{6, 4}, {6, 5}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, HandmadeConnectedGraph2) {
  // (0)--(1)
  //  | \  |
  //  |  \ |
  // (3)--(2)
  Graph G = BuildGraph(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}});

  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);

  CutsetType correct = {{{0, 3}, {3, 2}},         {{0, 1}, {1, 2}},
                        {{0, 1}, {0, 2}, {0, 3}}, {{2, 3}, {2, 0}, {2, 1}},
                        {{0, 1}, {3, 2}, {0, 2}}, {{0, 3}, {0, 2}, {1, 2}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, HandmadeConnectedGraph3) {
  // (0)--(1)
  //  |\   | \
  //  | \  | (2)
  //  |  \ | /
  // (4)--(3)
  Graph G =
      BuildGraph(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 3}, {1, 3}});

  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);

  CutsetType correct = {{{0, 1}, {0, 3}, {0, 4}},
                        {{1, 0}, {1, 3}, {1, 2}},
                        {{2, 1}, {2, 3}},
                        {{3, 4}, {3, 0}, {3, 1}, {3, 2}},
                        {{4, 0}, {4, 3}},
                        {{0, 4}, {0, 3}, {1, 3}, {1, 2}},
                        {{0, 1}, {1, 3}, {2, 3}},
                        {{1, 2}, {1, 3}, {3, 4}, {3, 0}},
                        {{4, 0}, {3, 0}, {3, 1}, {3, 2}},
                        {{0, 1}, {0, 3}, {4, 3}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, HandmadeDisconnectedGraph) {
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

  auto cutsets = algorithms::GetCutsets(G);
  auto cutsets_bf = algorithms_bf::GetCutsets(G);

  CutsetType correct = {{{0, 1}},  // first component
                        {{0, 2}},
                        {{1, 4}, {4, 5}},
                        {{4, 5}, {5, 3}},
                        {{5, 3}, {3, 1}},
                        {{3, 1}, {1, 4}},
                        {{4, 5}, {1, 3}},
                        {{1, 4}, {3, 5}},
                        {{2, 6}, {6, 7}},
                        {{6, 7}, {7, 2}},
                        {{7, 2}, {2, 6}},
                        {{8, 9}},    // second component
                        {{10, 15}},  // third component
                        {{13, 18}},
                        {{18, 19}},
                        {{18, 20}},
                        {{18, 21}},
                        {{20, 22}},
                        {{21, 25}},
                        {{22, 23}, {23, 24}},
                        {{23, 24}, {24, 22}},
                        {{24, 22}, {22, 23}},
                        {{15, 16}, {16, 17}},
                        {{16, 17}, {17, 15}},
                        {{17, 15}, {15, 16}},
                        {{10, 11}, {11, 12}},
                        {{11, 12}, {12, 13}},
                        {{12, 13}, {13, 14}},
                        {{13, 14}, {14, 10}},
                        {{14, 10}, {10, 11}},
                        {{10, 11}, {12, 13}},
                        {{11, 12}, {13, 14}},
                        {{12, 13}, {14, 10}},
                        {{10, 11}, {13, 14}},
                        {{11, 12}, {14, 10}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, Random50) {
  for (int t = 0; t < 50; ++t) {
    auto G = GenRandomGraph(13, 30);  // cca 1200 cutsets per test case.

    Timer algo_timer;
    auto cutsets = algorithms::GetCutsets(G);
    LOG(INFO) << "Paper algo runtime: "
              << std::chrono::duration<double, std::milli>(algo_timer.Elapsed())
                     .count();

    Timer bf_timer;
    auto cutsets_bf = algorithms_bf::GetCutsets(G);
    LOG(INFO) << "Naive algo runtime: "
              << std::chrono::duration<double, std::milli>(bf_timer.Elapsed())
                     .count();

    LOG(INFO) << "Number of cutsets:" << cutsets_bf.size();
    ASSERT_TRUE(CheckCutsetsgraphdata(cutsets_bf, cutsets));
  }
}
