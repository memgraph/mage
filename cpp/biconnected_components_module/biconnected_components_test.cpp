#include <chrono>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>
#include <mg_graph.hpp>

#include "algorithm/biconnected_components.hpp"

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

bool CheckBCC(std::vector<std::vector<mg_graph::Edge<>>> user,
              std::vector<std::vector<std::pair<uint64_t, uint64_t>>> correct) {
  std::vector<std::set<std::pair<uint64_t, uint64_t>>> user_bcc, correct_bcc;

  for (auto &bcc : correct) {
    correct_bcc.push_back({});
    for (auto &p : bcc) {
      correct_bcc.back().insert({p.first, p.second});
      correct_bcc.back().insert({p.second, p.first});
    }
  }

  for (auto &bcc : user) {
    user_bcc.push_back({});
    for (const auto &edge : bcc) {
      user_bcc.back().insert({edge.from, edge.to});
      user_bcc.back().insert({edge.to, edge.from});
    }
  }

  std::sort(correct_bcc.begin(), correct_bcc.end());
  std::sort(user_bcc.begin(), user_bcc.end());

  return user_bcc == correct_bcc;
}

TEST(BCC, EmptyGraph) {
  auto *G = BuildGraph(0, {});
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {}));
}

TEST(BCC, SingleNode) {
  auto *G = BuildGraph(1, {});
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {}));
}

TEST(BCC, DisconnectedNodes) {
  auto *G = BuildGraph(100, {});
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {}));
}

TEST(BCC, Cycle) {
  uint64_t n = 100;
  std::vector<std::pair<uint64_t, uint64_t>> E;
  for (uint64_t i = 0; i < n; ++i) {
    E.emplace_back(i, (i + 1) % n);
  }
  auto *G = BuildGraph(n, E);
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {E}));
}

///    (4)
///   /   \
/// (2)   (1)
///  |   /   \
/// (0)(3)   (5)
TEST(BCC, SmallTree) {
  auto *G = BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}});
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{2, 4}}, {{1, 4}}, {{0, 2}}, {{1, 3}}, {{1, 5}}}));
}

TEST(BCC, RandomTree) {
  auto *G = GenRandomTree(10000);
  std::vector<std::vector<std::pair<uint64_t, uint64_t>>> correct_bcc;
  for (const auto &edge : G->Edges()) {
    if (edge.from < edge.to) correct_bcc.push_back({{edge.from, edge.to}});
  }
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, correct_bcc));
}

///    (1)--(3)--(7)
///   / |    |
/// (0) |   (4)--(5)
///   \ |     \  /
///    (2)     (6)
TEST(BCC, HandmadeConnectedGraph1) {
  auto *G = BuildGraph(8, {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 4}, {3, 7}, {4, 5}, {4, 6}, {5, 6}});
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}, {1, 2}, {2, 0}}, {{1, 3}}, {{3, 7}}, {{3, 4}}, {{4, 5}, {5, 6}, {4, 6}}}));
}

///    (1)--(3)--(7)     (10)     (14)
///   / |    |    |     /    \     |
/// (0) |   (4)--(5)--(9)   (12)--(13)
///   \ |     \  / \    \   /
///    (2)     (6) (8)   (11)
TEST(BCC, HandmadeConnectedGraph2) {
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
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}, {1, 2}, {2, 0}},
                             {{1, 3}},
                             {{3, 7}, {7, 5}, {5, 4}, {3, 4}, {4, 6}, {5, 6}},
                             {{5, 8}},
                             {{5, 9}},
                             {{9, 10}, {10, 12}, {11, 12}, {9, 11}},
                             {{12, 13}},
                             {{13, 14}}}));
}

///    (4)--(5)                   (12)         (19)         (23)
///     |    |                   /    \        /           /   \
///    (1)--(3)               (11)    (13)--(18)--(20)--(22)--(24)
///   /                        |       |       \
/// (0)             (8)--(9)  (10)----(14)      (21)--(25)
///   \                        |
///    (2)--(6)               (15)----(16)
///      \  /                    \    /
///       (7)                     (17)
TEST(BCC, HandmadeDisconnectedGraph) {
  auto *G = BuildGraph(
      26, {{0, 1},   {0, 2},   {1, 4},   {1, 3},   {3, 5},   {4, 5},   {2, 6},   {2, 7},   {6, 7},   {8, 9},
           {10, 11}, {11, 12}, {12, 13}, {13, 14}, {10, 14}, {10, 15}, {15, 16}, {16, 17}, {15, 17}, {13, 18},
           {18, 19}, {18, 21}, {18, 20}, {21, 25}, {20, 22}, {22, 23}, {23, 24}, {22, 24}});
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
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

///     (1)--(2)--(5)--(6)        (13)
///    / |\  /|    |  / | \      /    \
/// (0)  | \/ |    | /  | (7)--(10)--(11)
///    \ | /\ |    |/   |/       \    /
///     (3)--(4)  (9)--(8)        (12)
TEST(BCC, HandmadeCrossEdge) {
  auto *G = BuildGraph(
      14, {{0, 1}, {0, 3}, {1, 3}, {1, 4}, {1, 2}, {3, 4},  {2, 4},   {2, 3},   {2, 5},   {5, 9},   {9, 8},
           {8, 7}, {6, 7}, {6, 8}, {9, 6}, {5, 6}, {7, 10}, {10, 13}, {10, 11}, {10, 12}, {13, 11}, {11, 12}});
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}, {1, 3}, {0, 3}, {1, 4}, {1, 2}, {2, 4}, {3, 4}, {2, 3}},
                             {{2, 5}},
                             {{5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 5}, {9, 6}, {6, 8}},
                             {{7, 10}},
                             {{10, 11}, {10, 12}, {10, 13}, {11, 12}, {11, 13}}}));
}

/// (0)     (3)
///  | \   / |
///  |  (2)  |
///  | /   \ |
/// (1)     (4)   (9)
///        /   \ /  \
///      (5)---(6)--(10)
///        \   /
///         (8)
TEST(BCC, HandmadeArticulationPoint) {
  auto *G = BuildGraph(11, {{0, 1},
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
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  ASSERT_TRUE(CheckBCC(BCC, {{{0, 1}, {1, 2}, {2, 0}},
                             {{2, 3}, {2, 4}, {3, 4}},
                             {{4, 5}, {4, 6}, {5, 6}, {5, 8}, {6, 8}},
                             {{6, 9}, {9, 10}, {6, 10}}}));
}

TEST(BCC, Performance) {
  auto G = GenRandomGraph(10000, 25000);
  Timer timer;
  auto BCC = bcc_algorithm::GetBiconnectedComponents(G);
  auto time_elapsed = timer.Elapsed();
  ASSERT_TRUE(time_elapsed < std::chrono::seconds(1));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}