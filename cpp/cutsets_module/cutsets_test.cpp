#include <gtest/gtest.h>

#include <mg_graph.hpp>

#include "algorithm/cutsets.hpp"

using CutsetType = std::vector<std::vector<std::pair<uint64_t, uint64_t>>>;

namespace cutsets_bf {

void CompCntDfs(uint64_t node_id, const mg_graph::GraphView<> *G, const std::vector<mg_graph::Edge<>> &deleted_edges,
                cutsets_util::NodeState *state) {
  state->visited[node_id] = true;
  for (const auto &neigh : G->Neighbours(node_id)) {
    auto next_id = neigh.node_id;
    if (state->visited[next_id] ||
        std::find_if(deleted_edges.begin(), deleted_edges.end(),
                     [&neigh](const mg_graph::Edge<> &e) { return neigh.edge_id == e.id; }) != deleted_edges.end()) {
      continue;
    }
    cutsets_bf::CompCntDfs(next_id, G, deleted_edges, state);
  }
}

int ComponentCount(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component,
                   const std::vector<mg_graph::Edge<>> &deleted_edges) {
  cutsets_util::NodeState state;
  auto node_size = G->Nodes().size();
  state.visited.resize(node_size, false);

  int component_count = 0;
  for (auto node_id : component) {
    if (state.visited[node_id]) continue;
    ++component_count;
    cutsets_bf::CompCntDfs(node_id, G, deleted_edges, &state);
  }

  return component_count;
}

void FindComponent(uint64_t node_id, const mg_graph::GraphView<> *G, cutsets_util::NodeState *state,
                   std::vector<uint64_t> *component) {
  state->visited[node_id] = true;
  component->emplace_back(node_id);
  for (const auto &neigh : G->Neighbours(node_id)) {
    auto next_id = neigh.node_id;
    if (state->visited[next_id]) continue;
    cutsets_bf::FindComponent(next_id, G, state, component);
  }
}

void SolveComponent(const mg_graph::GraphView<> *G, const std::vector<uint64_t> &component,
                    std::vector<std::vector<mg_graph::Edge<>>> *cutsets) {
  // special case for a single node in connected component.
  if (component.size() == 1) return;

  for (int mask = 1; mask < (1 << component.size()); ++mask) {
    auto complement = (1 << component.size()) - 1 - mask;
    if (mask > complement) continue;  // this prevents doubling of cutsets

    // Find cutset for a given mask.
    std::vector<mg_graph::Edge<>> cutset;

    for (uint64_t i = 0; i < component.size(); ++i) {
      auto node_id = component[i];
      for (const auto &neigh : G->Neighbours(node_id)) {
        auto next_id = neigh.node_id;
        auto pos = std::find(component.begin(), component.end(), next_id);
        if (next_id < node_id || pos == component.end()) continue;
        auto j = pos - component.begin();
        auto color_node = (mask & (1 << i)) > 0;
        auto color_next = (mask & (1 << j)) > 0;
        if (color_node != color_next) cutset.emplace_back(G->GetEdge(neigh.edge_id));
      }
    }

    assert(!cutset.empty() && "Cutset cannot be empty!");
    if (cutsets_bf::ComponentCount(G, component, cutset) == 2) cutsets->emplace_back(cutset);
  }
}

std::vector<std::vector<mg_graph::Edge<>>> GetCutsets(const mg_graph::GraphView<> *G) {
  cutsets_util::NodeState state;
  auto node_size = G->Nodes().size();
  state.visited.resize(node_size, false);

  std::vector<std::vector<mg_graph::Edge<>>> cutsets;

  for (const auto &node : G->Nodes()) {
    if (state.visited[node.id]) continue;
    std::vector<uint64_t> component;
    cutsets_bf::FindComponent(node.id, G, &state, &component);
    cutsets_bf::SolveComponent(G, component, &cutsets);
  }

  return cutsets;
}

}  // namespace cutsets_bf

/// Builds the graph from a given number of nodes and a list of edges.
/// Nodes should be 0-indexed and each edge should be provided in both
/// directions.
inline mg_graph::Graph<> *BuildGraph(uint64_t nodes, std::vector<std::pair<uint64_t, uint64_t>> edges) {
  auto *G = new mg_graph::Graph<>();
  for (uint64_t i = 0; i < nodes; ++i) G->CreateNode(i);
  for (auto &p : edges) G->CreateEdge(p.first, p.second);

  return G;
}

/// Generates a complete graph with a given number of nodes.
inline mg_graph::Graph<> *GenCompleteGraph(uint64_t nodes) {
  std::vector<std::pair<uint64_t, uint64_t>> edges;
  for (uint64_t i = 0; i < nodes; ++i) {
    for (uint64_t j = i + 1; j < nodes; ++j) {
      edges.emplace_back(i, j);
    }
  }
  return BuildGraph(nodes, edges);
}

bool CheckCutsets(std::vector<std::vector<mg_graph::Edge<>>> user, CutsetType correct) {
  std::vector<std::set<std::pair<uint64_t, uint64_t>>> user_cutsets, correct_cutsets;

  for (auto &bcc : correct) {
    correct_cutsets.push_back({});
    for (auto &p : bcc) {
      correct_cutsets.back().insert({p.first, p.second});
      correct_cutsets.back().insert({p.second, p.first});
    }
  }

  for (auto &bcc : user) {
    user_cutsets.push_back({});
    for (const auto &edge : bcc) {
      user_cutsets.back().insert({edge.from, edge.to});
      user_cutsets.back().insert({edge.to, edge.from});
    }
  }

  std::sort(correct_cutsets.begin(), correct_cutsets.end());
  std::sort(user_cutsets.begin(), user_cutsets.end());

  // if (user_cutsets.size() != correct_cutsets.size()) {
  //   LOG(WARNING) << "The algorithm found " << user_cutsets.size() << " cutsets, but the correct value is "
  //                << correct_cutsets.size();
  // }

  return user_cutsets == correct_cutsets;
}

bool CheckCutsetsmg_graph(std::vector<std::vector<mg_graph::Edge<>>> A, std::vector<std::vector<mg_graph::Edge<>>> B) {
  std::vector<std::set<std::pair<uint64_t, uint64_t>>> A_set, B_set;

  for (auto &bcc : A) {
    A_set.push_back({});
    for (const auto &edge : bcc) {
      A_set.back().insert({edge.from, edge.to});
      A_set.back().insert({edge.to, edge.from});
    }
  }

  for (auto &bcc : B) {
    B_set.push_back({});
    for (const auto &edge : bcc) {
      B_set.back().insert({edge.from, edge.to});
      B_set.back().insert({edge.to, edge.from});
    }
  }

  std::sort(A_set.begin(), A_set.end());
  std::sort(B_set.begin(), B_set.end());

  return A_set == B_set;
}

TEST(Cutsets, EmptyGraph) {
  auto *G = BuildGraph(0, {});
  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);
  ASSERT_TRUE(CheckCutsets(cutsets, {}));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, {}));
}

TEST(Cutsets, SingleNode) {
  auto *G = BuildGraph(1, {});
  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);
  ASSERT_TRUE(CheckCutsets(cutsets, {}));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, {}));
}

TEST(Cutsets, DisconnectedNodes) {
  auto *G = BuildGraph(100, {});
  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);
  ASSERT_TRUE(CheckCutsets(cutsets, {}));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, {}));
}

TEST(Cutsets, SmallTree) {
  //    (4)
  //   /   \
  // (2)   (1)
  //  |   /   \
  // (0)(3)   (5)
  auto *G = BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}});

  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);

  CutsetType correct = {{{2, 4}}, {{1, 4}}, {{0, 2}}, {{1, 3}}, {{1, 5}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, CompleteGraphs) {
  for (int nodes = 1; nodes <= 13; ++nodes) {
    auto *G = GenCompleteGraph(nodes);
    auto cutsets = cutsets_alg::GetCutsets(G);
    auto cutsets_bf = cutsets_bf::GetCutsets(G);
    ASSERT_EQ(cutsets.size(), (1 << (nodes - 1)) - 1);
    ASSERT_EQ(cutsets_bf.size(), (1 << (nodes - 1)) - 1);
    ASSERT_TRUE(CheckCutsetsmg_graph(cutsets, cutsets_bf));
  }
}

TEST(Cutsets, Cycles) {
  for (int nodes = 1; nodes < 15; ++nodes) {
    std::vector<std::pair<uint64_t, uint64_t>> edges;
    for (int i = 1; i < nodes; ++i) {
      edges.emplace_back(i - 1, i);
    }
    edges.emplace_back(0, nodes - 1);

    auto *G = BuildGraph(nodes, edges);
    auto cutsets = cutsets_alg::GetCutsets(G);
    auto cutsets_bf = cutsets_bf::GetCutsets(G);

    CutsetType correct;
    for (int i = 0; i < nodes; ++i) {
      for (int j = i + 1; j < nodes; ++j) {
        std::vector<std::pair<uint64_t, uint64_t>> cutset;
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
  auto *G = BuildGraph(7, {{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 6}, {5, 6}});

  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);

  CutsetType correct = {{{0, 1}, {1, 2}}, {{0, 1}, {0, 2}}, {{0, 2}, {2, 1}}, {{3, 4}, {4, 6}}, {{4, 6}, {6, 5}},
                        {{6, 5}, {5, 3}}, {{5, 3}, {3, 4}}, {{3, 4}, {5, 6}}, {{3, 5}, {4, 6}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, HandmadeConnectedGraph1) {
  //    (1)--(3)--(7)
  //   / |    |
  // (0) |   (4)--(5)
  //   \ |     \  /
  //    (2)     (6)
  auto *G = BuildGraph(8, {{0, 1}, {0, 2}, {1, 2}, {1, 3}, {3, 4}, {3, 7}, {4, 5}, {4, 6}, {5, 6}});

  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);

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
  auto *G = BuildGraph(4, {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 2}});

  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);

  CutsetType correct = {{{0, 3}, {3, 2}},         {{0, 1}, {1, 2}},         {{0, 1}, {0, 2}, {0, 3}},
                        {{2, 3}, {2, 0}, {2, 1}}, {{0, 1}, {3, 2}, {0, 2}}, {{0, 3}, {0, 2}, {1, 2}}};

  ASSERT_TRUE(CheckCutsets(cutsets, correct));
  ASSERT_TRUE(CheckCutsets(cutsets_bf, correct));
}

TEST(Cutsets, HandmadeConnectedGraph3) {
  // (0)--(1)
  //  |\   | \
  //  | \  | (2)
  //  |  \ | /
  // (4)--(3)
  auto *G = BuildGraph(5, {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 0}, {0, 3}, {1, 3}});

  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);

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
  auto *G = BuildGraph(
      26, {{0, 1},   {0, 2},   {1, 4},   {1, 3},   {3, 5},   {4, 5},   {2, 6},   {2, 7},   {6, 7},   {8, 9},
           {10, 11}, {11, 12}, {12, 13}, {13, 14}, {10, 14}, {10, 15}, {15, 16}, {16, 17}, {15, 17}, {13, 18},
           {18, 19}, {18, 21}, {18, 20}, {21, 25}, {20, 22}, {22, 23}, {23, 24}, {22, 24}});

  auto cutsets = cutsets_alg::GetCutsets(G);
  auto cutsets_bf = cutsets_bf::GetCutsets(G);

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

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}