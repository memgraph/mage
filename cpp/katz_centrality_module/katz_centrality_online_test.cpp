#include <gtest/gtest.h>
#include <mg_generate.hpp>
#include <mg_test_utils.hpp>

#include "algorithm/katz.hpp"
namespace {
bool CompareRankingSort(const std::vector<std::pair<uint64_t, double>> &result,
                        const std::vector<std::uint64_t> &ranking, double threshold = 1e-6) {
  if (result.size() != ranking.size()) return false;

  std::vector<std::pair<std::uint64_t, double>> result_cp(result);

  std::sort(result_cp.begin(), result_cp.end(), [threshold](const auto &a, const auto &b) -> bool {
    auto [key_a, value_a] = a;
    auto [key_b, value_b] = b;

    auto diff = abs(value_b - value_a);
    return (value_a > value_b && diff > threshold) || (diff < threshold && key_a < key_b);
  });

  for (std::size_t i = 0; i < ranking.size(); i++) {
    auto [node_id, _] = result_cp[i];
    if (ranking[i] != node_id) return false;
  }
  return true;
}
}  // namespace

TEST(KatzCentrality, AddNodesAndEdges) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                       {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6, 13},
                                  {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {13}, {}, {}, {}, {});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6, 13}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6, 13},
                                  {{0, 5},
                                   {0, 1},
                                   {1, 4},
                                   {2, 1},
                                   {3, 1},
                                   {4, 0},
                                   {4, 3},
                                   {4, 1},
                                   {5, 0},
                                   {5, 4},
                                   {13, 1},
                                   {2, 13},
                                   {13, 5},
                                   {4, 13}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {}, {{13, 1}, {2, 13}, {13, 5}, {4, 13}}, {10, 11, 12, 13}, {}, {});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 5, 13, 3, 2, 6}));
}

TEST(KatzCentrality, AddEdges) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                       {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));

  graph = mg_generate::BuildGraph(
      {0, 1, 2, 3, 4, 5, 6},
      {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}, {3, 5}, {2, 5}},
      mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {}, {{3, 5}, {2, 5}}, {10, 11}, {}, {});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 5, 4, 0, 3, 2, 6}));
}

TEST(KatzCentrality, AddEdgesGradually) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6}, {{3, 1}, {4, 0}}, mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {0, 1, 2, 3, 4, 5, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6}, {{3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {}, {{4, 3}, {4, 1}, {5, 0}}, {2, 3, 4}, {}, {});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 0, 3, 2, 4, 5, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                  {{3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {0, 5}, {0, 1}, {1, 4}, {2, 1}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {}, {{0, 5}, {0, 1}, {1, 4}, {2, 1}}, {5, 6, 7, 8}, {}, {});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 0, 4, 5, 3, 2, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                  {{3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {0, 5}, {0, 1}, {1, 4}, {2, 1}, {5, 4}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {}, {{5, 4}}, {9}, {}, {});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));
}

TEST(KatzCentrality, DeleteEdges) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                       {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6}, {{0, 5}, {0, 1}, {1, 4}, {4, 0}, {4, 3}, {5, 0}, {5, 4}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {}, {}, {}, {}, {{2, 1}, {3, 1}, {4, 1}});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {0, 4, 1, 5, 3, 2, 6}));
}

TEST(KatzCentrality, DeleteAllEdges) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                       {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6}, {}, mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(
      *graph, {}, {}, {}, {}, {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {0, 1, 2, 3, 4, 5, 6}));
}

TEST(KatzCentrality, DeleteAllAndRevert) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                       {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6}, {}, mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(
      *graph, {}, {}, {}, {}, {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {0, 1, 2, 3, 4, 5, 6}));

  graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                  {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality =
      katz_alg::UpdateKatz(*graph, {}, {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                           {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {}, {});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));
}

TEST(KatzCentrality, DeleteNodes) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
                                       {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));

  std::cout << "Dynamic algorithm" << std::endl;
  graph = mg_generate::BuildGraph({0, 1, 2, 3, 5, 6}, {{0, 5}, {0, 1}, {2, 1}, {3, 1}, {5, 0}},
                                  mg_graph::GraphType::kDirectedGraph);
  katz_centrality = katz_alg::UpdateKatz(*graph, {}, {}, {}, {4}, {{1, 4}, {4, 0}, {4, 3}, {4, 1}, {5, 4}});
  ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 0, 5, 2, 3, 6}));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
