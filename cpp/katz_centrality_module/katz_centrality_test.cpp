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

TEST(KatzCentrality, KatzRankingExample_1) {
  auto graph = mg_generate::BuildGraph(std::vector<std::uint64_t>{0}, {}, mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{0};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_2) {
  auto graph = mg_generate::BuildGraph({0, 1}, {{0, 1}}, mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{1, 0};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_3) {
  auto graph = mg_generate::BuildGraph({0, 1, 2}, {{0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 0}, {2, 1}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{0, 1, 2};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_4) {
  auto graph =
      mg_generate::BuildGraph({0, 1, 2, 3}, {{0, 1}, {2, 1}, {3, 0}, {3, 1}}, mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{1, 0, 2, 3};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_5) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4}, {{0, 4}, {1, 4}, {2, 1}, {3, 0}, {3, 1}, {3, 4}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{4, 1, 0, 2, 3};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_6) {
  auto graph =
      mg_generate::BuildGraph({0, 1, 2, 3, 4, 5}, {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}},
                              mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{1, 0, 4, 5, 3, 2};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_7) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6}, {{0, 5}, {2, 1}, {4, 0}, {5, 0}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{0, 5, 1, 2, 3, 4, 6};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_8) {
  auto graph =
      mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6, 7},
                              {{0, 1}, {0, 3}, {0, 2}, {1, 0}, {3, 2}, {4, 3}, {4, 2}, {6, 0}, {6, 3}, {6, 5}, {7, 4}},
                              mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{2, 3, 0, 1, 4, 5, 6, 7};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_9) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6, 7, 8},
                                       {{0, 8}, {1, 0}, {1, 8}, {1, 6}, {1, 3}, {3, 2}, {3, 8}, {4, 3}, {6, 3}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{8, 3, 2, 0, 6, 1, 4, 5, 7};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_10) {
  auto graph =
      mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                              {{0, 1}, {0, 8}, {1, 0}, {1, 8}, {1, 9}, {3, 2}, {3, 8}, {4, 3}, {6, 0}, {6, 3}, {8, 9}},
                              mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{8, 9, 0, 3, 1, 2, 4, 5, 6, 7};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_11) {
  auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                                       {{0, 4},
                                        {0, 1},
                                        {0, 8},
                                        {0, 2},
                                        {1, 9},
                                        {2, 1},
                                        {3, 10},
                                        {3, 7},
                                        {6, 0},
                                        {6, 5},
                                        {8, 9},
                                        {8, 6},
                                        {9, 4},
                                        {10, 1},
                                        {10, 8}},
                                       mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{1, 9, 4, 8, 6, 0, 5, 2, 7, 10, 3};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_12) {
  auto graph =
      mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                              {{0, 11}, {1, 3}, {3, 2}, {3, 8}, {4, 3}, {8, 1}, {9, 6}, {10, 1}, {10, 11}, {11, 1}},
                              mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{1, 3, 11, 2, 8, 6, 0, 4, 5, 7, 9, 10};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_13) {
  auto graph = mg_generate::BuildGraph(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      {{0, 11}, {0, 12}, {1, 3}, {3, 2},  {3, 8},   {4, 3},  {4, 12},  {6, 3},  {7, 9},  {8, 1},
       {8, 3},  {9, 6},  {9, 0}, {10, 1}, {10, 11}, {11, 1}, {11, 10}, {11, 8}, {11, 6}, {12, 2}},
      mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{3, 1, 8, 2, 6, 11, 12, 10, 0, 9, 4, 5, 7};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

TEST(KatzCentrality, KatzRankingExample_14) {
  auto graph = mg_generate::BuildGraph(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
      {{0, 11}, {1, 3}, {3, 2}, {3, 8}, {4, 3}, {8, 1}, {9, 6}, {9, 0}, {10, 1}, {10, 11}, {11, 1}},
      mg_graph::GraphType::kDirectedGraph);
  auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
  std::vector<std::uint64_t> expected_ranking{1, 3, 11, 2, 8, 0, 6, 4, 5, 7, 9, 10, 12, 13};
  ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
}

// TEST(KatzCentrality, KatzSimpleScenario) {
//   auto graph = mg_generate::BuildGraph(5, {{0, 1}, {2, 3}, {3, 0}, {4, 2}, {0, 4}});
//   auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
//   std::vector<std::pair<uint64_t, double>> expected{{0, 0.1}, {1, 0.1}, {2, 0.1}, {3, 0.1}, {4, 0.1}};
//   ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(katz_centrality, expected));
// }

// TEST(KatzCentrality, KatzCentralityTest) {
//   auto graph =
//       mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}},
//       mg_graph::GraphType::kDirectedGraph);
//   auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);

//   // Simulating change in graph
//   graph = mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}, {4, 1}},
//                                   mg_graph::GraphType::kDirectedGraph);
//   auto results = katz_alg::UpdateKatz(*graph, {}, {{4, 1}}, {}, {});

//   std::vector<std::pair<uint64_t, double>> expected{{0, 0.1}, {1, 0.1}, {2, 0.1}, {3, 0.1}, {4, 0.1}};
//   ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(katz_centrality, expected));
// }

// TEST(KatzCentrality, DynamicAndStaticSimilarity) {
//   // Simulating dynamic
//   auto graph =
//       mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}},
//       mg_graph::GraphType::kDirectedGraph);
//   katz_alg::GetKatzCentrality(*graph);
//   graph = mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}, {4, 1}},
//                                   mg_graph::GraphType::kDirectedGraph);
//   auto dynamic_results = katz_alg::UpdateKatz(*graph, {}, {{4, 1}}, {}, {});

//   // Static result
//   auto static_results = katz_alg::GetKatzCentrality(*graph);
//   ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(dynamic_results, static_results));
// }

// TEST(KatzCentrality, KatzRankingExample1) {
//   auto graph =
//       mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}},
//       mg_graph::GraphType::kDirectedGraph);
//   auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
//   std::vector<std::uint64_t> expected_ranking{2, 0, 7, 1, 4};
//   ASSERT_TRUE(CompareRankingSort(katz_centrality, expected_ranking));
// }

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
