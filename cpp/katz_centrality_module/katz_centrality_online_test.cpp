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

// TEST(KatzCentrality, KatzDynamicAddEdges) {
//   auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
//                                        {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5,
//                                        4}}, mg_graph::GraphType::kDirectedGraph);
//   auto katz_centrality = katz_alg::GetKatzCentrality(*graph, 0.2);
//   ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 4, 0, 3, 5, 2, 6}));

//   graph = mg_generate::BuildGraph(
//       {0, 1, 2, 3, 4, 5, 6},
//       {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}, {3, 5}, {2, 5}},
//       mg_graph::GraphType::kDirectedGraph);
//   katz_centrality = katz_alg::UpdateKatz(*graph, {}, {{3, 5}, {2, 5}}, {}, {});
//   ASSERT_TRUE(CompareRankingSort(katz_centrality, {1, 5, 4, 0, 3, 2, 6}));
// }

TEST(KatzCentrality, KatzDynamicAddEdgesCompareSimilar) {
  // auto graph = mg_generate::BuildGraph({0, 1, 2, 3, 4, 5, 6},
  //                                      {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5,
  //                                      4}}, mg_graph::GraphType::kDirectedGraph);
  // katz_alg::GetKatzCentrality(*graph, 0.2);

  auto updated_graph = mg_generate::BuildGraph(
      {0, 1, 2, 3, 4, 5, 6},
      {{0, 5}, {0, 1}, {1, 4}, {2, 1}, {3, 1}, {4, 0}, {4, 3}, {4, 1}, {5, 0}, {5, 4}, {3, 5}, {2, 5}},
      mg_graph::GraphType::kDirectedGraph);
  // auto katz_centrality_dynamic = katz_alg::UpdateKatz(*updated_graph, {}, {{3, 5}, {2, 5}}, {}, {});

  auto katz_centrality_static = katz_alg::GetKatzCentrality(*updated_graph, 0.2);
  // ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(katz_centrality_static, katz_centrality_dynamic));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
