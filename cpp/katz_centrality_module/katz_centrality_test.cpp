#include <gtest/gtest.h>
#include <mg_generate.hpp>
#include <mg_test_utils.hpp>

#include "algorithm/katz.hpp"

// TEST(KatzCentrality, KatzSimpleScenario) {
//   auto graph = mg_generate::BuildGraph(5, {{0, 1}, {2, 3}, {3, 0}, {4, 2}, {0, 4}});
//   auto katz_centrality = katz_alg::GetKatzCentrality(*graph);
//   std::vector<std::pair<uint64_t, double>> expected{{0, 0.1}, {1, 0.1}, {2, 0.1}, {3, 0.1}, {4, 0.1}};
//   ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(katz_centrality, expected));
// }

// TEST(KatzCentrality, KatzCentralityTest) {
//   auto graph =
//       mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}},
//       mg_graph::GraphType::kDirectedGraph);
//   auto katz_centrality = katz_alg::GetKatzCentrality(*graph);

//   // Simulating change in graph
//   graph = mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}, {4, 1}},
//                                   mg_graph::GraphType::kDirectedGraph);
//   auto results = katz_alg::UpdateKatz(*graph, {}, {{4, 1}}, {}, {});

//   std::vector<std::pair<uint64_t, double>> expected{{0, 0.1}, {1, 0.1}, {2, 0.1}, {3, 0.1}, {4, 0.1}};
//   ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(katz_centrality, expected));
// }

TEST(KatzCentrality, DynamicAndStaticSimilarity) {
  // Simulating dynamic
  auto graph =
      mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}}, mg_graph::GraphType::kDirectedGraph);
  katz_alg::GetKatzCentrality(*graph);
  graph = mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}, {4, 1}},
                                  mg_graph::GraphType::kDirectedGraph);
  auto dynamic_results = katz_alg::UpdateKatz(*graph, {}, {{4, 1}}, {}, {});

  // Static result
  auto static_results = katz_alg::GetKatzCentrality(*graph);
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(dynamic_results, static_results));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
