#include <gtest/gtest.h>
#include <mg_test_utils.hpp>

#include "algorithm/pagerank.hpp"

TEST(Pagerank, OneNodeZeroEdges) {
  auto graph = pagerank_alg::PageRankGraph(1, 0, {});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{1.00};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, TwoNodesOneEdge) {
  auto graph = pagerank_alg::PageRankGraph(2, 1, {{0, 1}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.350877362, 0.649122638};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, ZeroNodesZeroEdges) {
  auto graph = pagerank_alg::PageRankGraph(0, 0, {});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, OneNodeOneSelfLoop) {
  auto graph = pagerank_alg::PageRankGraph(1, 1, {{0, 0}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{1.00};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, TwoNodesTwoMultipleEdges) {
  auto graph = pagerank_alg::PageRankGraph(2, 2, {{0, 1}, {0, 1}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.350877362, 0.649122638};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, TwoNodesOneSelfLoop) {
  auto graph = pagerank_alg::PageRankGraph(2, 1, {{1, 1}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.130435201, 0.869564799};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, RandomGraph5Nodes10Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      5, 10, {{0, 2}, {0, 0}, {2, 3}, {3, 1}, {1, 3}, {1, 0}, {1, 2}, {3, 0}, {0, 1}, {3, 2}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.240963851, 0.187763717, 0.240963851, 0.294163985, 0.036144598};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, RandomGraph10Nodes10Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      10, 10, {{9, 5}, {4, 4}, {3, 8}, {0, 5}, {5, 0}, {3, 0}, {7, 9}, {3, 9}, {0, 4}, {0, 4}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.114178360, 0.023587998, 0.023587998, 0.023587998, 0.588577186,
                               0.098712132, 0.023587998, 0.023587998, 0.030271265, 0.050321066};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, RandomGraph10Nodes9Edges) {
  auto graph =
      pagerank_alg::PageRankGraph(10, 9, {{8, 8}, {2, 2}, {0, 8}, {7, 8}, {1, 6}, {0, 0}, {1, 1}, {6, 3}, {9, 5}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.047683471, 0.047683471, 0.182781325, 0.067949168, 0.027417774,
                               0.050723042, 0.047683471, 0.027417774, 0.473242731, 0.027417774};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}
TEST(Pagerank, RandomGraph5Nodes25Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      5, 25, {{3, 3}, {0, 3}, {4, 2}, {1, 1}, {3, 2}, {2, 0}, {4, 0}, {4, 4}, {3, 2}, {4, 1}, {2, 4}, {2, 2}, {2, 3},
              {3, 3}, {0, 0}, {1, 0}, {4, 2}, {4, 0}, {1, 2}, {1, 4}, {4, 0}, {4, 0}, {0, 0}, {4, 0}, {3, 3}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.304824023, 0.049593211, 0.217782046, 0.331928795, 0.095871925};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, RandomGraph4Nodes4Edges) {
  auto graph = pagerank_alg::PageRankGraph(4, 4, {{1, 0}, {3, 0}, {2, 0}, {3, 0}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.541985357, 0.152671548, 0.152671548, 0.152671548};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

TEST(Pagerank, RandomGraph7Nodes30Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      7, 30, {{0, 6}, {3, 0}, {6, 2}, {0, 3}, {2, 3}, {6, 4}, {1, 1}, {2, 0}, {0, 3}, {5, 0},
              {0, 4}, {5, 2}, {1, 5}, {5, 3}, {2, 3}, {6, 1}, {2, 0}, {6, 1}, {2, 6}, {2, 2},
              {0, 0}, {6, 0}, {6, 0}, {0, 6}, {3, 3}, {6, 3}, {1, 3}, {4, 0}, {1, 2}, {2, 1}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  std::vector<double> expected{0.318471859, 0.075311781, 0.071307161, 0.295999683,
                               0.081155915, 0.037432346, 0.120321254};
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}