#include <gtest/gtest.h>

#include "algorithm/pagerank.hpp"

namespace {

double MaxAbsoluteError(const std::vector<double> &result, const std::vector<double> &correct) {
  auto number_of_nodes = correct.size();
  double max_absolute_error = 0.0;
  for (size_t node_id = 0; node_id < number_of_nodes; node_id++) {
    double absolute_distance = std::abs(result[node_id] - correct[node_id]);
    max_absolute_error = std::max(max_absolute_error, absolute_distance);
  }
  return max_absolute_error;
}
double AverageAbsoluteError(const std::vector<double> &result, const std::vector<double> &correct) {
  auto number_of_nodes = correct.size();
  double manhattan_distance = 0.0;
  for (size_t node_id = 0; node_id < number_of_nodes; node_id++) {
    double absolute_distance = std::abs(result[node_id] - correct[node_id]);
    manhattan_distance += absolute_distance;
  }
  double average_absolute_error = 0;
  if (number_of_nodes > 0) average_absolute_error = manhattan_distance / number_of_nodes;
  return average_absolute_error;
}

bool TestPageRank(const std::vector<double> &result, const std::vector<double> &correct) {
  double max_absolute_error = MaxAbsoluteError(result, correct);
  double average_absolute_error = AverageAbsoluteError(result, correct);
  return (max_absolute_error < 10e-4 && average_absolute_error < 10e-5);
}

}  // namespace

TEST(Pagerank, OneNodeZeroEdges) {
  auto graph = pagerank_alg::PageRankGraph(1, 0, {});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{1.00};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, TwoNodesOneEdge) {
  auto graph = pagerank_alg::PageRankGraph(2, 1, {{0, 1}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.350877362, 0.649122638};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, ZeroNodesZeroEdges) {
  auto graph = pagerank_alg::PageRankGraph(0, 0, {});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, OneNodeOneSelfLoop) {
  auto graph = pagerank_alg::PageRankGraph(1, 1, {{0, 0}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{1.00};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, TwoNodesTwoMultipleEdges) {
  auto graph = pagerank_alg::PageRankGraph(2, 2, {{0, 1}, {0, 1}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.350877362, 0.649122638};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, TwoNodesOneSelfLoop) {
  auto graph = pagerank_alg::PageRankGraph(2, 1, {{1, 1}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.130435201, 0.869564799};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, RandomGraph5Nodes10Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      5, 10, {{0, 2}, {0, 0}, {2, 3}, {3, 1}, {1, 3}, {1, 0}, {1, 2}, {3, 0}, {0, 1}, {3, 2}});

  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.240963851, 0.187763717, 0.240963851, 0.294163985, 0.036144598};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, RandomGraph10Nodes10Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      10, 10, {{9, 5}, {4, 4}, {3, 8}, {0, 5}, {5, 0}, {3, 0}, {7, 9}, {3, 9}, {0, 4}, {0, 4}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.114178360, 0.023587998, 0.023587998, 0.023587998, 0.588577186,
                                      0.098712132, 0.023587998, 0.023587998, 0.030271265, 0.050321066};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, RandomGraph10Nodes9Edges) {
  auto graph =
      pagerank_alg::PageRankGraph(10, 9, {{8, 8}, {2, 2}, {0, 8}, {7, 8}, {1, 6}, {0, 0}, {1, 1}, {6, 3}, {9, 5}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.047683471, 0.047683471, 0.182781325, 0.067949168, 0.027417774,
                                      0.050723042, 0.047683471, 0.027417774, 0.473242731, 0.027417774};
  ASSERT_TRUE(TestPageRank(results, expected));
}
TEST(Pagerank, RandomGraph5Nodes25Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      5, 25, {{3, 3}, {0, 3}, {4, 2}, {1, 1}, {3, 2}, {2, 0}, {4, 0}, {4, 4}, {3, 2}, {4, 1}, {2, 4}, {2, 2}, {2, 3},
              {3, 3}, {0, 0}, {1, 0}, {4, 2}, {4, 0}, {1, 2}, {1, 4}, {4, 0}, {4, 0}, {0, 0}, {4, 0}, {3, 3}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.304824023, 0.049593211, 0.217782046, 0.331928795, 0.095871925};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, RandomGraph4Nodes4Edges) {
  auto graph = pagerank_alg::PageRankGraph(4, 4, {{1, 0}, {3, 0}, {2, 0}, {3, 0}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected = std::vector<double>{0.541985357, 0.152671548, 0.152671548, 0.152671548};
  ASSERT_TRUE(TestPageRank(results, expected));
}

TEST(Pagerank, RandomGraph7Nodes30Edges) {
  auto graph = pagerank_alg::PageRankGraph(
      7, 30, {{0, 6}, {3, 0}, {6, 2}, {0, 3}, {2, 3}, {6, 4}, {1, 1}, {2, 0}, {0, 3}, {5, 0},
              {0, 4}, {5, 2}, {1, 5}, {5, 3}, {2, 3}, {6, 1}, {2, 0}, {6, 1}, {2, 6}, {2, 2},
              {0, 0}, {6, 0}, {6, 0}, {0, 6}, {3, 3}, {6, 3}, {1, 3}, {4, 0}, {1, 2}, {2, 1}});
  auto results = pagerank_alg::ParallelIterativePageRank(graph);
  auto expected =
      std::vector<double>{0.318471859, 0.075311781, 0.071307161, 0.295999683, 0.081155915, 0.037432346, 0.120321254};
  ASSERT_TRUE(TestPageRank(results, expected));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}