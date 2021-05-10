#include <gtest/gtest.h>

#include <mg_generate.hpp>
#include <mg_test_utils.hpp>

#include "algorithm/betweenness_centrality.hpp"

TEST(BetweennessCentralityBFS, OneNodeZeroEdges) {
  auto graph = mg_generate::BuildGraph(10, {{0, 6}, {0, 9}, {1, 7}, {2, 6}, {3, 7}, {3, 8}, {4, 5}});
  auto number_of_nodes = graph->Nodes().size();

  // data structures used in BFS
  std::stack<std::uint64_t> visited;
  std::vector<std::vector<std::uint64_t>> predecessors (number_of_nodes, std::vector<std::uint64_t>());
  std::vector<std::uint64_t> shortest_paths_counter (number_of_nodes, 0);

  betweenness_centrality_util::BFS(0, *graph, visited, predecessors, shortest_paths_counter);

  auto expected = std::vector<double>({1});
  //ASSERT_TRUE(mg_test_utility::TestEqualVectors<double>(results, expected));
}

TEST(BetweennessCentrality, OneNodeZeroEdges) {
  auto graph = mg_generate::BuildGraph(1, {});

  auto results = betweenness_centrality_alg::BetweennessCentrality(*graph);
  auto expected = std::vector<double>({1});
  ASSERT_TRUE(mg_test_utility::TestEqualVectors<double>(results, expected));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}