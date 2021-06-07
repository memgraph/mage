#include <stack>
#include <vector>

#include <gtest/gtest.h>
#include <mg_generate.hpp>
#include <mg_test_utils.hpp>

#include "algorithm/betweenness_centrality.hpp"

class BCUtilBFSParametersTests
    : public ::testing::TestWithParam<
          std::tuple<int, std::vector<std::pair<std::uint64_t, std::uint64_t>>, mg_graph::GraphType,
                     std::stack<uint64_t>, std::vector<std::vector<uint64_t>>, std::vector<uint64_t>>> {
 protected:
  std::unique_ptr<mg_graph::Graph<>> graph =
      mg_generate::BuildGraph(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
  std::stack<uint64_t> expected_visited = std::get<3>(GetParam());
  std::vector<std::vector<uint64_t>> expected_predecessors = std::get<4>(GetParam());
  std::vector<uint64_t> expected_shortest_paths_counter = std::get<5>(GetParam());
};

TEST_P(BCUtilBFSParametersTests, BetweennessCentralityUtilBFS) {
  auto number_of_nodes = graph->Nodes().size();

  // data structures used in BFS
  std::stack<std::uint64_t> visited;
  std::vector<std::vector<std::uint64_t>> predecessors(number_of_nodes, std::vector<std::uint64_t>());
  std::vector<std::uint64_t> shortest_paths_counter(number_of_nodes, 0);
  betweenness_centrality_util::BFS(0, *graph, visited, predecessors, shortest_paths_counter);

  ASSERT_EQ(visited, expected_visited);

  for (auto node_id = 0; node_id < number_of_nodes; node_id++) {
    ASSERT_EQ(predecessors[node_id], expected_predecessors[node_id]);
  }

  ASSERT_EQ(shortest_paths_counter, expected_shortest_paths_counter);
}

INSTANTIATE_TEST_CASE_P(
    BCUtilTest, BCUtilBFSParametersTests,
    ::testing::Values(
        std::make_tuple(6,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>({{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}}),
                        mg_graph::GraphType::kUndirectedGraph, std::stack<std::uint64_t>({0, 2, 4, 1, 3, 5}),
                        std::vector<std::vector<uint64_t>>({{}, {4}, {0}, {1}, {2}, {1}}),
                        std::vector<uint64_t>({1, 1, 1, 1, 1, 1})),
        std::make_tuple(7,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>(
                            {{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 6}, {5, 6}}),
                        mg_graph::GraphType::kUndirectedGraph, std::stack<uint64_t>({0, 1, 2}),
                        std::vector<std::vector<uint64_t>>({{}, {0}, {0}, {}, {}, {}, {}}),
                        std::vector<uint64_t>({1, 1, 1, 0, 0, 0, 0})),
        std::make_tuple(9,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>(
                            {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {5, 7}, {6, 7}, {7, 8}}),
                        mg_graph::GraphType::kUndirectedGraph, std::stack<uint64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8}),
                        std::vector<std::vector<uint64_t>>({{}, {0}, {0}, {1, 2}, {3}, {4}, {4}, {5, 6}, {7}}),
                        std::vector<uint64_t>({1, 1, 1, 2, 2, 2, 2, 4, 4})),
        std::make_tuple(9,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>(
                            {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {5, 7}, {6, 7}, {7, 8}}),
                        mg_graph::GraphType::kUndirectedGraph, std::stack<uint64_t>({0, 1, 2, 3, 4, 5, 6, 7, 8}),
                        std::vector<std::vector<uint64_t>>({{}, {0}, {0}, {1, 2}, {3}, {4}, {4}, {5, 6}, {7}}),
                        std::vector<uint64_t>({1, 1, 1, 2, 2, 2, 2, 4, 4}))));

class BetweennessCentralityParametersTests
    : public ::testing::TestWithParam<std::tuple<int, std::vector<std::pair<std::uint64_t, std::uint64_t>>,
                                                 mg_graph::GraphType, std::vector<double>, bool, bool>> {
 protected:
  std::unique_ptr<mg_graph::Graph<>> graph =
      mg_generate::BuildGraph(std::get<0>(GetParam()), std::get<1>(GetParam()), std::get<2>(GetParam()));
  std::vector<double> expected = std::get<3>(GetParam());
  bool directed = std::get<4>(GetParam());
  bool normalized = std::get<5>(GetParam());
};

TEST_P(BetweennessCentralityParametersTests, BetweennessCentralityTest) {
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, directed, normalized);
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

INSTANTIATE_TEST_CASE_P(
    BCUtilTest, BetweennessCentralityParametersTests,
    ::testing::Values(
        std::make_tuple(1, std::vector<std::pair<std::uint64_t, std::uint64_t>>({}),
                        mg_graph::GraphType::kUndirectedGraph, std::vector<double>({0.0}), false, false),
        std::make_tuple(100, std::vector<std::pair<std::uint64_t, std::uint64_t>>({}),
                        mg_graph::GraphType::kUndirectedGraph, std::vector<double>(100, 0.0), false, false),
        std::make_tuple(9,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>(
                            {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {5, 7}, {6, 7}, {7, 8}}),
                        mg_graph::GraphType::kUndirectedGraph,
                        std::vector<double>({0.5, 3.0, 3.0, 15.5, 16.5, 5.0, 5.0, 7.5, 0.0}), false, false),
        std::make_tuple(9,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>(
                            {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {4, 6}, {5, 7}, {6, 7}, {7, 8}}),
                        mg_graph::GraphType::kUndirectedGraph,
                        std::vector<double>({0.017857142857142856, 0.10714285714285714, 0.10714285714285714,
                                             0.5535714285714285, 0.5892857142857143, 0.17857142857142855,
                                             0.17857142857142855, 0.26785714285714285, 0.0}),
                        false, true),
        std::make_tuple(6,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>(
                            {{0, 1}, {1, 3}, {1, 4}, {2, 0}, {2, 3}, {3, 2}, {3, 1}, {4, 5}, {5, 3}}),
                        mg_graph::GraphType::kDirectedGraph, std::vector<double>({1.5, 9.0, 4.0, 11.5, 4.0, 4.0}), true,
                        false),
        std::make_tuple(6,
                        std::vector<std::pair<std::uint64_t, std::uint64_t>>(
                            {{0, 1}, {1, 3}, {1, 4}, {2, 0}, {2, 3}, {3, 2}, {3, 1}, {4, 5}, {5, 3}}),
                        mg_graph::GraphType::kDirectedGraph,
                        std::vector<double>({0.07500000000000001, 0.45, 0.2, 0.5750000000000001, 0.2, 0.2}), true,
                        true)));

TEST(BetweennessCentrality, EmptyGraph) {
  auto graph = mg_generate::BuildGraph(0, {});
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, false, false);
  ASSERT_EQ(result.size(), 0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}