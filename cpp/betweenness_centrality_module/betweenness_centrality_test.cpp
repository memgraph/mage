#include <vector>
#include <stack>
#include <iostream>

#include <gtest/gtest.h>
#include <mg_generate.hpp>
#include <mg_test_utils.hpp>

#include "algorithm/betweenness_centrality.hpp"

namespace betweenness_centrality_test_util{
  template <typename T>
  bool TestExactEqualVectors(const std::vector<T> &result, const std::vector<T> &correct) {
    if (result.size() != correct.size()) return false;
    for (auto index = 0; index < result.size(); index++){
      if (result[index] != correct[index]) return false;
    }
    return true;
  }

  template <typename T>
  bool TestExactEqualStacks(std::stack<T> &result, std::stack<T> &correct) {
    if (result.size() != correct.size()) return false;
    while (!result.empty()) {
      auto result_value = result.top(); result.pop();
      auto correct_value = correct.top(); correct.pop();
      if (result_value != correct_value) return false;
    }
    return true;
  }
}

TEST(BetweennessCentralityUtilBFS, UndirectedTree) {
  //    (4)
  //   /   \
  // (2)   (1)
  //  |   /   \
  // (0)(3)   (5)
  auto graph = mg_generate::BuildGraph(6, {{2, 4}, {1, 4}, {0, 2}, {1, 3}, {1, 5}}, mg_graph::GraphType::kUndirectedGraph);
  auto number_of_nodes = graph->Nodes().size();

  // data structures used in BFS
  std::stack<std::uint64_t> visited;
  std::vector<std::vector<std::uint64_t>> predecessors (number_of_nodes, std::vector<std::uint64_t>());
  std::vector<std::uint64_t> shortest_paths_counter (number_of_nodes, 0);
  betweenness_centrality_util::BFS(0, *graph, visited, predecessors, shortest_paths_counter);

  std::stack<uint64_t> expected_visited ({0, 2, 4, 1, 3, 5});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualStacks(visited, expected_visited));

  std::vector<std::vector<uint64_t>> expected_predecessors ({{}, {4}, {0}, {1}, {2}, {1}});
  for(auto node_id = 0; node_id < number_of_nodes; node_id++){
    ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(predecessors[node_id], expected_predecessors[node_id]));
  }

  std::vector<uint64_t> expected_shortest_paths_counter ({1, 1, 1, 1, 1, 1});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(shortest_paths_counter, expected_shortest_paths_counter));
}

TEST(BetweennessCentralityUtilBFS, DisconnectedUndirectedGraph) {
  //    (1)  (3)---(4)
  //   / |    |     |
  // (0) |    |     |
  //   \ |    |     |
  //    (2)  (5)---(6)
  auto graph = mg_generate::BuildGraph(7, {{0, 1}, {0, 2}, {1, 2}, {3, 4}, {3, 5}, {4, 6}, {5, 6}}, mg_graph::GraphType::kUndirectedGraph);
  auto number_of_nodes = graph->Nodes().size();

  // data structures used in BFS
  std::stack<std::uint64_t> visited;
  std::vector<std::vector<std::uint64_t>> predecessors (number_of_nodes, std::vector<std::uint64_t>());
  std::vector<std::uint64_t> shortest_paths_counter (number_of_nodes, 0);
  betweenness_centrality_util::BFS(0, *graph, visited, predecessors, shortest_paths_counter);

  std::stack<uint64_t> expected_visited ({0, 1, 2});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualStacks(visited, expected_visited));

  std::vector<std::vector<uint64_t>> expected_predecessors ({{},{0},{0},{},{},{},{}});
  for(auto node_id = 0; node_id < number_of_nodes; node_id++){
    ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(predecessors[node_id], expected_predecessors[node_id]));
  }

  std::vector<uint64_t> expected_shortest_paths_counter ({1, 1, 1, 0, 0, 0, 0});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(shortest_paths_counter, expected_shortest_paths_counter));
}

TEST(BetweennessCentralityUtilBFS, UndirectedCyclicGraph) {
  //    (1)--(3)
  //   / |    | \
  // (0) |    |  (5)
  //   \ |    | /
  //    (2)--(4)
  auto graph = mg_generate::BuildGraph(6, {{0, 1}, {0, 2}, {1, 3}, {1, 2}, {2, 4}, {3, 4}, {3, 5}, {4, 5}},
                                      mg_graph::GraphType::kUndirectedGraph);
  auto number_of_nodes = graph->Nodes().size();

  // data structures used in BFS
  std::stack<std::uint64_t> visited;
  std::vector<std::vector<std::uint64_t>> predecessors (number_of_nodes, std::vector<std::uint64_t>());
  std::vector<std::uint64_t> shortest_paths_counter (number_of_nodes, 0);
  betweenness_centrality_util::BFS(0, *graph, visited, predecessors, shortest_paths_counter);

  std::stack<uint64_t> expected_visited ({0, 1, 2, 3, 4, 5});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualStacks(visited, expected_visited));

  std::vector<std::vector<uint64_t>> expected_predecessors ({{}, {0}, {0}, {1}, {2}, {3, 4}});
  for(auto node_id = 0; node_id < number_of_nodes; node_id++){
    ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(predecessors[node_id], expected_predecessors[node_id]));
  }

  std::vector<uint64_t> expected_shortest_paths_counter ({1, 1, 1, 1, 1, 2});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(shortest_paths_counter, expected_shortest_paths_counter));
}

TEST(BetweennessCentralityUtilBFS, UndirectedMultipleShortestPaths) { 
  //    (1)        (5)
  //   /   \      /   \
  // (0)   (3)--(4)    (7)--(8)
  //   \   /      \   /
  //    (2)        (6)  
  auto graph = mg_generate::BuildGraph(9, {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {4, 6},
                                      {5, 7}, {6, 7}, {7, 8}}, mg_graph::GraphType::kUndirectedGraph);
  auto number_of_nodes = graph->Nodes().size();

  // data structures used in BFS
  std::stack<std::uint64_t> visited;
  std::vector<std::vector<std::uint64_t>> predecessors (number_of_nodes, std::vector<std::uint64_t>());
  std::vector<std::uint64_t> shortest_paths_counter (number_of_nodes, 0);
  betweenness_centrality_util::BFS(0, *graph, visited, predecessors, shortest_paths_counter);

  std::stack<uint64_t> expected_visited ({0, 1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualStacks(visited, expected_visited));

  std::vector<std::vector<uint64_t>> expected_predecessors ({{}, {0}, {0}, {1, 2}, {3}, {4}, {4}, {5, 6}, {7}});
  for(auto node_id = 0; node_id < number_of_nodes; node_id++){
    ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(predecessors[node_id], expected_predecessors[node_id]));
  }

  std::vector<uint64_t> expected_shortest_paths_counter ({1, 1, 1, 2, 2, 2, 2, 4, 4});
  ASSERT_TRUE(betweenness_centrality_test_util::TestExactEqualVectors(shortest_paths_counter, expected_shortest_paths_counter));
}

TEST(BetweennessCentrality, EmptyGraph) {
  auto graph = mg_generate::BuildGraph(0, {});
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, false, false);
  ASSERT_EQ(result.size(), 0);
}

TEST(BetweennessCentrality, SingleNode) {
  auto graph = mg_generate::BuildGraph(1, {});
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, false, false);
  std::vector<double> expected ({0.0});
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

TEST(BetweennessCentrality, DisconnectedNodes) {
  auto graph = mg_generate::BuildGraph(100, {});
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, false, false);
  std::vector<double> expected (100, 0.0);
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

TEST(BetweennessCentrality, UndirectedMultipleShortestPaths) { 
  //    (1)        (5)
  //   /   \      /   \
  // (0)   (3)--(4)    (7)--(8)
  //   \   /      \   /
  //    (2)        (6)  
  auto graph = mg_generate::BuildGraph(9, {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {4, 6},
                                      {5, 7}, {6, 7}, {7, 8}}, mg_graph::GraphType::kUndirectedGraph);
  
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, false, false);
  std::vector<double> expected ({0.5, 3.0, 3.0, 15.5, 16.5, 5.0, 5.0, 7.5, 0.0});
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

TEST(BetweennessCentrality, UndirectedMultipleShortestPathsNormalized) { 
  //    (1)        (5)
  //   /   \      /   \
  // (0)   (3)--(4)    (7)--(8)
  //   \   /      \   /
  //    (2)        (6)  
  auto graph = mg_generate::BuildGraph(9, {{0, 1}, {0, 2}, {1, 3}, {2, 3}, {3, 4}, {4, 5}, {4, 6},
                                      {5, 7}, {6, 7}, {7, 8}}, mg_graph::GraphType::kUndirectedGraph);
  
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, false, true);
  std::vector<double> expected ({0.017857142857142856, 0.10714285714285714, 0.10714285714285714, 0.5535714285714285,
                                0.5892857142857143, 0.17857142857142855, 0.17857142857142855, 0.26785714285714285, 0.0});
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

TEST(BetweennessCentrality, DirectedMultipleShortestPaths) { 
  //
  //  (0) → (1) → (4)
  //   ↑     ↕     ↓
  //  (2) ↔ (3) ← (5)
  //   
  auto graph = mg_generate::BuildGraph(6, {{0, 1}, {1, 3}, {1, 4}, {2, 0}, {2, 3}, {3, 2}, {3, 1},
                                      {4, 5}, {5, 3}}, mg_graph::GraphType::kDirectedGraph);
  
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, true, false);
  for(auto a : result) std::cout << a << " ";
  std::vector<double> expected ({1.5, 9.0, 4.0, 11.5, 4.0, 4.0});
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

TEST(BetweennessCentrality, DirectedMultipleShortestPathsNormalized) { 
  //
  //  (0) → (1) → (4)
  //   ↑     ↕     ↓
  //  (2) ↔ (3) ← (5)
  //   
  auto graph = mg_generate::BuildGraph(6, {{0, 1}, {1, 3}, {1, 4}, {2, 0}, {2, 3}, {3, 2}, {3, 1},
                                      {4, 5}, {5, 3}}, mg_graph::GraphType::kDirectedGraph);
  
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, true, true);
  std::vector<double> expected ({0.07500000000000001, 0.45, 0.2, 0.5750000000000001, 0.2, 0.2});
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

TEST(BetweennessCentrality, HandmadeConnectedGraph) {
  //    (1)--(3)--(7)     (10)     (14)
  //   / |    |    |     /    \     |
  // (0) |   (4)--(5)--(9)   (12)--(13)
  //   \ |     \  / \    \   /
  //    (2)     (6) (8)   (11)
  auto graph = mg_generate::BuildGraph(15, {{0, 1},
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
  auto result = betweenness_centrality_alg::BetweennessCentrality(*graph, false, false);
  std::vector<double> expected ({0.0, 24.0, 0.0, 33.5, 20.0, 56.5, 0.0, 16.0, 0.0, 45.5, 15.0, 15.0, 24.5, 13.0, 0.0});
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(result, expected));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}