#include <tuple>

#include <gtest/gtest.h>
#include <mg_generate.hpp>
#include <mg_graph.hpp>
#include <mg_test_utils.hpp>

#include "algorithm_approx/pagerank.hpp"

class PagerankApproxTest : public ::testing::Test {
 protected:
  std::unique_ptr<mg_graph::Graph<>> graph;
  virtual void SetUp() {
    pagerank_approx_alg::Reset();
    graph =
        mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}}, mg_graph::GraphType::kDirectedGraph);
  }
};

TEST_F(PagerankApproxTest, SmallGraphSet) {
  auto results = pagerank_approx_alg::SetPagerank(*graph);

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.268}, {1, 0.081}, {2, 0.289}, {7, 0.22}, {4, 0.081}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphGet) {
  auto results = pagerank_approx_alg::GetPagerank(*graph);

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.328}, {1, 0.081}, {2, 0.289}, {7, 0.22}, {4, 0.081}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphUpdate) {
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {}, {}, {});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.328}, {1, 0.081}, {2, 0.289}, {7, 0.22}, {4, 0.081}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphUpdateCreateVertex) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({0, 1, 2, 7, 4, 10}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}},
                                  mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {10}, {}, {}, {});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.303}, {1, 0.075}, {2, 0.268},
                                                       {7, 0.204}, {4, 0.075}, {10, 0.075}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphUpdateCreateEdge) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}, {7, 4}},
                                  mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {{7, 4}}, {}, {});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.276}, {1, 0.068}, {2, 0.244}, {7, 0.186}, {4, 0.226}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphUpdateCreateVertexAndEdge) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({0, 1, 2, 7, 4, 10, 6}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}, {7, 4}, {0, 10}},
                                  mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {10, 6}, {{7, 4}, {0, 10}}, {}, {});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.229}, {1, 0.068},  {2, 0.19}, {7, 0.133},
                                                       {4, 0.18},  {10, 0.133}, {6, 0.068}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphUpdateDeleteEdge) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 7}, {1, 2}, {2, 0}}, mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {}, {}, {{0, 2}});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.268}, {1, 0.104}, {2, 0.193}, {7, 0.331}, {4, 0.104}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphUpdateDeleteEdgeAndVertex) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({0, 1, 2, 7}, {{1, 2}, {2, 0}}, mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {}, {7}, {{0, 2}, {0, 7}});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.401}, {1, 0.156}, {2, 0.288}, {4, 0.156}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, SmallGraphUpdateMixActions) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}, {7, 4}},
                                  mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {{7, 4}}, {}, {});

  graph = mg_generate::BuildGraph({0, 1, 2, 7, 4, 10}, {{0, 7}, {1, 2}, {2, 0}, {7, 4}},
                                  mg_graph::GraphType::kDirectedGraph);
  results = pagerank_approx_alg::UpdatePagerank(*graph, {10}, {}, {}, {{0, 2}});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.193}, {1, 0.075}, {2, 0.139},
                                                       {7, 0.239}, {4, 0.278}, {10, 0.075}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, DeleteDetachEdge) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({0, 2, 7, 4}, {{0, 2}, {0, 7}, {2, 0}}, mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {}, {1}, {{1, 2}});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.347}, {2, 0.267}, {7, 0.267}, {4, 0.119}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, DeleteDetachMultiEdge) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({1, 2, 7, 4}, {{1, 2}}, mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {}, {0}, {{0, 2}, {0, 7}, {2, 0}});

  std::vector<std::pair<uint64_t, double>> expected = {{1, 0.206}, {2, 0.382}, {7, 0.206}, {4, 0.206}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, DeleteGraph) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({}, {}, mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {}, {0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}});

  std::vector<std::pair<uint64_t, double>> expected = {};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

TEST_F(PagerankApproxTest, DeleteAndRevertGraph) {
  pagerank_approx_alg::SetPagerank(*graph);

  // Simulating change in graph
  graph = mg_generate::BuildGraph({}, {}, mg_graph::GraphType::kDirectedGraph);
  auto results = pagerank_approx_alg::UpdatePagerank(*graph, {}, {}, {0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}});

  graph =
      mg_generate::BuildGraph({0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}}, mg_graph::GraphType::kDirectedGraph);
  results = pagerank_approx_alg::UpdatePagerank(*graph, {0, 1, 2, 7, 4}, {{0, 2}, {0, 7}, {1, 2}, {2, 0}}, {}, {});

  std::vector<std::pair<uint64_t, double>> expected = {{0, 0.268}, {1, 0.081}, {2, 0.289}, {7, 0.22}, {4, 0.081}};
  ASSERT_TRUE(mg_test_utility::TestEqualVectorPairs(results, expected, 0.05));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
