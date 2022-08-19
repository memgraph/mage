#include <tuple>

#include <gtest/gtest.h>
#include <mg_generate.hpp>
#include <mg_graph.hpp>
#include <mg_test_utils.hpp>

#include "algorithm/degree_centrality.hpp"

class DegreeCentralityTest : public testing::TestWithParam<std::tuple<mg_graph::Graph<>, std::vector<double>>> {};

TEST_P(DegreeCentralityTest, ParametrizedTest) {
  auto graph = std::get<0>(GetParam());
  auto expected = std::get<1>(GetParam());
  auto results = degree_cenntrality_alg::GetDegreeCentrality(graph);
  ASSERT_TRUE(mg_test_utility::TestEqualVectors(results, expected));
}

INSTANTIATE_TEST_SUITE_P(
    DegreeCentrality, DegreeCentralityTest,
    ///
    ///@brief Parametrized test consists out of tuple.
    ///
    testing::Values(
        std::make_tuple(*mg_generate::BuildGraph(0, {}, mg_graph::GraphType::kUndirectedGraph), std::vector<double>{}),
        std::make_tuple(*mg_generate::BuildGraph(5, {{0, 4}, {2, 3}}, mg_graph::GraphType::kUndirectedGraph),
                        std::vector<double>{0.2500, 0.0000, 0.2500, 0.2500, 0.2500}),
        std::make_tuple(
            *mg_generate::BuildGraph(
                10, {{0, 4}, {0, 8}, {1, 5}, {1, 8}, {2, 6}, {3, 5}, {4, 7}, {5, 6}, {5, 8}, {6, 8}, {7, 9}, {8, 9}},
                mg_graph::GraphType::kUndirectedGraph),
            std::vector<double>{0.2222, 0.2222, 0.1111, 0.1111, 0.2222, 0.4444, 0.3333, 0.2222, 0.5556, 0.2222}),
        std::make_tuple(*mg_generate::BuildGraph(15, {{0, 4},  {0, 8},  {0, 13}, {1, 3},   {1, 8},   {1, 13}, {2, 8},
                                                      {2, 11}, {2, 13}, {3, 5},  {3, 8},   {3, 9},   {3, 11}, {3, 13},
                                                      {4, 7},  {4, 8},  {4, 11}, {5, 13},  {6, 7},   {6, 8},  {6, 9},
                                                      {6, 13}, {7, 14}, {8, 10}, {10, 13}, {12, 13}, {13, 14}},
                                                 mg_graph::GraphType::kUndirectedGraph),
                        std::vector<double>{0.2143, 0.2143, 0.2143, 0.4286, 0.2857, 0.1429, 0.2857, 0.2143, 0.5000,
                                            0.1429, 0.1429, 0.2143, 0.0714, 0.6429, 0.1429})));

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}