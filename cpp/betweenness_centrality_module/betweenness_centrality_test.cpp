#include <gtest/gtest.h>

#include <mg_generate.hpp>

#include "algorithm/betweenness_centrality.hpp"


TEST(BetweennessCentrality, OneNodeZeroEdges) {
  auto graph = mg_generate::BuildGraph(1, {});

  auto results = betweenness_centrality_alg::BetweennessCentralityUnweighted(*graph);
  auto expected = std::vector<double>({{0,1}});
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}