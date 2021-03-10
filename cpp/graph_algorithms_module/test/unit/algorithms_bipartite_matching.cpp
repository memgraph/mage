#include <glog/logging.h>
#include <gtest/gtest.h>

#include "algorithms/algorithms.hpp"
#include "utils.hpp"

TEST(BipartiteMatching, RandomCompleteBipartiteGraphs) {
  auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  for (int t = 0; t < 200; ++t) {
    std::uniform_int_distribution<size_t> dist(1, 250);
    size_t size_a = dist(rng), size_b = dist(rng);
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (int i = 1; i <= size_a; ++i)
      for (int j = 1; j <= size_b; ++j) edges.emplace_back(i, j);
    size_t max_match = algorithms::BipartiteMatching(edges);
    ASSERT_EQ(max_match, std::min(size_a, size_b));
  }
}

TEST(BipartiteMatching, NotBipartiteGraph) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();

  g.CreateEdge(n1, n2);
  g.CreateEdge(n1, n3);
  g.CreateEdge(n1, n4);
  g.CreateEdge(n1, n5);
  g.CreateEdge(n2, n3);
  g.CreateEdge(n2, n4);
  g.CreateEdge(n2, n5);
  g.CreateEdge(n3, n4);
  g.CreateEdge(n3, n5);
  g.CreateEdge(n4, n5);

  bool is_graph_bipartite = algorithms::IsGraphBipartite(g);
  ASSERT_FALSE(is_graph_bipartite);
}

TEST(BipartiteMatching, BipartiteGraph) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();

  g.CreateEdge(n1, n2);
  g.CreateEdge(n2, n3);
  g.CreateEdge(n3, n4);
  g.CreateEdge(n4, n1);

  bool is_graph_bipartite = algorithms::IsGraphBipartite(g);
  ASSERT_TRUE(is_graph_bipartite);
}

TEST(BipartiteMatching, BipartiteGraphWith2Components) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();

  g.CreateEdge(n1, n2);
  g.CreateEdge(n2, n3);
  g.CreateEdge(n4, n5);

  bool is_graph_bipartite = algorithms::IsGraphBipartite(g);
  ASSERT_TRUE(is_graph_bipartite);
}

TEST(BipartiteMatching, NotBipartiteGraphWithSelfLoop) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();

  g.CreateEdge(n1, n2);
  g.CreateEdge(n2, n3);
  g.CreateEdge(n3, n4);
  g.CreateEdge(n1, n1);

  bool is_graph_bipartite = algorithms::IsGraphBipartite(g);
  ASSERT_FALSE(is_graph_bipartite);
}

TEST(BipartiteMatching, Handmade1) {
  size_t size_a = 6, size_b = 6;
  std::vector<std::pair<uint32_t, uint32_t>> edges = {
      {1, 2}, {1, 3}, {2, 1}, {2, 4}, {3, 3}, {4, 3}, {4, 4}, {5, 5}};
  size_t max_match = algorithms::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, HandmadeGraph1) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();
  uint32_t n6 = g.CreateNode();
  uint32_t n7 = g.CreateNode();
  uint32_t n8 = g.CreateNode();
  uint32_t n9 = g.CreateNode();
  uint32_t n10 = g.CreateNode();

  g.CreateEdge(n1, n7);
  g.CreateEdge(n1, n8);
  g.CreateEdge(n2, n6);
  g.CreateEdge(n2, n9);
  g.CreateEdge(n3, n8);
  g.CreateEdge(n4, n8);
  g.CreateEdge(n4, n9);
  g.CreateEdge(n5, n10);

  size_t max_match = algorithms::BipartiteMatching(g);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, Handmade2) {
  size_t size_a = 5, size_b = 4;
  std::vector<std::pair<uint32_t, uint32_t>> edges = {{5, 2}, {1, 2}, {4, 3},
                                                      {3, 1}, {2, 2}, {4, 4}};
  size_t max_match = algorithms::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 3);
}

TEST(BipartiteMatching, HandmadeGraph2) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();
  uint32_t n6 = g.CreateNode();
  uint32_t n7 = g.CreateNode();
  uint32_t n8 = g.CreateNode();
  uint32_t n9 = g.CreateNode();

  g.CreateEdge(n5, n7);
  g.CreateEdge(n1, n7);
  g.CreateEdge(n4, n8);
  g.CreateEdge(n3, n6);
  g.CreateEdge(n2, n7);
  g.CreateEdge(n4, n9);

  size_t max_match = algorithms::BipartiteMatching(g);
  ASSERT_EQ(max_match, 3);
}

TEST(BipartiteMatching, Handmade3) {
  size_t size_a = 5, size_b = 5;
  std::vector<std::pair<uint32_t, uint32_t>> edges = {
      {1, 2}, {1, 5}, {2, 3}, {3, 2}, {4, 3}, {4, 4}, {5, 1}};
  size_t max_match = algorithms::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, HandmadeGraph3) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();
  uint32_t n6 = g.CreateNode();
  uint32_t n7 = g.CreateNode();
  uint32_t n8 = g.CreateNode();
  uint32_t n9 = g.CreateNode();
  uint32_t n10 = g.CreateNode();

  g.CreateEdge(n1, n7);
  g.CreateEdge(n1, n10);
  g.CreateEdge(n2, n8);
  g.CreateEdge(n3, n7);
  g.CreateEdge(n4, n8);
  g.CreateEdge(n4, n9);
  g.CreateEdge(n5, n6);

  size_t max_match = algorithms::BipartiteMatching(g);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, Handmade4) {
  size_t size_a = 10, size_b = 10;
  std::vector<std::pair<uint32_t, uint32_t>> edges = {
      {1, 4}, {1, 10}, {2, 6}, {2, 8}, {2, 9},  {3, 9},  {4, 6}, {4, 8},
      {5, 1}, {5, 3},  {5, 9}, {6, 1}, {6, 5},  {6, 7},  {7, 4}, {7, 7},
      {8, 2}, {8, 8},  {9, 3}, {9, 5}, {10, 1}, {10, 2}, {10, 7}};
  size_t max_match = algorithms::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 10);
}

TEST(BipartiteMatching, HandmadeGraph4) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();
  uint32_t n6 = g.CreateNode();
  uint32_t n7 = g.CreateNode();
  uint32_t n8 = g.CreateNode();
  uint32_t n9 = g.CreateNode();
  uint32_t n10 = g.CreateNode();
  uint32_t n11 = g.CreateNode();
  uint32_t n12 = g.CreateNode();
  uint32_t n13 = g.CreateNode();
  uint32_t n14 = g.CreateNode();
  uint32_t n15 = g.CreateNode();
  uint32_t n16 = g.CreateNode();
  uint32_t n17 = g.CreateNode();
  uint32_t n18 = g.CreateNode();
  uint32_t n19 = g.CreateNode();
  uint32_t n20 = g.CreateNode();

  g.CreateEdge(n1, n14);
  g.CreateEdge(n1, n20);
  g.CreateEdge(n2, n16);
  g.CreateEdge(n2, n18);
  g.CreateEdge(n2, n19);
  g.CreateEdge(n3, n19);
  g.CreateEdge(n4, n16);
  g.CreateEdge(n4, n18);
  g.CreateEdge(n5, n11);
  g.CreateEdge(n5, n13);
  g.CreateEdge(n5, n19);
  g.CreateEdge(n6, n11);
  g.CreateEdge(n6, n15);
  g.CreateEdge(n6, n17);
  g.CreateEdge(n7, n14);
  g.CreateEdge(n7, n17);
  g.CreateEdge(n8, n12);
  g.CreateEdge(n8, n18);
  g.CreateEdge(n9, n13);
  g.CreateEdge(n9, n15);
  g.CreateEdge(n10, n11);
  g.CreateEdge(n10, n12);
  g.CreateEdge(n10, n17);

  size_t max_match = algorithms::BipartiteMatching(g);
  ASSERT_EQ(max_match, 10);
}

TEST(BipartiteMatching, Handmade5) {
  size_t size_a = 10, size_b = 10;
  std::vector<std::pair<uint32_t, uint32_t>> edges = {
      {1, 7}, {2, 1}, {2, 6}, {2, 8}, {3, 4},  {3, 5},  {3, 6},  {3, 7}, {4, 2},
      {4, 4}, {4, 5}, {5, 5}, {5, 6}, {6, 3},  {6, 4},  {6, 7},  {7, 5}, {7, 7},
      {8, 4}, {8, 5}, {9, 3}, {9, 6}, {9, 10}, {10, 1}, {10, 8}, {10, 9}};
  size_t max_match = algorithms::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 9);
}

TEST(BipartiteMatching, HandmadeGraph5) {
  graphdata::Graph g;
  uint32_t n1 = g.CreateNode();
  uint32_t n2 = g.CreateNode();
  uint32_t n3 = g.CreateNode();
  uint32_t n4 = g.CreateNode();
  uint32_t n5 = g.CreateNode();
  uint32_t n6 = g.CreateNode();
  uint32_t n7 = g.CreateNode();
  uint32_t n8 = g.CreateNode();
  uint32_t n9 = g.CreateNode();
  uint32_t n10 = g.CreateNode();
  uint32_t n11 = g.CreateNode();
  uint32_t n12 = g.CreateNode();
  uint32_t n13 = g.CreateNode();
  uint32_t n14 = g.CreateNode();
  uint32_t n15 = g.CreateNode();
  uint32_t n16 = g.CreateNode();
  uint32_t n17 = g.CreateNode();
  uint32_t n18 = g.CreateNode();
  uint32_t n19 = g.CreateNode();
  uint32_t n20 = g.CreateNode();

  g.CreateEdge(n1, n17);
  g.CreateEdge(n2, n11);
  g.CreateEdge(n2, n16);
  g.CreateEdge(n2, n18);
  g.CreateEdge(n3, n14);
  g.CreateEdge(n3, n15);
  g.CreateEdge(n3, n16);
  g.CreateEdge(n3, n17);
  g.CreateEdge(n4, n12);
  g.CreateEdge(n4, n14);
  g.CreateEdge(n4, n15);
  g.CreateEdge(n5, n15);
  g.CreateEdge(n5, n16);
  g.CreateEdge(n6, n13);
  g.CreateEdge(n6, n14);
  g.CreateEdge(n6, n17);
  g.CreateEdge(n7, n15);
  g.CreateEdge(n7, n17);
  g.CreateEdge(n8, n14);
  g.CreateEdge(n8, n15);
  g.CreateEdge(n9, n13);
  g.CreateEdge(n9, n16);
  g.CreateEdge(n9, n20);
  g.CreateEdge(n10, n11);
  g.CreateEdge(n10, n18);
  g.CreateEdge(n10, n19);

  size_t max_match = algorithms::BipartiteMatching(g);
  ASSERT_EQ(max_match, 9);
}

TEST(BipartiteMatching, Performance) {
  auto seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  int n = 250;
  std::set<std::pair<uint32_t, uint32_t>> edges;
  for (int i = 0; i < n * n / 5; ++i) {
    std::uniform_int_distribution<size_t> dist(1, n);
    int a = dist(rng), b = dist(rng);
    edges.insert({a, b});
  }

  Timer timer;
  size_t max_match = algorithms::BipartiteMatching(
      std::vector<std::pair<uint32_t, uint32_t>>(edges.begin(), edges.end()));
  auto time_elapsed = timer.Elapsed();
  ASSERT_TRUE(timer.Elapsed() < std::chrono::seconds(1));
}
