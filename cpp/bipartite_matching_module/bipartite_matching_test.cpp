#include <chrono>
#include <random>

#include <gtest/gtest.h>
#include <mg_graph.hpp>

#include "algorithm/bipartite_matching.hpp"

/// This class is threadsafe
class Timer {
 public:
  Timer() : start_time_(std::chrono::steady_clock::now()) {}

  template <typename TDuration = std::chrono::duration<double>>
  TDuration Elapsed() const {
    return std::chrono::duration_cast<TDuration>(std::chrono::steady_clock::now() - start_time_);
  }

 private:
  std::chrono::steady_clock::time_point start_time_;
};

TEST(BipartiteMatching, RandomCompleteBipartiteGraphs) {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  for (int t = 0; t < 200; ++t) {
    std::uniform_int_distribution<> dist(1, 250);
    auto size_a = dist(rng), size_b = dist(rng);
    std::vector<std::pair<uint64_t, uint64_t>> edges;

    for (uint64_t i = 1; i <= size_a; ++i)
      for (uint64_t j = 1; j <= size_b; ++j) edges.emplace_back(i, j);

    auto max_match = bipartite_matching_alg::BipartiteMatching(edges);
    ASSERT_EQ(max_match, std::min(size_a, size_b));
  }
}

TEST(BipartiteMatching, NotBipartiteGraph) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  ;
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);
  auto n5 = g->CreateNode(4);

  g->CreateEdge(n1, n2);
  g->CreateEdge(n1, n3);
  g->CreateEdge(n1, n4);
  g->CreateEdge(n1, n5);
  g->CreateEdge(n2, n3);
  g->CreateEdge(n2, n4);
  g->CreateEdge(n2, n5);
  g->CreateEdge(n3, n4);
  g->CreateEdge(n3, n5);
  g->CreateEdge(n4, n5);

  auto is_graph_bipartite = bipartite_matching_alg::IsGraphBipartite(g);
  ASSERT_FALSE(is_graph_bipartite);
}

TEST(BipartiteMatching, BipartiteGraph) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  ;
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);

  g->CreateEdge(n1, n2);
  g->CreateEdge(n2, n3);
  g->CreateEdge(n3, n4);
  g->CreateEdge(n4, n1);

  auto is_graph_bipartite = bipartite_matching_alg::IsGraphBipartite(g);
  ASSERT_TRUE(is_graph_bipartite);
}

TEST(BipartiteMatching, BipartiteGraphWith2Components) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  ;
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);
  auto n5 = g->CreateNode(4);

  g->CreateEdge(n1, n2);
  g->CreateEdge(n2, n3);
  g->CreateEdge(n4, n5);

  auto is_graph_bipartite = bipartite_matching_alg::IsGraphBipartite(g);
  ASSERT_TRUE(is_graph_bipartite);
}

TEST(BipartiteMatching, NotBipartiteGraphWithSelfLoop) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  ;
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);

  g->CreateEdge(n1, n2);
  g->CreateEdge(n2, n3);
  g->CreateEdge(n3, n4);
  g->CreateEdge(n1, n1);

  auto is_graph_bipartite = bipartite_matching_alg::IsGraphBipartite(g);
  ASSERT_FALSE(is_graph_bipartite);
}

TEST(BipartiteMatching, Handmade1) {
  std::vector<std::pair<uint64_t, uint64_t>> edges = {{1, 2}, {1, 3}, {2, 1}, {2, 4}, {3, 3}, {4, 3}, {4, 4}, {5, 5}};
  auto max_match = bipartite_matching_alg::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, HandmadeGraph1) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);
  auto n5 = g->CreateNode(4);
  auto n6 = g->CreateNode(5);
  auto n7 = g->CreateNode(6);
  auto n8 = g->CreateNode(7);
  auto n9 = g->CreateNode(8);
  auto n10 = g->CreateNode(9);

  g->CreateEdge(n1, n7);
  g->CreateEdge(n1, n8);
  g->CreateEdge(n2, n6);
  g->CreateEdge(n2, n9);
  g->CreateEdge(n3, n8);
  g->CreateEdge(n4, n8);
  g->CreateEdge(n4, n9);
  g->CreateEdge(n5, n10);

  auto max_match = bipartite_matching_alg::BipartiteMatching(g);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, Handmade2) {
  std::vector<std::pair<uint64_t, uint64_t>> edges = {{5, 2}, {1, 2}, {4, 3}, {3, 1}, {2, 2}, {4, 4}};
  auto max_match = bipartite_matching_alg::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 3);
}

TEST(BipartiteMatching, HandmadeGraph2) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);
  auto n5 = g->CreateNode(4);
  auto n6 = g->CreateNode(5);
  auto n7 = g->CreateNode(6);
  auto n8 = g->CreateNode(7);
  auto n9 = g->CreateNode(8);

  g->CreateEdge(n5, n7);
  g->CreateEdge(n1, n7);
  g->CreateEdge(n4, n8);
  g->CreateEdge(n3, n6);
  g->CreateEdge(n2, n7);
  g->CreateEdge(n4, n9);

  auto max_match = bipartite_matching_alg::BipartiteMatching(g);
  ASSERT_EQ(max_match, 3);
}

TEST(BipartiteMatching, Handmade3) {
  std::vector<std::pair<uint64_t, uint64_t>> edges = {{1, 2}, {1, 5}, {2, 3}, {3, 2}, {4, 3}, {4, 4}, {5, 1}};
  auto max_match = bipartite_matching_alg::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, HandmadeGraph3) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);
  auto n5 = g->CreateNode(4);
  auto n6 = g->CreateNode(5);
  auto n7 = g->CreateNode(6);
  auto n8 = g->CreateNode(7);
  auto n9 = g->CreateNode(8);
  auto n10 = g->CreateNode(9);

  g->CreateEdge(n1, n7);
  g->CreateEdge(n1, n10);
  g->CreateEdge(n2, n8);
  g->CreateEdge(n3, n7);
  g->CreateEdge(n4, n8);
  g->CreateEdge(n4, n9);
  g->CreateEdge(n5, n6);

  auto max_match = bipartite_matching_alg::BipartiteMatching(g);
  ASSERT_EQ(max_match, 5);
}

TEST(BipartiteMatching, Handmade4) {
  std::vector<std::pair<uint64_t, uint64_t>> edges = {{1, 4}, {1, 10}, {2, 6}, {2, 8}, {2, 9},  {3, 9},  {4, 6}, {4, 8},
                                                      {5, 1}, {5, 3},  {5, 9}, {6, 1}, {6, 5},  {6, 7},  {7, 4}, {7, 7},
                                                      {8, 2}, {8, 8},  {9, 3}, {9, 5}, {10, 1}, {10, 2}, {10, 7}};
  auto max_match = bipartite_matching_alg::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 10);
}

TEST(BipartiteMatching, HandmadeGraph4) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);
  auto n5 = g->CreateNode(4);
  auto n6 = g->CreateNode(5);
  auto n7 = g->CreateNode(6);
  auto n8 = g->CreateNode(7);
  auto n9 = g->CreateNode(8);
  auto n10 = g->CreateNode(9);
  auto n11 = g->CreateNode(10);
  auto n12 = g->CreateNode(11);
  auto n13 = g->CreateNode(12);
  auto n14 = g->CreateNode(13);
  auto n15 = g->CreateNode(14);
  auto n16 = g->CreateNode(15);
  auto n17 = g->CreateNode(16);
  auto n18 = g->CreateNode(17);
  auto n19 = g->CreateNode(18);
  auto n20 = g->CreateNode(19);

  g->CreateEdge(n1, n14);
  g->CreateEdge(n1, n20);
  g->CreateEdge(n2, n16);
  g->CreateEdge(n2, n18);
  g->CreateEdge(n2, n19);
  g->CreateEdge(n3, n19);
  g->CreateEdge(n4, n16);
  g->CreateEdge(n4, n18);
  g->CreateEdge(n5, n11);
  g->CreateEdge(n5, n13);
  g->CreateEdge(n5, n19);
  g->CreateEdge(n6, n11);
  g->CreateEdge(n6, n15);
  g->CreateEdge(n6, n17);
  g->CreateEdge(n7, n14);
  g->CreateEdge(n7, n17);
  g->CreateEdge(n8, n12);
  g->CreateEdge(n8, n18);
  g->CreateEdge(n9, n13);
  g->CreateEdge(n9, n15);
  g->CreateEdge(n10, n11);
  g->CreateEdge(n10, n12);
  g->CreateEdge(n10, n17);

  auto max_match = bipartite_matching_alg::BipartiteMatching(g);
  ASSERT_EQ(max_match, 10);
}

TEST(BipartiteMatching, Handmade5) {
  std::vector<std::pair<uint64_t, uint64_t>> edges = {
      {1, 7}, {2, 1}, {2, 6}, {2, 8}, {3, 4}, {3, 5}, {3, 6}, {3, 7}, {4, 2}, {4, 4},  {4, 5},  {5, 5},  {5, 6},
      {6, 3}, {6, 4}, {6, 7}, {7, 5}, {7, 7}, {8, 4}, {8, 5}, {9, 3}, {9, 6}, {9, 10}, {10, 1}, {10, 8}, {10, 9}};
  auto max_match = bipartite_matching_alg::BipartiteMatching(edges);
  ASSERT_EQ(max_match, 9);
}

TEST(BipartiteMatching, HandmadeGraph5) {
  mg_graph::Graph<> *g = new mg_graph::Graph<>();
  auto n1 = g->CreateNode(0);
  auto n2 = g->CreateNode(1);
  auto n3 = g->CreateNode(2);
  auto n4 = g->CreateNode(3);
  auto n5 = g->CreateNode(4);
  auto n6 = g->CreateNode(5);
  auto n7 = g->CreateNode(6);
  auto n8 = g->CreateNode(7);
  auto n9 = g->CreateNode(8);
  auto n10 = g->CreateNode(9);
  auto n11 = g->CreateNode(10);
  auto n12 = g->CreateNode(11);
  auto n13 = g->CreateNode(12);
  auto n14 = g->CreateNode(13);
  auto n15 = g->CreateNode(14);
  auto n16 = g->CreateNode(15);
  auto n17 = g->CreateNode(16);
  auto n18 = g->CreateNode(17);
  auto n19 = g->CreateNode(18);
  auto n20 = g->CreateNode(19);

  g->CreateEdge(n1, n17);
  g->CreateEdge(n2, n11);
  g->CreateEdge(n2, n16);
  g->CreateEdge(n2, n18);
  g->CreateEdge(n3, n14);
  g->CreateEdge(n3, n15);
  g->CreateEdge(n3, n16);
  g->CreateEdge(n3, n17);
  g->CreateEdge(n4, n12);
  g->CreateEdge(n4, n14);
  g->CreateEdge(n4, n15);
  g->CreateEdge(n5, n15);
  g->CreateEdge(n5, n16);
  g->CreateEdge(n6, n13);
  g->CreateEdge(n6, n14);
  g->CreateEdge(n6, n17);
  g->CreateEdge(n7, n15);
  g->CreateEdge(n7, n17);
  g->CreateEdge(n8, n14);
  g->CreateEdge(n8, n15);
  g->CreateEdge(n9, n13);
  g->CreateEdge(n9, n16);
  g->CreateEdge(n9, n20);
  g->CreateEdge(n10, n11);
  g->CreateEdge(n10, n18);
  g->CreateEdge(n10, n19);

  auto max_match = bipartite_matching_alg::BipartiteMatching(g);
  ASSERT_EQ(max_match, 9);
}

TEST(BipartiteMatching, Performance) {
  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  int n = 250;
  std::set<std::pair<uint64_t, uint64_t>> edges;
  for (int i = 0; i < n * n / 5; ++i) {
    std::uniform_int_distribution<> dist(1, n);
    int a = dist(rng), b = dist(rng);
    edges.insert({a, b});
  }

  Timer timer;
  bipartite_matching_alg::BipartiteMatching(std::vector<std::pair<uint64_t, uint64_t>>(edges.begin(), edges.end()));
  auto time_elapsed = timer.Elapsed();
  ASSERT_TRUE(timer.Elapsed() < std::chrono::seconds(1));
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}