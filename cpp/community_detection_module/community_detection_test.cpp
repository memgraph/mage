#include <gtest/gtest.h>

#include <mg_generate.hpp>
#include <mg_test_utils.hpp>

#include "algorithm/louvain.hpp"

TEST(LouvainCommunityDetection, EmptyGraph) {
  auto empty_graph = mg_generate::BuildGraph(0, {});

  auto communities = louvain_alg::GetCommunities(*empty_graph);
  std::vector<std::int64_t> correct_communities{};

  ASSERT_TRUE(communities == correct_communities);
}

TEST(LouvainCommunityDetection, SmallGraph) {
  auto graph = mg_generate::BuildGraph(6, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {1, 3}, {4, 2}});

  auto communities = louvain_alg::GetCommunities(*graph);
  std::vector<std::int64_t> correct_communities{0, 0, 0, 1, 1, 1};

  ASSERT_TRUE(communities == correct_communities);
}

TEST(LouvainCommunityDetection, SmallGraph3Communities) {
  auto graph = mg_generate::BuildGraph(
      9, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}, {6, 7}, {7, 8}, {8, 6}, {6, 0}, {1, 3}, {4, 2}});

  auto communities = louvain_alg::GetCommunities(*graph);
  std::vector<std::int64_t> correct_communities{0, 0, 0, 1, 1, 1, 2, 2, 2};

  ASSERT_TRUE(communities == correct_communities);
}

TEST(LouvainCommunityDetection, TwoComponentGraph) {
  auto graph = mg_generate::BuildGraph(6, {{0, 1}, {1, 2}, {2, 0}, {3, 4}, {4, 5}, {5, 3}});

  auto communities = louvain_alg::GetCommunities(*graph);
  std::vector<std::int64_t> correct_communities{0, 0, 0, 1, 1, 1};

  ASSERT_TRUE(communities == correct_communities);
}

TEST(LouvainCommunityDetection, ComplexGraph) {
  auto graph = mg_generate::BuildGraph(
      16, {{0, 2},  {0, 3},  {0, 4},  {0, 5},   {1, 2},   {1, 4},   {1, 7},   {2, 4},  {2, 5},  {2, 6},
           {3, 7},  {4, 10}, {5, 7},  {5, 11},  {6, 7},   {6, 11},  {8, 9},   {8, 10}, {8, 11}, {8, 14},
           {8, 15}, {9, 12}, {9, 14}, {10, 11}, {10, 12}, {10, 13}, {10, 14}, {11, 13}});

  auto communities = louvain_alg::GetCommunities(*graph);
  std::vector<std::int64_t> correct_communities{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};

  ASSERT_TRUE(communities == correct_communities);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
