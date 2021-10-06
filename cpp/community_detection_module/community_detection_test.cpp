#include <gtest/gtest.h>

#include <mg_generate.hpp>

#include "algorithm/label_propagation.hpp"

TEST(LabelRankT, EmptyGraph) {
  auto empty_graph = mg_generate::BuildGraph(0, {});

  LabelRankT::LabelRankT algorithm = LabelRankT::LabelRankT(empty_graph);
  auto labels = algorithm.calculate_labels();

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {};

  ASSERT_TRUE(labels == correct_labels);
}

TEST(LabelRankT, SmallGraph) {
  auto small_graph = mg_generate::BuildGraph(
      6, {{0, 1}, {0, 2}, {1, 2}, {2, 3}, {3, 4}, {3, 5}, {4, 5}});

  LabelRankT::LabelRankT algorithm = LabelRankT::LabelRankT(small_graph);
  auto labels = algorithm.calculate_labels();

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {
      {0, 2}, {1, 2}, {2, 2}, {3, 3}, {4, 3}, {5, 3}};

  ASSERT_TRUE(labels == correct_labels);
}

TEST(LabelRankT, DisconnectedGraph) {
  auto disconnected_graph = mg_generate::BuildGraph(
      16, {
              {0, 1},   {0, 2},   {0, 3},   {1, 2},   {1, 4},   {2, 3},
              {2, 9},   {3, 13},  {4, 5},   {4, 6},   {4, 7},   {4, 8},
              {5, 7},   {5, 8},   {6, 7},   {6, 8},

              {8, 10},  {9, 10},  {9, 12},  {9, 13},  {9, 14},  {10, 11},
              {10, 13}, {10, 14}, {11, 12}, {11, 13}, {11, 14}, {12, 14},

          });

  LabelRankT::LabelRankT algorithm = LabelRankT::LabelRankT(disconnected_graph);
  auto labels = algorithm.calculate_labels();

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {
      {0, 2},   {1, 2},   {2, 2},   {3, 2},  {4, 4},   {5, 4},
      {6, 4},   {7, 4},   {8, 4},   {9, 10}, {10, 10}, {11, 10},
      {12, 10}, {13, 10}, {14, 10}, {15, 15}};

  ASSERT_TRUE(labels == correct_labels);
}

TEST(LabelRankT, GetLabels) {
  auto example_graph = mg_generate::BuildGraph(
      15, {
              {0, 1},   {0, 2},   {0, 3},   {1, 2},   {1, 4},   {2, 3},
              {2, 9},   {3, 13},  {4, 5},   {4, 6},   {4, 7},   {4, 8},
              {5, 7},   {5, 8},   {6, 7},   {6, 8},

              {8, 10},  {9, 10},  {9, 12},  {9, 13},  {9, 14},  {10, 11},
              {10, 13}, {10, 14}, {11, 12}, {11, 13}, {11, 14}, {12, 14},

          });

  LabelRankT::LabelRankT graph = LabelRankT::LabelRankT(example_graph);
  graph.calculate_labels();
  auto labels = graph.get_labels();

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {
      {0, 2}, {1, 2},  {2, 2},   {3, 2},   {4, 4},   {5, 4},   {6, 4},  {7, 4},
      {8, 4}, {9, 10}, {10, 10}, {11, 10}, {12, 10}, {13, 10}, {14, 10}};

  ASSERT_TRUE(labels == correct_labels);
}

TEST(LabelRankT, GetLabelsUninitialized) {
  auto example_graph = mg_generate::BuildGraph(
      15, {
              {0, 1},   {0, 2},   {0, 3},   {1, 2},   {1, 4},   {2, 3},
              {2, 9},   {3, 13},  {4, 5},   {4, 6},   {4, 7},   {4, 8},
              {5, 7},   {5, 8},   {6, 7},   {6, 8},

              {8, 10},  {9, 10},  {9, 12},  {9, 13},  {9, 14},  {10, 11},
              {10, 13}, {10, 14}, {11, 12}, {11, 13}, {11, 14}, {12, 14},

          });

  LabelRankT::LabelRankT graph = LabelRankT::LabelRankT(example_graph);
  auto labels = graph.update_labels({}, {{}}, {}, {{}});

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {
      {0, 2}, {1, 2},  {2, 2},   {3, 2},   {4, 4},   {5, 4},   {6, 4},  {7, 4},
      {8, 4}, {9, 10}, {10, 10}, {11, 10}, {12, 10}, {13, 10}, {14, 10}};

  ASSERT_TRUE(labels == correct_labels);
}

TEST(LabelRankT, FindLabels) {
  auto example_graph = mg_generate::BuildGraph(
      15, {
              {0, 1},   {0, 2},   {0, 3},   {1, 2},   {1, 4},   {2, 3},
              {2, 9},   {3, 13},  {4, 5},   {4, 6},   {4, 7},   {4, 8},
              {5, 7},   {5, 8},   {6, 7},   {6, 8},

              {8, 10},  {9, 10},  {9, 12},  {9, 13},  {9, 14},  {10, 11},
              {10, 13}, {10, 14}, {11, 12}, {11, 13}, {11, 14}, {12, 14},

          });

  LabelRankT::LabelRankT graph = LabelRankT::LabelRankT(example_graph);
  auto labels = graph.calculate_labels();

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {
      {0, 2}, {1, 2},  {2, 2},   {3, 2},   {4, 4},   {5, 4},   {6, 4},  {7, 4},
      {8, 4}, {9, 10}, {10, 10}, {11, 10}, {12, 10}, {13, 10}, {14, 10}};

  ASSERT_TRUE(labels == correct_labels);
}

TEST(LabelRankT, UpdateLabelsEdgesChanged) {
  auto example_graph = mg_generate::BuildGraph(
      15, {
              {0, 1},   {0, 2},   {0, 3},   {1, 2},   {1, 4},   {2, 3},
              {2, 9},   {3, 13},  {4, 5},   {4, 6},   {4, 7},   {4, 8},
              {5, 7},   {5, 8},   {6, 7},   {6, 8},

              {8, 10},  {9, 10},  {9, 12},  {9, 13},  {9, 14},  {10, 11},
              {10, 13}, {10, 14}, {11, 12}, {11, 13}, {11, 14}, {12, 14},

          });

  LabelRankT::LabelRankT graph = LabelRankT::LabelRankT(example_graph);
  auto labels = graph.calculate_labels();

  example_graph->EraseEdge(9, 12);
  example_graph->EraseEdge(9, 14);
  example_graph->EraseEdge(10, 13);

  example_graph->CreateEdge(0, 13, mg_graph::GraphType::kUndirectedGraph);
  example_graph->CreateEdge(3, 9, mg_graph::GraphType::kUndirectedGraph);
  example_graph->CreateEdge(10, 12, mg_graph::GraphType::kUndirectedGraph);

  labels = graph.update_labels({}, {{0, 13}, {3, 9}, {10, 12}}, {},
                               {{9, 12}, {9, 14}, {10, 13}});

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {
      {0, 2}, {1, 2}, {2, 2},   {3, 2},   {4, 4},   {5, 4},  {6, 4},  {7, 4},
      {8, 4}, {9, 2}, {10, 10}, {11, 10}, {12, 10}, {13, 2}, {14, 10}};

  ASSERT_TRUE(labels == correct_labels);
}

TEST(LabelRankT, UpdateLabelsNodesChanged) {
  auto example_graph = mg_generate::BuildGraph(
      15, {
              {0, 1},   {0, 2},   {0, 3},   {1, 2},   {1, 4},   {2, 3},
              {2, 9},   {3, 13},  {4, 5},   {4, 6},   {4, 7},   {4, 8},
              {5, 7},   {5, 8},   {6, 7},   {6, 8},

              {8, 10},  {9, 10},  {9, 12},  {9, 13},  {9, 14},  {10, 11},
              {10, 13}, {10, 14}, {11, 12}, {11, 13}, {11, 14}, {12, 14},

          });

  LabelRankT::LabelRankT graph = LabelRankT::LabelRankT(example_graph);
  auto labels = graph.calculate_labels();

  example_graph->EraseEdge(9, 12);
  example_graph->EraseEdge(9, 14);
  example_graph->EraseEdge(10, 13);
  
  example_graph->CreateEdge(0, 13, mg_graph::GraphType::kUndirectedGraph);
  example_graph->CreateEdge(3, 9, mg_graph::GraphType::kUndirectedGraph);
  example_graph->CreateEdge(10, 12, mg_graph::GraphType::kUndirectedGraph);

  labels = graph.update_labels({}, {{0, 13}, {3, 9}, {10, 12}}, {},
                               {{9, 12}, {9, 14}, {10, 13}});

  example_graph = mg_generate::BuildGraph(
      16, {{0, 1},   {0, 2},   {0, 3},   {0, 13},  {1, 2},   {1, 4},
           {1, 7},   {2, 3},   {2, 4},   {2, 6},   {2, 9},   {3, 9},
           {3, 13},  {4, 6},   {4, 7},   {4, 8},

           {6, 7},

           {8, 10},  {8, 15},  {9, 10},  {9, 13},  {10, 11}, {10, 12},
           {10, 14}, {10, 15}, {11, 12}, {11, 13}, {11, 14}, {12, 14},

           {14, 15}});

  labels = graph.update_labels(
      {15}, {{8, 15}, {10, 15}, {14, 15}, {1, 7}, {2, 4}, {2, 6}}, {5},
      {{4, 5}, {5, 7}, {5, 8}, {6, 8}});

  std::unordered_map<std::uint64_t, std::uint64_t> correct_labels = {
      {0, 2}, {1, 2},   {2, 2},   {3, 2},   {4, 2},  {6, 2},   {7, 2},  {8, 10},
      {9, 2}, {10, 10}, {11, 10}, {12, 10}, {13, 2}, {14, 10}, {15, 10}};

  ASSERT_TRUE(labels == correct_labels);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
