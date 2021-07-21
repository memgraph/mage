#pragma once

#include <random>
#include <unordered_set>

namespace pagerank_approx_alg {
namespace {
class PageRankData {
 public:
  void Init() {
    walks.clear();
    walks_counter.clear();
    walks_table.clear();
  }

  bool IsEmpty() { return walks.empty(); }

  std::vector<std::vector<std::uint64_t>> walks;
  std::unordered_map<std::uint64_t, uint64_t> walks_counter;
  std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> walks_table;
};

extern PageRankData context;
extern std::uint64_t global_R;
extern double global_epsilon;

void NormalizeRank(std::vector<double> &rank);

std::vector<double> CalculatePageRank(const mg_graph::GraphView<> &graph);

void CreateRoute(const mg_graph::GraphView<> &graph, const std::uint64_t start_id, std::vector<std::uint64_t> &walk,
                 const std::uint64_t walk_index, const double epsilon, std::uniform_real_distribution<float> distr,
                 std::mt19937 gen);
}  // namespace

std::vector<double> PageRankApprox(const mg_graph::GraphView<> &graph, const std::uint64_t R = 10,
                                   const double epsilon = 0.2);

std::vector<double> UpdateCreate(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, uint64_t> new_edge);

std::vector<double> UpdateCreate(const mg_graph::GraphView<> &graph, const std::uint64_t new_vertex);

std::vector<double> UpdateDelete(const mg_graph::GraphView<> &graph,
                                 const std::pair<std::uint64_t, std::uint64_t> removed_edge);

std::vector<double> UpdateDelete(const mg_graph::GraphView<> &graph, const std::uint64_t removed_vertex);

}  // namespace pagerank_approx_alg
