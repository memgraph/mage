#pragma once

#include <unordered_set>

namespace pagerank_approx_util {

void NormalizeRank(std::vector<double> &rank);

}  // namespace pagerank_approx_util

namespace pagerank_approx_alg {

class PageRankData {
 public:
  void Init() {
    walks.clear();
    walks_tracker.clear();
    walks_table.clear();
  }

  std::vector<std::vector<std::uint64_t>> walks;
  std::unordered_map<std::uint64_t, uint64_t> walks_tracker;
  std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> walks_table;
};

std::vector<double> PageRankApprox(const mg_graph::GraphView<> &graph, const std::uint64_t R, const double epsilon);

std::vector<double> Update(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, uint64_t> new_edge);

extern PageRankData context;
extern std::uint64_t global_R;
extern double global_epsilon;

}  // namespace pagerank_approx_alg
