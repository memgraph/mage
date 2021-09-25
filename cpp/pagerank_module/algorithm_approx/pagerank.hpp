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

    gen = new std::mt19937(42);
    distr = new std::uniform_real_distribution<float>(0.0, 1.0);
  }

  bool IsEmpty() { return walks.empty(); }

  std::vector<std::vector<std::uint64_t>> walks;
  std::unordered_map<std::uint64_t, uint64_t> walks_counter;
  std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> walks_table;
  std::mt19937 *gen;
  std::uniform_real_distribution<float> *distr;
};

extern PageRankData context;
extern std::uint64_t global_R;
extern double global_epsilon;

void NormalizeRank(std::vector<double> &rank);

std::vector<std::pair<std::uint64_t, double>> CalculatePageRank();

void CreateRoute(const mg_graph::GraphView<> &graph, const std::uint64_t start_id, std::vector<std::uint64_t> &walk,
                 const std::uint64_t walk_index, const double epsilon, std::uniform_real_distribution<float> distr,
                 std::mt19937 gen);

void UpdateCreate(const mg_graph::GraphView<> &graph, const std::uint64_t new_vertex);

void UpdateCreate(const mg_graph::GraphView<> &graph, const std::uint64_t new_vertex);

void UpdateDelete(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, std::uint64_t> removed_edge);

void UpdateDelete(const mg_graph::GraphView<> &graph, const std::uint64_t removed_vertex);

}  // namespace

std::vector<std::pair<std::uint64_t, double>> SetPagerank(const mg_graph::GraphView<> &graph,
                                                          const std::uint64_t R = 10, const double epsilon = 0.2);

std::vector<std::pair<std::uint64_t, double>> GetPagerank(const mg_graph::GraphView<> &graph);

std::vector<std::pair<std::uint64_t, double>> UpdatePagerank(
    const mg_graph::GraphView<> &graph, const std::vector<std::uint64_t> new_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> new_edges, const std::vector<std::uint64_t> deleted_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> deleted_edges);

void Reset();

}  // namespace pagerank_approx_alg
