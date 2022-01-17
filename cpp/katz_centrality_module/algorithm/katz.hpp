#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <mg_exceptions.hpp>
#include <mg_graph.hpp>

namespace katz_alg {
namespace {

class KatzCentralityData;

std::uint64_t MaxDegree(const mg_graph::GraphView<> &graph);

bool Converged(std::set<std::uint64_t> &active_nodes, std::uint64_t k, double epsilon);

std::vector<std::pair<std::uint64_t, double>> KatzCentralityLoop(std::set<std::uint64_t> &active_nodes,
                                                                 const mg_graph::GraphView<> &graph, double alpha,
                                                                 std::uint64_t k, double epsilon, double gamma);

void UpdateLevel(KatzCentralityData &context_new, std::set<std::uint64_t> &from_nodes,
                 const std::vector<std::pair<std::uint64_t, std::uint64_t>> &new_edges,
                 const std::vector<std::pair<std::uint64_t, std::uint64_t>> &deleted_edges,
                 const std::set<std::uint64_t> &new_edge_ids, const mg_graph::GraphView<> &graph);
}  // namespace

std::vector<std::pair<std::uint64_t, double>> GetKatzCentrality(const mg_graph::GraphView<> &graph, double alpha = 0.2,
                                                                std::uint64_t k = 3, double epsilon = 1e-2);

std::vector<std::pair<std::uint64_t, double>> UpdateKatz(
    const mg_graph::GraphView<> &graph, const std::vector<std::uint64_t> &new_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &new_edges, const std::set<std::uint64_t> &new_edge_ids,
    const std::vector<std::uint64_t> &deleted_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_edges);
}  // namespace katz_alg
