#include <algorithm>
#include <cmath>
#include <map>
#include <set>

#include <mg_exceptions.hpp>
#include <mg_graph.hpp>

namespace katz_alg {
namespace {

bool Converged(std::vector<std::uint64_t> &active_nodes, std::uint64_t k, double epsilon);

void InitVertexMap(std::unordered_map<std::uint64_t, double> &map, double default_value,
                   const mg_graph::GraphView<> &graph);
}  // namespace

std::vector<std::pair<std::uint64_t, double>> GetKatzCentrality(const mg_graph::GraphView<> &graph, double alpha = 0.1,
                                                                std::uint64_t k = 5, double epsilon = 1e-2);

std::vector<std::pair<std::uint64_t, double>> UpdateKatz(
    const mg_graph::GraphView<> &graph, const std::vector<std::uint64_t> &new_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &new_edges,
    const std::vector<std::uint64_t> &deleted_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_edges);
}  // namespace katz_alg
