#include <algorithm>
#include <cmath>
#include <map>

#include <mg_exceptions.hpp>
#include <mg_graph.hpp>

namespace katz_alg {
namespace {
bool Converged(std::vector<std::uint64_t> &active_nodes, std::uint64_t k, double epsilon,
               const std::unordered_map<std::uint64_t, double> &centrality,
               const std::unordered_map<std::uint64_t, double> &lr,
               const std::unordered_map<std::uint64_t, double> &ur);

std::uint64_t MaxDegree(const mg_graph::GraphView<> &graph);

void InitVertexMap(std::unordered_map<std::uint64_t, double> &map, double default_value,
                   const mg_graph::GraphView<> &graph);
}  // namespace
std::vector<std::pair<std::uint64_t, double>> GetKatzCentrality(const mg_graph::GraphView<> &graph);
}  // namespace katz_alg
