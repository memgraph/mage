#pragma once

#include <stack>
#include <vector>

#include <mg_graph.hpp>
#include <unordered_set>

namespace dynamic_bc_algorithm {

enum class Operation { INSERTION, DELETION };

class BetweennessCentralityData {
 public:
  /*--- Constructors ---*/
  BetweennessCentralityData() = default;

  void init(uint64_t numberOfNodes) {
    if (!BC.empty()) {
      BC.clear();
    }

    BC.reserve(numberOfNodes);
  };

  bool isBCEmpty() { return BC.empty(); }

  /*--- Members ---*/
  std::unordered_map<uint64_t, uint64_t> BC;
  std::unordered_set<uint64_t> articulationPoints;
  std::vector<std::unordered_set<std::uint64_t>> biconnectedComponents;
};

extern BetweennessCentralityData context;

void iCentral(const mg_graph::GraphView<> &graph, const uint64_t &firstNode, const uint64_t &secondNode,
              const Operation &operation);
std::unordered_map<uint64_t, uint64_t> getBetweennessCentrality(const mg_graph::GraphView<> &graph);
std::unordered_map<uint64_t, uint64_t> getOriginalNodeIDMapping(const mg_graph::GraphView<> &graph);
void updateBiconnectedComponentsAndArticulationPoints(
    const mg_graph::GraphView<> &graph, std::vector<std::unordered_set<std::uint64_t>> &biconnectedComponents,
    std::unordered_set<uint64_t> &articulationPoints);
}  // namespace dynamic_bc_algorithm
