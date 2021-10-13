#include "dynamic_betweenness_centrality.hpp"

#include <mg_generate.hpp>
#include "bcc_utility.hpp"

// TODO: skinuti noviju verziju (gdje GraphView ima fje za dohvacanje inner node IDjeva)
// TODO: promijeniti da argument funkcije bude const mg_graph::GraphView<>
/**
 * @brief A function for retrieving the original node ID
 *
 * @param[in] graph
 *
 * @return An unordered map that maps node IDs to original node IDs from the database
 */
std::unordered_map<uint64_t, uint64_t> dynamic_bc_algorithm::getOriginalNodeIDMapping(
    const mg_graph::GraphView<> &graph) {
  std::unordered_map<uint64_t, uint64_t> nodeMap;

  nodeMap.reserve(graph.Nodes().size());

  for (auto node : graph.Nodes()) {
    uint64_t nodeID = node.id;
    nodeMap[nodeID] = graph.GetInnerNodeId(nodeID);
  }

  return nodeMap;
}

/**
 * @brief A function for initializing the BC map
 *
 * @param[in] graph
 */

void initializeBetweennessCentrality(const mg_graph::GraphView<> &graph) {
  dynamic_bc_algorithm::context.init(graph.Nodes().size());
}

/**
 * @brief A function for getting the betweenness centrality of nodes
 *
 * @return An unordered map containing betweenness centrality of nodes
 */

std::unordered_map<uint64_t, uint64_t> dynamic_bc_algorithm::getBetweennessCentrality(
    const mg_graph::GraphView<> &graph) {
  if (!context.isBCEmpty()) {
    context.init(graph.Nodes().size());
  }

  return context.BC;
}

void dynamic_bc_algorithm::iCentral(const mg_graph::GraphView<> &graph, const uint64_t &firstNode,
                                    const uint64_t &secondNode, const Operation &operation) {
  // TODO: prilagoditi Brandesa da vraca mape, a ne vektore
  std::unordered_map<uint64_t, uint64_t> initialBC =
      const_cast<const std::unordered_map<uint64_t, uint64_t> &>(context.BC);

  if (context.isBCEmpty()) {
    initializeBetweennessCentrality(graph);
  }

  context.BC[0] = 42;

  for (auto el : initialBC) {
    std::cout << el.first << " -> " << el.second << std::endl;
  }

  std::cout << std::endl;

  for (auto el : context.BC) {
    std::cout << el.first << " -> " << el.second << std::endl;
  }
}

/**
 * @brief A function that updates biconnected components and articulation points of a graph
 *
 * @param graph
 * @param biconnectedComponents A vector of biconnected components
 * @param articulationPoints An unordered set of articulation points
 */
void dynamic_bc_algorithm::updateBiconnectedComponentsAndArticulationPoints(
    const mg_graph::GraphView<> &graph, std::vector<std::unordered_set<std::uint64_t>> &biconnectedComponents,
    std::unordered_set<uint64_t> &articulationPoints) {
  biconnectedComponents = bcc_algorithm::GetBiconnectedComponents(graph, articulationPoints);
}

int main(int argc, char const *argv[]) {
  auto graph = mg_generate::BuildGraph(
      20, {{8, 9},   {8, 10},  {9, 0},   {10, 0},  {12, 11}, {12, 13}, {11, 5}, {13, 5}, {17, 15},
           {17, 19}, {15, 16}, {19, 16}, {16, 14}, {16, 18}, {14, 4},  {18, 4}, {0, 1},  {1, 2},
           {0, 3},   {2, 3},   {2, 4},   {3, 5},   {5, 6},   {6, 7},   {7, 4}},
      mg_graph::GraphType::kUndirectedGraph);

  // graph->CreateEdge(3, 7);

  uint64_t from = 3;
  uint64_t to = 7;

  dynamic_bc_algorithm::iCentral(*graph, from, to, dynamic_bc_algorithm::Operation::INSERTION);
}