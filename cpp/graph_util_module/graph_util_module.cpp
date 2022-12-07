#include <mgp.hpp>

#include "algorithms/ancestors.hpp"
#include "algorithms/connect_nodes.hpp"
#include "algorithms/descendants.hpp"
#include "algorithms/topological_sort.hpp"

const char *kProcedureAncestors = "ancestors";
const char *kProcedureConnectNodes = "connect_nodes";
const char *kProcedureDescendants = "descendants";
const char *kProcedureTopologicalSort = "topological_sort";

const char *kReturnAncestors = "ancestors";
const char *kReturnConnections = "connections";
const char *kReturnDescendants = "descendants";
const char *kReturnSortedNodes = "sorted_nodes";

const char *kArgumentNode = "node";
const char *kArgumentNodes = "nodes";


extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    // Register ancestors procedure
    const auto ancestors_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(Ancestors, kProcedureAncestors, mgp::ProdecureType::Read, {mgp::Parameter(kArgumentNode, mgp::Type::Node)},
                 {mgp::Return(kReturnAncestors, ancestors_return)}, module, memory);

    // Register connect nodes procedure
    const auto connect_nodes_input = std::make_pair(mgp::Type::List, mgp::Type::Node);
    const auto connect_nodes_return = std::make_pair(mgp::Type::List, mgp::Type::Relationship);

    AddProcedure(ConnectNodes, kProcedureConnectNodes, mgp::ProdecureType::Read,
                 {mgp::Parameter(kArgumentNodes, connect_nodes_input)}, {mgp::Return(kReturnConnections, connect_nodes_return)},
                 module, memory);

    // Register descendants procedure
    const auto descendants_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(Descendants, kProcedureDescendants, mgp::ProdecureType::Read, {mgp::Parameter(kArgumentNode, mgp::Type::Node)},
                 {mgp::Return(kReturnDescendants, descendants_return)}, module, memory);

    // Register topological sort procedure
    const auto topological_sort_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(TopologicalSort, kProcedureTopologicalSort, mgp::ProdecureType::Read, {},
                 {mgp::Return(kReturnSortedNodes, topological_sort_return)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
