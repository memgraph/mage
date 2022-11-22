#include <mgp.hpp>

#include "algorithms/ancestors.hpp"
#include "algorithms/connect_nodes.hpp"
#include "algorithms/descendants.hpp"
#include "algorithms/topological_sort.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    // Register ancestors procedure
    const auto ancestors_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(Ancestors, "ancestors", mgp::ProdecureType::Read, {mgp::Parameter("node", mgp::Type::Node)},
                 {mgp::Return("ancestors", ancestors_return)}, module, memory);

    // Register connect nodes procedure
    const auto connect_nodes_input = std::make_pair(mgp::Type::List, mgp::Type::Node);
    const auto connect_nodes_return = std::make_pair(mgp::Type::List, mgp::Type::Relationship);

    AddProcedure(ConnectNodes, "connect_nodes", mgp::ProdecureType::Read,
                 {mgp::Parameter("nodes", connect_nodes_input)}, {mgp::Return("connections", connect_nodes_return)},
                 module, memory);

    // Register descendants procedure
    const auto descendants_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(Descendants, "descendants", mgp::ProdecureType::Read, {mgp::Parameter("node", mgp::Type::Node)},
                 {mgp::Return("descendants", descendants_return)}, module, memory);

    // Register topological sort procedure
    const auto topological_sort_return = std::make_pair(mgp::Type::List, mgp::Type::Node);

    AddProcedure(TopologicalSort, "topological_sort", mgp::ProdecureType::Read, {},
                 {mgp::Return("sorted_nodes", topological_sort_return)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
