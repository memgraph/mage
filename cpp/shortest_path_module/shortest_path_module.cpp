#include <mg_utils.hpp>
#include "algorithm/k_weighted_shortest_path.hpp"
#include "mgp.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    AddProcedure(KWeightedShortestPath::KWeightedShortestPath, KWeightedShortestPath::kProcedure,
                 mgp::ProcedureType::Read,
                 {mgp::Parameter(KWeightedShortestPath::kArgumentStartNode, mgp::Type::Node),
                  mgp::Parameter(KWeightedShortestPath::kArgumentEndNode, mgp::Type::Node),
                  mgp::Parameter(KWeightedShortestPath::kArgumentNumberOfWeightedShortestPaths, mgp::Type::Int,
                                 mgp::Value(KWeightedShortestPath::kDefaultNumberOfWeightedShortestPaths))},

                 {}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
}

extern "C" int mgp_shutdown_module() { return 0; }