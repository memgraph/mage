#include <mg_utils.hpp>
#include "algorithm/k_weighted_shortest_paths.hpp"
#include "mgp.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    AddProcedure(KWeightedShortestPaths::KWeightedShortestPaths, KWeightedShortestPaths::kProcedure,
                 mgp::ProcedureType::Read,
                 {mgp::Parameter(KWeightedShortestPaths::kArgumentStartNode, mgp::Type::Node),
                  mgp::Parameter(KWeightedShortestPaths::kArgumentEndNode, mgp::Type::Node),
                  mgp::Parameter(KWeightedShortestPaths::kArgumentNumberOfWeightedShortestPaths, mgp::Type::Int,
                                 mgp::Value(KWeightedShortestPaths::kDefaultNumberOfWeightedShortestPaths)),
                  mgp::Parameter(KWeightedShortestPaths::kArgumentWeightName, mgp::Type::String,
                                 mgp::Value(KWeightedShortestPaths::kDefaultWeightName))},

                 {}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
