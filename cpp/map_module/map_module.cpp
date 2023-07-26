#include <mgp.hpp>

#include "algorithm/map.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Map::FromNodes, Map::kProcedureFromNodes, mgp::ProcedureType::Read,
                 {mgp::Parameter(Map::kFromNodesArg1, mgp::Type::String), mgp::Parameter(Map::kFromNodesArg2, mgp::Type::String)},
                 {mgp::Return(Map::kResultFromNodes, mgp::Type::Map)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
