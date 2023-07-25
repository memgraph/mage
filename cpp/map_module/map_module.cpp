#include <mgp.hpp>

#include "algorithm/map.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Map::FromPairs, Map::kProcedureFromPairs, mgp::ProcedureType::Read,
                 {mgp::Parameter(Map::kArgumentsInputList, {mgp::Type::List, mgp::Type::List})},
                 {mgp::Return(Map::kReturnFromPairs, {mgp::Type::Map, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
