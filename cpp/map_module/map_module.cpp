#include <mgp.hpp>

#include "algorithm/map.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Map::SetKey, Map::kProcedureSetKey, mgp::ProcedureType::Read,
                 {mgp::Parameter(Map::kSetKeyArg1, mgp::Type::Map), mgp::Parameter(Map::kSetKeyArg2, mgp::Type::String),
                  mgp::Parameter(Map::kSetKeyArg3, mgp::Type::Any)},
                 {mgp::Return(Map::kResultSetKey, mgp::Type::Map)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
