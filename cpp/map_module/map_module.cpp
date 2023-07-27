#include <mgp.hpp>

#include "algorithm/map.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Map::Merge, Map::kProcedureMerge, mgp::ProcedureType::Read,
                 {mgp::Parameter(Map::kArgumentsInputMap1, mgp::Type::Map),
                  mgp::Parameter(Map::kArgumentsInputMap2, mgp::Type::Map)},
                 {mgp::Return(Map::kReturnMerge, mgp::Type::Map)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
