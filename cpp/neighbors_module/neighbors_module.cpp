#include <mgp.hpp>

#include "algorithm/neighbors.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Neighbors::AtHop, Neighbors::kProcedureAtHop, mgp::ProcedureType::Read,
                 {mgp::Parameter(Neighbors::kArgumentsNode, mgp::Type::Node),
                  mgp::Parameter(Neighbors::kArgumentsRelType, mgp::Type::String),
                  mgp::Parameter(Neighbors::kArgumentsDistance, mgp::Type::Int)},
                 {mgp::Return(Neighbors::kReturnAtHop, mgp::Type::Node)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
