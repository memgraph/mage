#include <mgp.hpp>

#include "algorithm/create.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Create::RemoveLabels, Create::kProcedureRemoveLabels, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kArgumentsNodes, mgp::Type::Any),
                  mgp::Parameter(Create::kArgumentsLabels, {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(Create::kReturnRemoveLabels, mgp::Type::Node)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
