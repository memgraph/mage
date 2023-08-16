#include <mgp.hpp>

#include "algorithm/label.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Label::Exists, Label::kProcedureExists, mgp::ProcedureType::Read,
                 {mgp::Parameter(Label::kArgumentsNode, mgp::Type::Any),
                  mgp::Parameter(Label::kArgumentsLabel, mgp::Type::String)},
                 {mgp::Return(Label::kReturnExists, mgp::Type::Bool)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
