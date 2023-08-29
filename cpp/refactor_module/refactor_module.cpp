#include <mgp.hpp>

#include "algorithm/refactor.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Refactor::Categorize, Refactor::kProcedureCategorize, mgp::ProcedureType::Write,
                 {
                     mgp::Parameter(Refactor::kArgumentsCatSourceKey, mgp::Type::String),
                     mgp::Parameter(Refactor::kArgumentsCatRelType, mgp::Type::String),
                     mgp::Parameter(Refactor::kArgumentsCatRelOutgoing, mgp::Type::Bool),
                     mgp::Parameter(Refactor::kArgumentsCatLabelName, mgp::Type::String),
                     mgp::Parameter(Refactor::kArgumentsCatPropKey, mgp::Type::String),
                     mgp::Parameter(Refactor::kArgumentsCopyPropKeys, {mgp::Type::List, mgp::Type::String},
                                    mgp::Value(mgp::List{})),
                 },
                 {mgp::Return(Refactor::kReturnCategorize, mgp::Type::String)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
