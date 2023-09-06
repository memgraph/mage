#include <mgp.hpp>

#include "algorithm/refactor.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(
        Refactor::RenameLabel, Refactor::kProcedureRenameLabel, mgp::ProcedureType::Write,
        {mgp::Parameter(Refactor::kRenameLabelArg1, mgp::Type::String),
         mgp::Parameter(Refactor::kRenameLabelArg2, mgp::Type::String),
         mgp::Parameter(Refactor::kRenameLabelArg3, {mgp::Type::List, mgp::Type::Node}, mgp::Value(mgp::List{}))},
        {mgp::Return(Refactor::kRenameLabelResult, mgp::Type::Int)}, module, memory);

    AddProcedure(Refactor::RenameNodeProperty, Refactor::kProcedureRenameNodeProperty, mgp::ProcedureType::Write,
                 {mgp::Parameter(Refactor::kRenameNodePropertyArg1, mgp::Type::String),
                  mgp::Parameter(Refactor::kRenameNodePropertyArg2, mgp::Type::String),
                  mgp::Parameter(Refactor::kRenameNodePropertyArg3, {mgp::Type::List, mgp::Type::Node},
                                 mgp::Value(mgp::List{}))},
                 {mgp::Return(Refactor::kRenameNodePropertyResult, mgp::Type::Int)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
