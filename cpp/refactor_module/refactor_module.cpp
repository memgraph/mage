#include <mgp.hpp>

#include "algorithm/refactor.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Refactor::CloneNodes, Refactor::kProcedureCloneNodes, mgp::ProcedureType::Write,
                 {mgp::Parameter(Refactor::kArgumentsNodesToClone, {mgp::Type::List, mgp::Type::Node}),
                  mgp::Parameter(Refactor::kArgumentsCloneRels, mgp::Type::Bool, false),
                  mgp::Parameter(Refactor::kArgumentsSkipPropClone, {mgp::Type::List, mgp::Type::String},
                                 mgp::Value(mgp::List{}))},
                 {mgp::Return(Refactor::kReturnClonedNodeId, mgp::Type::Int),
                  mgp::Return(Refactor::kReturnNewNode, mgp::Type::Node)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }