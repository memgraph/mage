#include <mgp.hpp>

#include "algorithm/refactor.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(
        Refactor::CloneSubgraphFromPaths, Refactor::kProcedureCSFP, mgp::ProcedureType::Write,
        {mgp::Parameter(Refactor::kArgumentsPath, {mgp::Type::List, mgp::Type::Path}),
         mgp::Parameter(Refactor::kArgumentsConfigMap, {mgp::Type::Map, mgp::Type::Any}, mgp::Value(mgp::Map{}))},
        {mgp::Return(Refactor::kReturnClonedNodeId, mgp::Type::Int),
         mgp::Return(Refactor::kReturnNewNode, mgp::Type::Node)},
        module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
