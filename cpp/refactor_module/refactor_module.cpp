#include <mgp.hpp>

#include "algorithm/refactor.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Refactor::CollapseNode, std::string(Refactor::kProcedureCollapseNode).c_str(),
                 mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Refactor::kArgumentNodesCollapseNode).c_str(), mgp::Type::Any),
                  mgp::Parameter(std::string(Refactor::kArgumentTypeCollapseNode).c_str(), mgp::Type::String)},
                 {mgp::Return(std::string(Refactor::kReturnIdCollapseNode).c_str(), mgp::Type::Int),
                  mgp::Return(std::string(Refactor::kReturnRelationshipCollapseNode).c_str(), mgp::Type::Relationship)},
                 module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
