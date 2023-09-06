#include <mgp.hpp>

#include "algorithm/refactor.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Refactor::Invert, std::string(Refactor::kProcedureInvert).c_str(), mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Refactor::kArgumentRelationship).c_str(), mgp::Type::Any)},
                 {mgp::Return(std::string(Refactor::kReturnIdInvert).c_str(), mgp::Type::Int),
                  mgp::Return(std::string(Refactor::kReturnRelationshipInvert).c_str(), mgp::Type::Relationship)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
