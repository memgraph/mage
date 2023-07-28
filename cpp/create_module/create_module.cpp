#include <mgp.hpp>

#include "algorithm/create.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Create::SetProperty, std::string(Create::kProcedureSetProperty).c_str(), mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Create::kArgumentNodeSetProperty).c_str(), mgp::Type::Node),
                  mgp::Parameter(std::string(Create::kArgumentKeySetProperty).c_str(), mgp::Type::String),
                  mgp::Parameter(std::string(Create::kArgumentValueSetProperty).c_str(), mgp::Type::Any)},
                 {mgp::Return(std::string(Create::kReturntSetProperty).c_str(), mgp::Type::Node)}, module, memory);


  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
