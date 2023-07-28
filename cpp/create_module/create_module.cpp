#include <mgp.hpp>

#include "algorithm/create.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    
    AddProcedure(Create::SetRelProperties, Create::kProcedureSetRelProperties, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kSetRelPropertiesArg1, mgp::Type::Any),
                  mgp::Parameter(Create::kSetRelPropertiesArg2, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(Create::kSetRelPropertiesArg3, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Create::kResultSetRelProperties, mgp::Type::Relationship)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
