#include <mgp.hpp>

#include "algorithms/collections.hpp"




extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(unionAll, std::string(kProcedureUnionAll).c_str(), mgp::ProcedureType::Read,
    {mgp::Parameter(std::string(kArgumentList1).c_str(), {mgp::Type::List, mgp::Type::Any}), mgp::Parameter(std::string(kArgumentList2).c_str(), {mgp::Type::List, mgp::Type::Any})},
    {mgp::Return(std::string(kReturnValue).c_str(), {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }

