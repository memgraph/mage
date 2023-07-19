#include <mgp.hpp>

#include "algorithms/sum.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(Sum, "sum", mgp::ProcedureType::Read, {mgp::Parameter("inputList", {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return("sum", mgp::Type::Double)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
