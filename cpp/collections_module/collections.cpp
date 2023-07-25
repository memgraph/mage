#include <mgp.hpp>

#include "algorithm/algorithm.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Collections::ContainsAll, Collections::kProcedureContainsAll, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kAnyList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kAnyList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnContainsAll, mgp::Type::Bool)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
