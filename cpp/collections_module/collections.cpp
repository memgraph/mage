#include <mgp.hpp>
#include "algorithm/algorithm.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Collections::toSet, Collections::kProcedureToSet, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentListToSet, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnToSet, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
