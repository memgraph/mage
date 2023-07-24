#include <mgp.hpp>

#include "algorithms/algorithms.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Collections::RemoveAll, Collections::kProcedureRemoveAll, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentsInputList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kArgumentsRemoveList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnRemoveAll, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
