#include <mgp.hpp>

#include "algorithm/algorithm.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto parameter_list_of_any =
        mgp::Parameter(Collections::kAnyList, std::make_pair(mgp::Type::List, mgp::Type::Any));

    AddProcedure(Collections::Intersection, Collections::kProcedureIntersection, mgp::ProcedureType::Read,
                 {parameter_list_of_any, parameter_list_of_any},
                 {mgp::Return(Collections::kReturnIntersection, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
