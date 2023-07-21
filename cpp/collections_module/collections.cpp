#include <mgp.hpp>

#include "algorithms/containsSorted.hpp"

constexpr std::string_view kReturnCS = "contains";

constexpr std::string_view kProcedureCS = "containsSorted";

constexpr std::string_view kArgumentInputList = "input_list";
constexpr std::string_view kArgumentElement = "element";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(ContainsSorted, kProcedureCS, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentInputList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(kArgumentElement, mgp::Type::Any)},
                 {mgp::Return(kReturnCS, mgp::Type::Bool)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
