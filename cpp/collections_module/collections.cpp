#include <mgp.hpp>

#include "algorithms/union.hpp"

constexpr std::string_view kReturnUnion = "union";

constexpr std::string_view kProcedureUnion = "union";

constexpr std::string_view kArgumentsInputList1 = "input_list1";
constexpr std::string_view kArgumentsInputList2 = "input_list2";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(Union, kProcedureUnion, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentsInputList1, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(kArgumentsInputList2, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(kReturnUnion, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
