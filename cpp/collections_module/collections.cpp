#include <mgp.hpp>

#include "algorithms/sort.hpp"

constexpr std::string_view kReturnSort = "sorted";

constexpr std::string_view kProcedureSort = "sort";

constexpr std::string_view kArgumentsInputList = "input_list";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;
  try {
    AddProcedure(Sort, kProcedureSort, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentsInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(kReturnSort, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
